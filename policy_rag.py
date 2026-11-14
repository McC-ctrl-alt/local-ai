"""Policy-first RAG.

A question and a role are checked against policy.yaml. One short Qwen call
lists the departments the question needs. Python allows or refuses each one.
Only allowed folders are opened. Cosine keeps the top 10 chunks, the
cross-encoder reranker reorders them, and one qwen2.5:3b call answers from
the top 6 reranked passages with page citations.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "policy.yaml"
DEMO_PATH = ROOT / "demo_questions.json"
CACHE_PATH = ROOT / "policy_embeddings.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"
# Same knobs and model as curator.py.
CANDIDATE_COUNT = int(os.environ.get("RAG_CANDIDATE_COUNT", "10"))
FINAL_CONTEXT_COUNT = int(os.environ.get("RAG_FINAL_CONTEXT_COUNT", "6"))
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
CACHE_SCHEMA = 1
_RERANKER = None


def load_policy(path=POLICY_PATH):
    with open(path, encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    departments = policy.get("departments") if isinstance(policy, dict) else None
    if not isinstance(departments, dict) or not departments:
        raise ValueError("policy.yaml must define departments.")
    for dept_id, department in departments.items():
        for key in ("name", "folder", "description", "allowed_roles", "rules"):
            if key not in department:
                raise ValueError(f"{dept_id} is missing {key}.")
        if not any(rule.get("kind") == "access" for rule in department["rules"]):
            raise ValueError(f"{dept_id} needs an access rule.")
    return policy


def rule_by_kind(department, kind):
    for rule in department["rules"]:
        if rule.get("kind") == kind:
            return rule
    return None


def decide_access(department, role):
    rule = rule_by_kind(department, "access")
    allowed = role.strip() in department["allowed_roles"]
    return allowed, rule


def refusal_text(rule):
    return f"Refused by {rule['id']}: {rule['text']}"


def router_prompt(policy, question):
    lines = [
        "Return the smallest set of departments that own this question.",
        "Include a department only when its description matches the question.",
        "If two descriptions match, list both. If one matches, list only that one.",
        "Each reason is a short topic name. Do not answer the question.",
        "Do not include a number, a date, or a rule.",
        "Departments:",
    ]
    for dept_id, department in policy["departments"].items():
        lines.append(f"- {dept_id}: {department['description']}")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def _route_entry(item, department_ids, fallback_reason):
    if isinstance(item, str):
        department = item.strip().lower()
        reason = fallback_reason
    elif isinstance(item, dict):
        department = str(item.get("department", "")).strip().lower()
        reason = " ".join(str(item.get("reason", fallback_reason)).split())
    else:
        raise ValueError("Router returned a department entry that is not an object.")
    if department not in department_ids:
        raise ValueError(
            f"Router returned an unknown department: {department or '(empty)'}."
        )
    return department, reason or "No reason given."


def parse_route(payload, department_ids):
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        data = json.loads(text)
    else:
        data = payload
    if not isinstance(data, dict):
        raise ValueError("Router returned a value that is not an object.")

    fallback_reason = " ".join(str(data.get("reason", "")).split())
    if "departments" in data:
        raw = data["departments"]
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ValueError("Router returned no departments.")
        entries = raw
    elif "department" in data:
        entries = [data]
    else:
        raise ValueError("Router returned no departments.")

    routes = []
    seen = set()
    for item in entries:
        department, reason = _route_entry(item, department_ids, fallback_reason)
        if department in seen:
            continue
        seen.add(department)
        routes.append((department, reason))
    return routes


def split_access(policy, role, routes):
    decisions = []
    for department_id, reason in routes:
        department = policy["departments"][department_id]
        allowed, rule = decide_access(department, role)
        decisions.append(
            {
                "department_id": department_id,
                "reason": reason,
                "department": department,
                "rule": rule,
                "allowed": allowed,
            }
        )
    allowed_rows = [item for item in decisions if item["allowed"]]
    denied_rows = [item for item in decisions if not item["allowed"]]
    return decisions, allowed_rows, denied_rows


def ollama_json(messages, response_format, num_predict):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "format": response_format,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0, "num_predict": num_predict},
            },
            timeout=300,
        )
    except requests.ConnectionError as exc:
        raise SystemExit(
            f"Ollama is not running at {OLLAMA_URL}. "
            f"Start it, then pull {EMBED_MODEL} and {CHAT_MODEL}."
        ) from exc
    response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def route_question(policy, question):
    department_ids = list(policy["departments"])
    schema = {
        "type": "object",
        "properties": {
            "departments": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "enum": department_ids},
                        "reason": {"type": "string"},
                    },
                    "required": ["department", "reason"],
                },
            }
        },
        "required": ["departments"],
    }
    parsed = ollama_json(
        [
            {
                "role": "system",
                "content": (
                    "Return only departments whose descriptions match the question. "
                    "A reason is a topic name, not an answer."
                ),
            },
            {"role": "user", "content": router_prompt(policy, question)},
        ],
        schema,
        num_predict=140,
    )
    return parse_route(parsed, set(department_ids))


def make_chunk_record(sentences, chunk_index, source):
    # Same record as curator.py, plus the file the pages came from.
    chunk_text = " ".join(sentence for sentence, _page_number in sentences)
    page_sentences = {}
    for sentence, page_number in sentences:
        page_sentences.setdefault(page_number, []).append(sentence)
    page_segments = [
        {
            "page_number": page_number,
            "text": " ".join(page_sentences[page_number]),
        }
        for page_number in sorted(page_sentences)
    ]
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:10]
    return {
        "chunk_id": f"chunk-{chunk_index:03d}-{content_hash}",
        "source": source,
        "page_numbers": [segment["page_number"] for segment in page_segments],
        "page_segments": page_segments,
        "sentences": [
            {"page_number": page_number, "text": sentence}
            for sentence, page_number in sentences
        ],
        "text": chunk_text,
    }


def build_page_aware_chunks(pdf_reader, source, index_start=0):
    # Keep sentence/page pairs while chunking, including the sentence overlap.
    page_sentences = []
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\n", " ")
        page_sentences.extend(
            (sentence.strip(), page_number)
            for sentence in page_text.split(". ")
            if sentence.strip()
        )

    page_chunks = []
    current_sentences = []

    for sentence, page_number in page_sentences:
        current_text = " ".join(item[0] for item in current_sentences)
        if current_sentences and len(current_text) + 1 + len(sentence) > 1000:
            page_chunks.append(
                make_chunk_record(
                    current_sentences,
                    index_start + len(page_chunks),
                    source,
                )
            )

            overlap = []
            overlap_length = 0
            for previous_sentence in reversed(current_sentences):
                previous_length = len(previous_sentence[0])
                if overlap_length + previous_length > 300:
                    break
                overlap.insert(0, previous_sentence)
                overlap_length += previous_length
            current_sentences = overlap

        current_sentences.append((sentence, page_number))

    if current_sentences:
        page_chunks.append(
            make_chunk_record(
                current_sentences,
                index_start + len(page_chunks),
                source,
            )
        )

    return page_chunks


def load_department_chunks(folder):
    folder = Path(folder)
    pdfs = sorted(folder.glob("*.pdf"))
    chunks = []
    for pdf_path in pdfs:
        source = pdf_path.resolve().relative_to(ROOT).as_posix()
        reader = PdfReader(str(pdf_path))
        chunks.extend(
            build_page_aware_chunks(reader, source, index_start=len(chunks))
        )
    return pdfs, chunks


def folder_hash(pdfs):
    digest = hashlib.sha256()
    for pdf_path in pdfs:
        digest.update(pdf_path.name.encode())
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    return dot_product / (magnitude_a * magnitude_b)


def embed_texts(texts):
    vectors = []
    try:
        for text in texts:
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
                timeout=300,
            )
            response.raise_for_status()
            vectors.append(response.json()["embeddings"][0])
    except requests.ConnectionError as exc:
        raise SystemExit(
            f"Ollama is not running at {OLLAMA_URL}. "
            f"Start it, then pull {EMBED_MODEL} and {CHAT_MODEL}."
        ) from exc
    return vectors


def load_cache():
    if not CACHE_PATH.exists():
        return {"schema": CACHE_SCHEMA, "model": EMBED_MODEL, "departments": {}}
    with open(CACHE_PATH, encoding="utf-8") as handle:
        cache = json.load(handle)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("model") != EMBED_MODEL:
        return {"schema": CACHE_SCHEMA, "model": EMBED_MODEL, "departments": {}}
    cache.setdefault("departments", {})
    return cache


def embeddings_for(department_id, pdf_hash, chunks):
    cache = load_cache()
    entry = cache["departments"].get(department_id, {})
    reusable = (
        entry.get("pdf_hash") == pdf_hash
        and entry.get("chunks") == chunks
        and len(entry.get("embeddings", [])) == len(chunks)
    )
    if reusable:
        return entry["embeddings"], True

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    cache["departments"][department_id] = {
        "pdf_hash": pdf_hash,
        "chunks": chunks,
        "embeddings": embeddings,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as handle:
        json.dump(cache, handle)
    return embeddings, False


def rank_by_cosine(question, chunks, embeddings):
    question_vector = embed_texts([question])[0]
    scored = [
        (cosine_similarity(question_vector, embedding), chunk)
        for chunk, embedding in zip(chunks, embeddings)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def order_reranked(candidates, reranker_scores):
    # candidates are (cosine_score, chunk) in cosine order, as in curator.py.
    ordered = [
        (float(reranker_score), cosine_score, chunk)
        for reranker_score, (cosine_score, chunk) in zip(reranker_scores, candidates)
    ]
    ordered.sort(key=lambda item: item[0], reverse=True)
    return ordered


def get_reranker():
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import warnings

    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    try:
        from sentence_transformers import CrossEncoder
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc
    try:
        _RERANKER = CrossEncoder(
            RERANKER_MODEL,
            backend="onnx",
            device="cpu",
            local_files_only=False,
        )
    except Exception:
        print(
            "ONNX reranker extra is not installed. "
            f"Loading {RERANKER_MODEL} with PyTorch."
        )
        try:
            _RERANKER = CrossEncoder(
                RERANKER_MODEL,
                backend="torch",
                device="cpu",
                local_files_only=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Reranker model '{RERANKER_MODEL}' is not available locally. "
                "Download it before running offline."
            ) from exc
    return _RERANKER


def rerank_candidates(question, candidates):
    pairs = [(question, chunk["text"]) for _cosine_score, chunk in candidates]
    scores = get_reranker().predict(pairs, batch_size=8, show_progress_bar=False)
    return order_reranked(candidates, scores)


def format_hit(score, chunk):
    pages = ",".join(str(page) for page in chunk["page_numbers"])
    name = Path(chunk["source"]).name
    return f"{chunk['chunk_id']} p.{pages} {score:8.4f} {name}"


def print_orders(candidates, reranked):
    rows = []
    for cosine_item, rerank_item in zip(candidates, reranked):
        cosine_score, cosine_chunk = cosine_item
        rerank_score, _cosine_score, rerank_chunk = rerank_item
        rows.append(
            (format_hit(cosine_score, cosine_chunk), format_hit(rerank_score, rerank_chunk))
        )
    header = "cosine: chunk, page, score"
    width = max([len(header), *(len(left) for left, _right in rows)])
    print(f"\nCosine top {len(candidates)} beside the reranked order")
    print(f"{'#':>2}  {header:<{width}} | reranked: chunk, page, score")
    for rank, (left, right) in enumerate(rows, start=1):
        print(f"{rank:>2}  {left:<{width}} | {right}")


def print_citations(result):
    if not result["valid"]:
        print(f"Model answer: {result['answer']}")
        for problem in result["problems"]:
            print(f"Citation issue: {problem}")
    for citation in result["citations"]:
        quote = " ".join(citation["quote"].split())
        if len(quote) > 240:
            quote = quote[:240].rstrip() + "..."
        print(
            f"Citation: {citation['source']} p. {citation['page_number']} | "
            f"{citation['chunk_id']} | \"{quote}\""
        )


def evidence_from(selected_chunks):
    # One excerpt per sentence so a citation is the line that states the fact.
    # Chunk boundaries stay the same as curator.py.
    evidence = {}
    for chunk in selected_chunks:
        units = chunk.get("sentences") or [
            {"page_number": segment["page_number"], "text": segment["text"]}
            for segment in chunk["page_segments"]
        ]
        for unit in units:
            if not unit["text"].strip():
                continue
            evidence_id = f"E{len(evidence) + 1}"
            evidence[evidence_id] = {
                "chunk_id": chunk["chunk_id"],
                "page_number": unit["page_number"],
                "quote": unit["text"],
                "source": chunk["source"],
            }
    return evidence


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def is_abstention(answer):
    return normalize_text(answer).rstrip(".") == "i don't know"


def validate_citations(answer, citations, selected_chunks):
    if normalize_text(answer).startswith("i don't know"):
        if citations:
            return False, ["An I don't know answer cannot include a citation."]
        return True, []
    if not citations:
        return False, ["Answer has no citation."]

    selected_by_id = {chunk["chunk_id"]: chunk for chunk in selected_chunks}
    problems = []
    quoted_numbers = set()
    quoted_text = []
    for citation in citations:
        chunk = selected_by_id.get(citation.get("chunk_id"))
        page_number = citation.get("page_number")
        quote = citation.get("quote", "")
        if chunk is None:
            problems.append(f"Unknown chunk ID: {citation.get('chunk_id')}")
            continue
        if page_number not in chunk["page_numbers"]:
            problems.append(f"Page {page_number} is not part of {chunk['chunk_id']}.")
            continue
        page_text = " ".join(
            segment["text"]
            for segment in chunk["page_segments"]
            if segment["page_number"] == page_number
        )
        sentence_hit = any(
            item["page_number"] == page_number
            and normalize_text(item["text"]) == normalize_text(quote)
            for item in chunk.get("sentences", [])
        )
        if not sentence_hit and normalize_text(quote) not in normalize_text(page_text):
            problems.append(f"Quote was not found on page {page_number}.")
            continue
        quoted_numbers.update(re.findall(r"\d+", quote))
        quoted_text.append(quote)

    missing_numbers = [
        number
        for number in dict.fromkeys(re.findall(r"\d+", answer))
        if number not in quoted_numbers
    ]
    if missing_numbers:
        problems.append(
            "Answer uses numbers that are not in the cited pages: "
            + ", ".join(missing_numbers)
        )
    cited = normalize_text(" ".join(quoted_text))
    for month, day in re.findall(
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})\b",
        answer,
        flags=re.IGNORECASE,
    ):
        if f"{month.casefold()} {day}" not in cited:
            problems.append(f"Cited pages do not state {month} {day}.")
    return not problems, problems


def answer_question(question, department, evidence, selected_chunks):
    if not evidence:
        return {
            "answer": "I don't know.",
            "delivered": "I don't know.",
            "citations": [],
            "valid": True,
            "problems": [],
        }

    context = "\n\n".join(
        f"[{evidence_id}] {item['source']} p. {item['page_number']}\n{item['quote']}"
        for evidence_id, item in evidence.items()
    )
    prompt = f"""Answer the question in one sentence using only the numbered excerpts.
Return only the evidence IDs that state the fact you use.
If one excerpt states the fact, answer from it even when another excerpt says that file does not cover the topic.
Every number, and any month with a day, in your answer must appear in a cited excerpt.
If only part of the question is supported, answer that part and say you do not know the rest.
If the excerpts do not state the asked fact, answer exactly "I don't know." and return an empty evidence_ids list.
Do not answer a different question the excerpts happen to mention.

Question:
{question}

Evidence:
{context}
"""
    schema = {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(evidence)},
            },
            "answer": {"type": "string"},
        },
        "required": ["evidence_ids", "answer"],
    }
    parsed = ollama_json(
        [
            {
                "role": "system",
                "content": (
                    f"Use only the supplied {department['name']} evidence. "
                    "Do not use any other department."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        schema,
        num_predict=220,
    )
    answer = parsed.get("answer", "")
    evidence_ids = parsed.get("evidence_ids")
    if not isinstance(answer, str) or not answer.strip():
        return {
            "answer": "",
            "delivered": "I don't know. (Citation validation failed.)",
            "citations": [],
            "valid": False,
            "problems": ["Qwen did not return an answer."],
        }
    answer = answer.strip()
    if is_abstention(answer):
        return {
            "answer": "I don't know.",
            "delivered": "I don't know.",
            "citations": [],
            "valid": True,
            "problems": [],
        }
    if not isinstance(evidence_ids, list) or any(
        not isinstance(evidence_id, str) or evidence_id not in evidence
        for evidence_id in evidence_ids
    ):
        return {
            "answer": answer,
            "delivered": "I don't know. (Citation validation failed.)",
            "citations": [],
            "valid": False,
            "problems": ["Qwen returned invalid evidence IDs."],
        }

    citations = [evidence[evidence_id] for evidence_id in dict.fromkeys(evidence_ids)]
    valid, problems = validate_citations(answer, citations, selected_chunks)
    delivered = answer if valid else "I don't know. (Citation validation failed.)"
    return {
        "answer": answer,
        "delivered": delivered,
        "citations": citations if valid else [],
        "valid": valid,
        "problems": problems,
    }


def print_trail(role, decisions, citations):
    print("\n--- Trail ---")
    print(f"Role: {role}")
    rule_bits = []
    for item in decisions:
        rule_bits.append(
            f"{item['rule']['id']} {'allow' if item['allowed'] else 'refuse'}"
        )
        if item["allowed"]:
            scope = rule_by_kind(item["department"], "scope")
            if scope:
                rule_bits.append(f"{scope['id']} {item['department']['folder']} only")
    print("Rules: " + "; ".join(rule_bits))
    chosen = []
    for item in decisions:
        bit = f"{item['department_id']} — {item['reason']}"
        if not item["allowed"]:
            bit += f" (refused by {item['rule']['id']})"
        chosen.append(bit)
    print("Departments: " + "; ".join(chosen))
    cited = []
    for citation in citations:
        label = f"{Path(citation['source']).name} p. {citation['page_number']}"
        if label not in cited:
            cited.append(label)
    print("Cited: " + ("; ".join(cited) if cited else "none"))


def run_question(policy, role, question, case_id=None):
    label = f" {case_id}" if case_id else ""
    print(f"\n=== Question{label} ===")
    print(f"Role: {role}")
    print(f"Question: {question}")

    try:
        routes = route_question(policy, question)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Router failed ({exc}). No documents opened.")
        return 1

    decisions, allowed_rows, denied_rows = split_access(policy, role, routes)
    for item in denied_rows:
        print(refusal_text(item["rule"]))
    if not allowed_rows:
        print("No documents opened. Answer call skipped.")
        print_trail(role, decisions, [])
        return 0

    chunks = []
    embeddings = []
    opened_names = []
    for item in allowed_rows:
        folder = ROOT / item["department"]["folder"]
        if not folder.is_dir():
            raise SystemExit(f"Missing department folder: {item['department']['folder']}")
        pdfs, dept_chunks = load_department_chunks(folder)
        opened_names.extend(pdf.name for pdf in pdfs)
        if not dept_chunks:
            continue
        dept_embeddings, _from_cache = embeddings_for(
            item["department_id"], folder_hash(pdfs), dept_chunks
        )
        chunks.extend(dept_chunks)
        embeddings.extend(dept_embeddings)

    opened_ids = {item["department_id"] for item in allowed_rows}
    not_opened = [
        department["folder"]
        for department_id, department in policy["departments"].items()
        if department_id not in opened_ids
    ]
    print("Opened: " + ", ".join(opened_names))
    print("Not opened: " + ", ".join(not_opened))

    if not chunks:
        print("Answer: I don't know.")
        print_trail(role, decisions, [])
        return 0

    cosine_ranked = rank_by_cosine(question, chunks, embeddings)
    candidates = cosine_ranked[:CANDIDATE_COUNT]
    reranked = rerank_candidates(question, candidates)
    selected = [
        chunk for _rerank_score, _cosine_score, chunk in reranked[:FINAL_CONTEXT_COUNT]
    ]
    print_orders(candidates, reranked)
    print(f"One {CHAT_MODEL} call uses reranked top {len(selected)}.")

    names = " and ".join(item["department"]["name"] for item in allowed_rows)
    evidence = evidence_from(selected)
    started = time.perf_counter()
    result = answer_question(question, {"name": names}, evidence, selected)
    seconds = time.perf_counter() - started
    print(f"\nAnswer: {result['delivered']}")
    print_citations(result)
    print(f"Generation time: {seconds:.2f}s")
    print_trail(role, decisions, result["citations"])
    return 0


def load_demo_questions():
    with open(DEMO_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list) or not questions:
        raise ValueError("demo_questions.json must be a non-empty list.")
    return questions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Policy-first department RAG.")
    parser.add_argument("--role", help="Ask one question as this role.")
    parser.add_argument("--question", help="Ask one question instead of the demo set.")
    parser.add_argument("--case", help="Run one demo question by id.")
    args = parser.parse_args(argv)

    if bool(args.role) != bool(args.question):
        parser.error("--role and --question are used together.")
    if args.case and args.question:
        parser.error("Use either --case or --role/--question.")

    policy = load_policy()
    if args.question:
        return run_question(policy, args.role, args.question)

    questions = load_demo_questions()
    if args.case:
        questions = [item for item in questions if item.get("id") == args.case]
        if not questions:
            raise SystemExit(f"No demo question with id {args.case}.")

    status = 0
    for item in questions:
        status = max(status, run_question(policy, item["role"], item["question"], item["id"]))
    return status


if __name__ == "__main__":
    sys.exit(main())
