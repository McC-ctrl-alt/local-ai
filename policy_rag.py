"""Policy-first RAG.

A question and a role are checked against policy.yaml. Departments come from
the words in that file, not from a second model call. Python allows or
refuses each one. Only allowed folders are opened. engine.py embeds them,
keeps the cosine top 10, reranks those 10, and abstains when the top score
is below -0.7. Otherwise one Qwen call answers from the top 6 passages.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

import engine

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "policy.yaml"
DEMO_PATH = ROOT / "demo_questions.json"
CACHE_PATH = ROOT / "policy_embeddings.json"
OLLAMA_URL = engine.OLLAMA_URL
EMBED_MODEL = engine.EMBED_MODEL
CHAT_MODEL = engine.CHAT_MODEL
CANDIDATE_COUNT = engine.CANDIDATE_COUNT
FINAL_CONTEXT_COUNT = engine.FINAL_CONTEXT_COUNT
RERANKER_MODEL = engine.RERANKER_MODEL
CACHE_SCHEMA = 1
_RERANKER = None

# Routing compares question words to policy.yaml. These words show up in
# almost every department, so they do not choose one.
_ROUTE_STOP = frozenset(
    {
        "a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "by",
        "with", "from", "that", "this", "is", "are", "be", "may", "only",
        "when", "how", "what", "does", "do", "did", "it", "its", "not", "any",
        "other", "into", "about", "than", "then", "must", "their", "them",
        "your", "our", "we", "you", "have", "has", "been", "after", "before",
        "which", "who", "whom", "whose", "why", "where", "will", "can",
        "could", "should", "would", "using", "use", "used", "each", "per",
        "via", "within", "without", "under", "over", "also", "such", "these",
        "those", "there", "here", "policy", "policies", "folder", "document",
        "documents", "controlled", "answer", "read", "role", "roles",
        "include", "including", "between", "during", "while", "both", "same",
    }
)


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load a policy file and reject a department that cannot be enforced.

    The file has one top-level key, ``departments``. Each department id
    maps to ``name``, ``folder``, ``description``, ``allowed_roles`` (role
    names), and ``rules``. A rule has ``id``, ``kind``, and ``text``.
    One rule in every department must use ``kind: access``.
    """
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


def rule_by_kind(department: dict[str, Any], kind: str) -> dict[str, Any] | None:
    """Return the first rule of this kind, or None when the department has none."""
    for rule in department["rules"]:
        if rule.get("kind") == kind:
            return rule
    return None


def decide_access(
    department: dict[str, Any], role: str
) -> tuple[bool, dict[str, Any] | None]:
    """Return whether this role may read the department, and the access rule."""
    rule = rule_by_kind(department, "access")
    allowed = role.strip() in department["allowed_roles"]
    return allowed, rule


def refusal_text(rule: dict[str, Any]) -> str:
    """Format the access rule as the refusal line shown to the user."""
    return f"Refused by {rule['id']}: {rule['text']}"


def _route_keep(policy: dict[str, Any]) -> set[str]:
    """Department ids and names stay matchable even when they are short."""
    keep = set()
    for department_id, department in policy["departments"].items():
        keep.add(department_id.casefold())
        keep.add(str(department["name"]).casefold())
    return keep


def _route_keys(text: str, keep_short: set[str]) -> set[str]:
    """Normalize words so 'deduction' and 'deductible' can meet."""
    keys = set()
    for word in re.findall(r"[a-z0-9]+", text.casefold()):
        if word in _ROUTE_STOP:
            continue
        if len(word) < 3 and word not in keep_short:
            continue
        original = word
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("es") and len(word) > 5 and not word.endswith(("ss", "us")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            word = word[:-1]
        if len(word) < 3 and original not in keep_short:
            continue
        if original in keep_short:
            keys.add(original)
        keys.add(word[:6])
    return keys


def _department_route_text(department_id: str, department: dict[str, Any]) -> str:
    """Policy text and file names for one department. File bodies stay closed."""
    parts = [department_id, department["name"], department["description"]]
    for rule in department["rules"]:
        if rule.get("kind") == "content":
            parts.append(rule["text"])
    folder = ROOT / department["folder"]
    if folder.is_dir():
        parts.extend(path.stem.replace("-", " ") for path in engine.list_pdfs(folder))
    return "\n".join(parts)


def _reason_for(
    department: dict[str, Any], question_keys: set[str], keep_short: set[str]
) -> str:
    """Use the policy sentence that shares the most words with the question."""
    candidates = [department["description"]]
    candidates.extend(
        rule["text"] for rule in department["rules"] if rule.get("kind") == "content"
    )
    best = candidates[0]
    best_overlap = len(_route_keys(best, keep_short) & question_keys)
    for text in candidates[1:]:
        overlap = len(_route_keys(text, keep_short) & question_keys)
        if overlap > best_overlap:
            best = text
            best_overlap = overlap
    return " ".join(str(best).split())


def route_question(policy: dict[str, Any], question: str) -> list[tuple[str, str]]:
    """Pick departments from policy.yaml. This does not call Qwen or open a file.

    A department needs two shared words, otherwise the single best match.
    Nothing is hardcoded to a question. No shared words means no department.
    """
    keep_short = _route_keep(policy)
    question_keys = _route_keys(question, keep_short)
    scored = []
    for department_id, department in policy["departments"].items():
        overlap = _route_keys(
            _department_route_text(department_id, department), keep_short
        ) & question_keys
        scored.append((len(overlap), department_id, department))
    chosen = [item for item in scored if item[0] >= 2]
    if not chosen:
        best = max((item[0] for item in scored), default=0)
        chosen = [item for item in scored if item[0] == best and best >= 1]
    return [
        (department_id, _reason_for(department, question_keys, keep_short))
        for _count, department_id, department in chosen
    ]


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


def parse_route(
    payload: str | dict[str, Any], department_ids: set[str]
) -> list[tuple[str, str]]:
    """Turn router JSON into unique (department id, reason) pairs."""
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


def split_access(
    policy: dict[str, Any], role: str, routes: list[tuple[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Allow or refuse each routed department. A refused folder stays closed."""
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


def ollama_json(
    messages: list[dict[str, str]], response_format: dict[str, Any], num_predict: int
) -> dict[str, Any]:
    """Call Qwen once through engine.chat_answer and return its JSON object."""
    system = ""
    user = ""
    for message in messages:
        if message.get("role") == "system":
            system = message.get("content", "")
        elif message.get("role") == "user":
            user = message.get("content", "")
    try:
        content = engine.chat_answer(system, user, response_format, num_predict)
    except RuntimeError as exc:
        raise SystemExit(
            f"Ollama is not running at {OLLAMA_URL}. "
            f"Start it, then pull {EMBED_MODEL} and {CHAT_MODEL}."
        ) from exc
    except (KeyError, TypeError) as error:
        raise ValueError("Qwen returned an empty response.") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen returned an empty response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen returned invalid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError("Qwen returned JSON that is not an object.")
    return parsed


def build_page_aware_chunks(
    pdf_reader: PdfReader, source: str, index_start: int = 0
) -> list[dict[str, Any]]:
    """Chunk with engine.py, then keep the file name and the sentences."""
    return engine.build_page_aware_chunks(
        pdf_reader, source=source, index_start=index_start
    )


def load_department_chunks(
    folder: Path | str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Read every PDF in a department folder into chunk records."""
    folder = Path(folder)
    pdfs = engine.list_pdfs(folder)
    chunks = []
    for pdf_path in pdfs:
        source = pdf_path.resolve().relative_to(ROOT).as_posix()
        reader = PdfReader(str(pdf_path))
        chunks.extend(
            build_page_aware_chunks(reader, source, index_start=len(chunks))
        )
    return pdfs, chunks


def folder_hash(pdfs: list[Path]) -> str:
    """Hash PDF names and bytes so a changed folder misses the embedding cache."""
    digest = hashlib.sha256()
    for pdf_path in pdfs:
        digest.update(pdf_path.name.encode())
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed each text with engine.embed, in input order."""
    if not texts:
        return []
    try:
        return [engine.embed(text) for text in texts]
    except RuntimeError as exc:
        raise SystemExit(
            f"Ollama is not running at {OLLAMA_URL}. "
            f"Start it, then pull {EMBED_MODEL} and {CHAT_MODEL}."
        ) from exc


def load_cache() -> dict[str, Any]:
    """Load the department embedding cache, or an empty cache when it is stale."""
    if not CACHE_PATH.exists():
        return {"schema": CACHE_SCHEMA, "model": EMBED_MODEL, "departments": {}}
    with open(CACHE_PATH, encoding="utf-8") as handle:
        cache = json.load(handle)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("model") != EMBED_MODEL:
        return {"schema": CACHE_SCHEMA, "model": EMBED_MODEL, "departments": {}}
    cache.setdefault("departments", {})
    return cache


def embeddings_for(
    department_id: str, pdf_hash: str, chunks: list[dict[str, Any]]
) -> tuple[list[list[float]], bool]:
    """Return cached chunk embeddings, or embed the chunks and store them."""
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


def rank_by_cosine(
    question: str, chunks: list[dict[str, Any]], embeddings: list[list[float]]
) -> list[tuple[float, dict[str, Any]]]:
    """Cosine top 10 from engine.rank_candidates, highest first."""
    ranked = engine.rank_candidates(
        question, {"chunks": chunks, "embeddings": embeddings}
    )
    return [(score, chunk) for score, _chunk_index, chunk in ranked]


def order_reranked(
    candidates: list[tuple[float, dict[str, Any]]],
    reranker_scores: Sequence[float],
) -> list[tuple[float, float, dict[str, Any]]]:
    """Reorder cosine candidates by the cross-encoder score, highest first."""
    ordered = [
        (float(reranker_score), cosine_score, chunk)
        for reranker_score, (cosine_score, chunk) in zip(reranker_scores, candidates)
    ]
    ordered.sort(key=lambda item: item[0], reverse=True)
    return ordered


def get_reranker() -> Any:
    """Load the engine's CPU cross-encoder once. Try ONNX, then PyTorch."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        _RERANKER = engine.load_reranker()
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Reranker model '{RERANKER_MODEL}' is not available locally. "
            "Download it before running offline."
        ) from exc
    return _RERANKER


def rerank_candidates(
    question: str, candidates: list[tuple[float, dict[str, Any]]]
) -> list[tuple[float, float, dict[str, Any]]]:
    """Rerank the cosine top 10 with engine.rerank_candidates."""
    if not candidates:
        return []
    adapted = [
        (cosine_score, index, chunk)
        for index, (cosine_score, chunk) in enumerate(candidates)
    ]
    reranked = engine.rerank_candidates(question, adapted, get_reranker())
    return [
        (score, cosine_score, chunk)
        for score, cosine_score, _index, chunk in reranked
    ]


def format_hit(score: float, chunk: dict[str, Any]) -> str:
    """Format one ranked chunk as id, pages, score, and file name."""
    pages = ",".join(str(page) for page in chunk["page_numbers"])
    name = Path(chunk["source"]).name
    return f"{chunk['chunk_id']} p.{pages} {score:8.4f} {name}"


def print_orders(
    candidates: list[tuple[float, dict[str, Any]]],
    reranked: list[tuple[float, float, dict[str, Any]]],
) -> None:
    """Print the cosine order beside the reranked order."""
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


def print_citations(result: dict[str, Any]) -> None:
    """Print citation problems, then each accepted citation."""
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


def evidence_from(selected_chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Give each selected sentence its own evidence id.

    A citation is then the line that states the fact. Chunk boundaries stay
    the same as curator.py.
    """
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


def normalize_text(text: str) -> str:
    """Collapse whitespace and case so citation checks ignore formatting."""
    return " ".join(str(text).casefold().split())


def is_abstention(answer: str) -> bool:
    """Return whether the answer is exactly "I don't know", period optional."""
    return normalize_text(answer).rstrip(".") == "i don't know"


def validate_citations(
    answer: str,
    citations: list[dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Check that every cited quote, number, and date is on the cited page."""
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


def answer_question(
    question: str,
    department: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    selected_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask Qwen once for an answer grounded in the top reranked passages."""
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


def known_roles(policy: dict[str, Any] | None = None) -> list[str]:
    """Roles named in policy.yaml, in department order, without duplicates."""
    if policy is None:
        policy = load_policy()
    roles = []
    for department in policy["departments"].values():
        for role in department["allowed_roles"]:
            if role not in roles:
                roles.append(role)
    return roles


def unknown_role_message(role: str, policy: dict[str, Any] | None = None) -> str:
    """Explain a role that policy.yaml does not list."""
    listed = ", ".join(known_roles(policy))
    return f"Unknown role {role!r}. Roles in policy.yaml: {listed}."


def trail_parts(
    role: str, decisions: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Role, rules, departments, and cited files for the trail."""
    rule_bits = []
    for item in decisions:
        rule_bits.append(
            f"{item['rule']['id']} {'allow' if item['allowed'] else 'refuse'}"
        )
        if item["allowed"]:
            scope = rule_by_kind(item["department"], "scope")
            if scope:
                rule_bits.append(f"{scope['id']} {item['department']['folder']} only")
    departments = []
    for item in decisions:
        bit = f"{item['department_id']} — {item['reason']}"
        if not item["allowed"]:
            bit += f" (refused by {item['rule']['id']})"
        departments.append(bit)
    cited = []
    for citation in citations:
        label = f"{Path(citation['source']).name} p. {citation['page_number']}"
        if label not in cited:
            cited.append(label)
    return {
        "role": role,
        "rules": rule_bits,
        "departments": departments,
        "cited": cited,
    }


def print_trail(
    role: str, decisions: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> None:
    """Print the role, the rules applied, the departments, and the citations."""
    parts = trail_parts(role, decisions, citations)
    print("\n--- Trail ---")
    print(f"Role: {parts['role']}")
    print("Rules: " + "; ".join(parts["rules"]))
    print("Departments: " + "; ".join(parts["departments"]))
    print("Cited: " + ("; ".join(parts["cited"]) if parts["cited"] else "none"))


def consult(policy: dict[str, Any], role: str, question: str) -> dict[str, Any]:
    """Route, enforce access, retrieve, and answer. Nothing is printed.

    Qwen is called once, and only when a folder is opened and the top rerank
    score is not below the cutoff. A refused role or an unmatched question
    does not call it.
    """
    routes = route_question(policy, question)
    decisions, allowed_rows, denied_rows = split_access(policy, role, routes)
    result: dict[str, Any] = {
        "role": role,
        "question": question,
        "decisions": decisions,
        "allowed_rows": allowed_rows,
        "denied_rows": denied_rows,
        "opened_names": [],
        "not_opened": [],
        "candidates": [],
        "reranked": [],
        "selected": [],
        "top_score": None,
        "model_result": None,
        "answer": "",
        "citations": [],
        "seconds": None,
        "stage": "answered",
    }
    if not allowed_rows:
        if denied_rows:
            result["stage"] = "refused"
            result["answer"] = " ".join(
                refusal_text(item["rule"]) for item in denied_rows
            )
        else:
            result["stage"] = "no-route"
            result["answer"] = "I don't know."
        return result

    chunks = []
    embeddings = []
    for item in allowed_rows:
        folder = ROOT / item["department"]["folder"]
        if not folder.is_dir():
            raise SystemExit(f"Missing department folder: {item['department']['folder']}")
        pdfs, dept_chunks = load_department_chunks(folder)
        result["opened_names"].extend(pdf.name for pdf in pdfs)
        if not dept_chunks:
            continue
        dept_embeddings, _from_cache = embeddings_for(
            item["department_id"], folder_hash(pdfs), dept_chunks
        )
        chunks.extend(dept_chunks)
        embeddings.extend(dept_embeddings)

    opened_ids = {item["department_id"] for item in allowed_rows}
    result["not_opened"] = [
        department["folder"]
        for department_id, department in policy["departments"].items()
        if department_id not in opened_ids
    ]
    if not chunks:
        result["stage"] = "no-chunks"
        result["answer"] = "I don't know."
        return result

    cosine_ranked = rank_by_cosine(question, chunks, embeddings)
    candidates = cosine_ranked[:CANDIDATE_COUNT]
    reranked = rerank_candidates(question, candidates)
    selected = [
        chunk for _rerank_score, _cosine_score, chunk in reranked[:FINAL_CONTEXT_COUNT]
    ]
    top_score = reranked[0][0] if reranked else None
    result["candidates"] = candidates
    result["reranked"] = reranked
    result["selected"] = selected
    result["top_score"] = top_score
    if engine.rerank_below_cutoff(top_score):
        result["stage"] = "cutoff"
        result["answer"] = "I don't know."
        return result

    names = " and ".join(item["department"]["name"] for item in allowed_rows)
    evidence = evidence_from(selected)
    started = time.perf_counter()
    model_result = answer_question(question, {"name": names}, evidence, selected)
    result["seconds"] = time.perf_counter() - started
    result["model_result"] = model_result
    result["answer"] = model_result["delivered"]
    result["citations"] = model_result["citations"]
    return result


def _print_consult(result: dict[str, Any]) -> None:
    """Print consult() in the same shape policy_rag.py has always used."""
    for item in result["denied_rows"]:
        print(refusal_text(item["rule"]))
    stage = result["stage"]
    if stage == "refused":
        print("No documents opened. Answer call skipped.")
        print_trail(result["role"], result["decisions"], [])
        return
    if stage == "no-route":
        print("Answer: I don't know.")
        print("No documents opened. Answer call skipped.")
        print_trail(result["role"], result["decisions"], [])
        return

    print("Opened: " + ", ".join(result["opened_names"]))
    print("Not opened: " + ", ".join(result["not_opened"]))
    if stage == "no-chunks":
        print("Answer: I don't know.")
        print_trail(result["role"], result["decisions"], [])
        return

    print_orders(result["candidates"], result["reranked"])
    if stage == "cutoff":
        print(
            f"Top rerank score {result['top_score']:.4f} is below {engine.RERANK_ABSTAIN_BELOW}."
        )
        print("Answer: I don't know.")
        print_trail(result["role"], result["decisions"], [])
        return

    print(f"One {CHAT_MODEL} call uses reranked top {len(result['selected'])}.")
    model_result = result["model_result"]
    print(f"\nAnswer: {model_result['delivered']}")
    print_citations(model_result)
    print(f"Generation time: {result['seconds']:.2f}s")
    print_trail(result["role"], result["decisions"], model_result["citations"])


def run_question(
    policy: dict[str, Any], role: str, question: str, case_id: str | None = None
) -> int:
    """Route one question, enforce access, retrieve, and answer it."""
    label = f" {case_id}" if case_id else ""
    print(f"\n=== Question{label} ===")
    print(f"Role: {role}")
    print(f"Question: {question}")

    try:
        result = consult(policy, role, question)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Router failed ({exc}). No documents opened.")
        return 1
    _print_consult(result)
    return 0


def load_demo_questions() -> list[dict[str, Any]]:
    """Load the demo questions from demo_questions.json."""
    with open(DEMO_PATH, encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list) or not questions:
        raise ValueError("demo_questions.json must be a non-empty list.")
    return questions


def main(argv: list[str] | None = None) -> int:
    """Run one demo case, one role and question, or the full demo set."""
    parser = argparse.ArgumentParser(
        description="Policy-first department RAG.",
        usage="%(prog)s [-h] [--case CASE | --role ROLE --question QUESTION]",
    )
    case_or_role = parser.add_mutually_exclusive_group()
    case = case_or_role.add_argument(
        "--case",
        default=None,
        help="Run one demo question by id.",
    )
    case_or_role.add_argument(
        "--role",
        default=None,
        help="Ask one question as this role.",
    )
    # --question is the other half of the --role pair. It is exclusive with
    # --case, and it stays valid together with --role.
    case_or_question = parser.add_mutually_exclusive_group()
    case_or_question._group_actions.append(case)
    case_or_question.add_argument(
        "--question",
        default=None,
        help="Ask one question instead of the demo set.",
    )
    args = parser.parse_args(argv)

    if (args.role is None) != (args.question is None):
        parser.error("--role and --question are used together.")

    policy = load_policy()
    if args.question is not None:
        return run_question(policy, args.role, args.question)

    questions = load_demo_questions()
    if args.case is not None:
        questions = [item for item in questions if item.get("id") == args.case]
        if not questions:
            raise SystemExit(f"No demo question with id {args.case}.")

    status = 0
    for item in questions:
        status = max(status, run_question(policy, item["role"], item["question"], item["id"]))
    return status


if __name__ == "__main__":
    sys.exit(main())
