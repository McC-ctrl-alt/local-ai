# Project Curator
"""Answer one question from every PDF in a folder, using local Qwen.

Embeddings and the written answer stay on this computer through Ollama.
A citation is delivered only when its quote is present on the named file
and page. If the files do not support an answer, the result is "I don't know."
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
from pypdf import PdfReader


DEFAULT_DOCS_DIR = "sample_documents"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "qwen2.5:3b"
CACHE_SCHEMA_VERSION = 5
CANDIDATE_COUNT = int(os.environ.get("RAG_CANDIDATE_COUNT", "10"))
FINAL_CONTEXT_COUNT = int(os.environ.get("RAG_FINAL_CONTEXT_COUNT", "6"))
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
CHUNK_CHARACTER_LIMIT = 1000
CHUNK_OVERLAP_LIMIT = 300


class LocalModelError(RuntimeError):
    """Ollama is not running, or a required local model is missing."""


class DocumentIndex:
    def __init__(self, folder, files, chunks, embeddings, reranker):
        self.folder = str(folder)
        self.files = list(files)
        self.chunks = chunks
        self.embeddings = embeddings
        self.reranker = reranker


def ollama_base_url():
    raw = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not raw.startswith("http://") and not raw.startswith("https://"):
        raw = "http://" + raw
    return raw


def ollama_post(endpoint, payload, timeout):
    url = f"{ollama_base_url()}/api/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.ConnectionError as exc:
        raise LocalModelError(
            "Ollama is not running at "
            f"{ollama_base_url()}. Start the Ollama app, then install the "
            f"local models with: ollama pull {EMBEDDING_MODEL} && "
            f"ollama pull {GENERATION_MODEL}"
        ) from exc
    except requests.Timeout as exc:
        raise LocalModelError(
            f"Ollama did not finish the {endpoint} request within {timeout} seconds."
        ) from exc

    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:500]
        hint = ""
        if response.status_code == 404 or "not found" in detail.lower():
            model_name = payload.get("model", GENERATION_MODEL)
            hint = f" If the model is missing, run: ollama pull {model_name}"
        raise LocalModelError(
            f"Ollama returned an error ({response.status_code}). {detail}.{hint}"
        )
    return response.json()


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def make_chunk_record(sentences, chunk_index, filename):
    # Splitting on ". " removes the period. Putting it back keeps the cited
    # passage readable and aligned with the PDF, including "Inc."
    chunk_text = ". ".join(sentence for sentence, _page_number in sentences)
    page_sentences = {}
    for sentence, page_number in sentences:
        page_sentences.setdefault(page_number, []).append(sentence)
    page_segments = [
        {
            "page_number": page_number,
            "text": ". ".join(page_sentences[page_number]),
        }
        for page_number in sorted(page_sentences)
    ]
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:10]
    slug = "".join(
        character if character.isalnum() else "-"
        for character in Path(filename).stem.lower()
    ).strip("-")
    return {
        "chunk_id": f"{slug}-chunk-{chunk_index:03d}-{content_hash}",
        "filename": filename,
        "page_numbers": [segment["page_number"] for segment in page_segments],
        "page_segments": page_segments,
        "text": chunk_text,
    }


def build_page_aware_chunks(pdf_reader, filename):
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
        if current_sentences and len(current_text) + 1 + len(sentence) > CHUNK_CHARACTER_LIMIT:
            page_chunks.append(
                make_chunk_record(current_sentences, len(page_chunks), filename)
            )

            # Carry whole sentences into the next chunk as overlap.
            overlap = []
            overlap_length = 0
            for previous_sentence in reversed(current_sentences):
                previous_length = len(previous_sentence[0])
                if overlap_length + previous_length > CHUNK_OVERLAP_LIMIT:
                    break
                overlap.insert(0, previous_sentence)
                overlap_length += previous_length
            current_sentences = overlap

        current_sentences.append((sentence, page_number))

    if current_sentences:
        page_chunks.append(
            make_chunk_record(current_sentences, len(page_chunks), filename)
        )

    return page_chunks


def list_pdfs(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Document folder not found: {folder}. "
            "Create it and add PDF files, or pass --docs sample_documents."
        )
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )


def load_chunks(folder):
    pdfs = list_pdfs(folder)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {folder}. Add the fictional engagement "
            "letters and policies, or point --docs at another folder of PDFs."
        )
    chunks = []
    for pdf_path in pdfs:
        reader = PdfReader(str(pdf_path))
        chunks.extend(build_page_aware_chunks(reader, pdf_path.name))
    return pdfs, chunks


def corpus_fingerprint(pdfs):
    digest = hashlib.sha256()
    for pdf_path in pdfs:
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()


def _cache_is_compatible(cache, fingerprint):
    required_keys = {"chunk_id", "filename", "page_numbers", "page_segments", "text"}
    return (
        cache.get("cache_schema_version") == CACHE_SCHEMA_VERSION
        and cache.get("corpus_fingerprint") == fingerprint
        and cache.get("embedding_model") == EMBEDDING_MODEL
        and len(cache.get("chunks", [])) == len(cache.get("embeddings", []))
        and all(
            isinstance(chunk, dict) and required_keys <= chunk.keys()
            for chunk in cache.get("chunks", [])
        )
    )


def embed_texts(texts):
    vectors = []
    for text in texts:
        data = ollama_post(
            "embed",
            {"model": EMBEDDING_MODEL, "input": text},
            timeout=180,
        )
        embeddings = data.get("embeddings") or []
        if not embeddings:
            raise LocalModelError(
                f"Ollama did not return an embedding from {EMBEDDING_MODEL}."
            )
        vectors.append(embeddings[0])
    return vectors


def load_or_create_embeddings(chunks, fingerprint, cache_file):
    cache_path = Path(cache_file)
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}

    if _cache_is_compatible(cache, fingerprint):
        print("Loading saved embeddings for this folder.")
        return cache["embeddings"]

    print(
        f"Creating embeddings for {len(chunks)} passages with {EMBEDDING_MODEL}."
    )
    embeddings = embed_texts(chunk["text"] for chunk in chunks)
    cache_path.write_text(
        json.dumps(
            {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "corpus_fingerprint": fingerprint,
                "embedding_model": EMBEDDING_MODEL,
                "chunks": chunks,
                "embeddings": embeddings,
            }
        )
    )
    print(f"Saved embeddings to {cache_path}.")
    return embeddings


def load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "The reranker package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        # Same public MiniLM reranker as before, on CPU. The ONNX backend needs
        # a helper that downgrades Transformers and breaks sentence-transformers.
        return CrossEncoder(
            RERANKER_MODEL,
            device="cpu",
            local_files_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Reranker model '{RERANKER_MODEL}' could not be loaded. "
            "The first run downloads this small public model. "
            f"Details: {exc}"
        ) from exc


def build_index(folder, cache_file="embeddings.json", use_reranker=True):
    folder = Path(folder)
    pdfs, chunks = load_chunks(folder)
    print(f"Reading {len(pdfs)} PDF file(s) from {folder}.")
    fingerprint = corpus_fingerprint(pdfs)
    embeddings = load_or_create_embeddings(chunks, fingerprint, cache_file)
    reranker = load_cross_encoder() if use_reranker else None
    return DocumentIndex(
        folder=folder,
        files=[pdf_path.name for pdf_path in pdfs],
        chunks=chunks,
        embeddings=embeddings,
        reranker=reranker,
    )


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def rank_by_cosine(question_embedding, chunks, embeddings):
    results = []
    for index, chunk in enumerate(chunks):
        score = cosine_similarity(question_embedding, embeddings[index])
        results.append((score, index, chunk))
    results.sort(key=lambda result: result[0], reverse=True)
    return results[:CANDIDATE_COUNT]


def rerank_candidates(reranker, question, candidates):
    if not candidates:
        return []
    pairs = [(question, chunk["text"]) for _cosine, _index, chunk in candidates]
    scores = reranker.predict(pairs, batch_size=8)
    reranked = [
        (float(score), cosine_score, chunk_index, chunk)
        for score, (cosine_score, chunk_index, chunk) in zip(scores, candidates)
    ]
    reranked.sort(key=lambda result: result[0], reverse=True)
    return reranked


def select_context(ranked_chunks, reranked=False):
    selected_chunks = [
        result[3] if reranked else result[2]
        for result in ranked_chunks[:FINAL_CONTEXT_COUNT]
    ]

    evidence_by_id = {}
    for chunk in selected_chunks:
        for segment in chunk["page_segments"]:
            if segment["text"].strip():
                evidence_id = f"E{len(evidence_by_id) + 1}"
                evidence_by_id[evidence_id] = {
                    "filename": chunk["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "page_number": segment["page_number"],
                    "quote": segment["text"],
                }
    return selected_chunks, evidence_by_id


def chunk_contains_quote(chunk, quote, page_number=None):
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        return False
    for segment in chunk.get("page_segments", []):
        if page_number is not None and segment["page_number"] != page_number:
            continue
        if normalized_quote in normalize_text(segment["text"]):
            return True
    return False


def answer_spans_missing_from_citations(answer, citations):
    """Numbers and time periods in the answer must be visible in the citation."""
    quoted = normalize_text(" ".join(citation.get("quote", "") for citation in citations))
    patterns = [
        r"\$\d[\d,]*(?:\.\d+)?",
        r"\b\d+(?:\.\d+)?\s+percent\b",
        r"\b\d+\s+(?:year|years|day|days|month|months)\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:business\s+)?(?:year|years|day|days|month|months)\b",
    ]
    missing = []
    for pattern in patterns:
        for span in re.findall(pattern, answer, flags=re.IGNORECASE):
            if normalize_text(span) not in quoted and span not in missing:
                missing.append(span)
    return missing


def validate_citations(answer, citations, selected_chunks):
    if normalize_text(answer).startswith("i don't know"):
        return not citations, []
    if not isinstance(citations, list) or not citations:
        return False, ["Answer has no citation."]

    selected_by_id = {chunk["chunk_id"]: chunk for chunk in selected_chunks}
    problems = []
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        page_number = citation.get("page_number")
        quote = citation.get("quote", "")
        filename = citation.get("filename")
        chunk = selected_by_id.get(chunk_id)

        if chunk is None:
            problems.append(f"Unknown or unselected chunk ID: {chunk_id}")
            continue
        if filename != chunk.get("filename"):
            problems.append(
                f"File {filename} does not match {chunk_id}."
            )
            continue
        if page_number not in chunk["page_numbers"]:
            problems.append(f"Page {page_number} is not part of {chunk_id}.")
            continue
        if not chunk_contains_quote(chunk, quote, page_number):
            problems.append(
                f"Quote was not found on page {page_number} in {chunk_id}."
            )

    if not problems:
        for span in answer_spans_missing_from_citations(answer, citations):
            problems.append(f'The cited passage does not contain "{span}".')

    return not problems, problems


def _evidence_prompt(question, evidence_by_id):
    context = "\n\n".join(
        (
            f"[{evidence_id}] file={source['filename']} "
            f"page={source['page_number']}\n{source['quote']}"
        )
        for evidence_id, source in evidence_by_id.items()
    )
    return f"""Answer the question in one sentence using only the numbered excerpts.
Choose the evidence before writing your answer. Return only the evidence IDs
needed to support it. If the evidence is partial, qualify the answer.
If the excerpts provide no useful evidence, answer "I don't know." with no IDs.
Every number, date, and time period in the answer must appear in the excerpts
for the evidence IDs you return.

Question:
{question}

Evidence:
{context}
"""


def _response_format(evidence_by_id):
    answered_format = {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(evidence_by_id)},
            },
            "answer": {"type": "string", "minLength": 1},
        },
        "required": ["evidence_ids", "answer"],
    }
    abstention_format = {
        "type": "object",
        "properties": {
            "evidence_ids": {"type": "array", "maxItems": 0},
            "answer": {"const": "I don't know."},
        },
        "required": ["evidence_ids", "answer"],
    }
    if evidence_by_id:
        return {"anyOf": [answered_format, abstention_format]}
    return abstention_format


def generate_answer(question, evidence_by_id, selected_chunks):
    prompt = _evidence_prompt(question, evidence_by_id)
    if os.environ.get("RAG_DEBUG"):
        print(prompt)

    started_at = time.perf_counter()
    data = ollama_post(
        "chat",
        {
            "model": GENERATION_MODEL,
            "format": _response_format(evidence_by_id),
            "messages": [
                {
                    "role": "system",
                    "content": "Use only the supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # One sentence of JSON. A cap stops a confused reply from running
            # until the context window is full.
            "options": {"temperature": 0, "num_predict": 256},
        },
        timeout=600,
    )
    generation_seconds = time.perf_counter() - started_at

    try:
        parsed = json.loads(data["message"]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    answer = parsed.get("answer", "")
    evidence_ids = parsed.get("evidence_ids")
    citations = []

    if not isinstance(answer, str) or not answer.strip():
        citations_valid = False
        citation_problems = ["Qwen did not return an answer."]
    elif normalize_text(answer).startswith("i don't know"):
        # Abstention stands even if the model also attached evidence IDs.
        # Those IDs are not shown, because the answer is that the files do not say.
        answer = "I don't know."
        citations = []
        citations_valid = True
        citation_problems = []
    elif not isinstance(evidence_ids, list) or any(
        not isinstance(evidence_id, str) or evidence_id not in evidence_by_id
        for evidence_id in evidence_ids
    ):
        citations_valid = False
        citation_problems = ["Qwen returned invalid evidence IDs."]
    else:
        answer = answer.strip()
        citations = [
            evidence_by_id[evidence_id]
            for evidence_id in dict.fromkeys(evidence_ids)
        ]
        citations_valid, citation_problems = validate_citations(
            answer, citations, selected_chunks
        )

    delivered_answer = (
        answer if citations_valid else "I don't know. (Citation validation failed.)"
    )
    return {
        "question": question,
        "model": GENERATION_MODEL,
        "model_answer": answer if isinstance(answer, str) else "",
        "delivered_answer": delivered_answer,
        "citations": citations if citations_valid else [],
        "citations_valid": citations_valid,
        "citation_problems": citation_problems,
        "generation_seconds": generation_seconds,
    }


def answer_question(question, index):
    question = " ".join(str(question).split())
    if not question:
        raise ValueError("Type a question first.")
    if index.reranker is None:
        raise RuntimeError("The reranker is not loaded, so no answer was requested.")

    question_embedding = embed_texts([question])[0]
    candidates = rank_by_cosine(question_embedding, index.chunks, index.embeddings)
    reranked = rerank_candidates(index.reranker, question, candidates)
    selected_chunks, evidence_by_id = select_context(reranked, reranked=True)
    # One Qwen call. The reranked passages are the only context it sees.
    return generate_answer(question, evidence_by_id, selected_chunks)


def print_answer(result):
    print(f"\nAnswer: {result['delivered_answer']}")
    if result["citations_valid"] and result["citations"]:
        print("\nSources:")
        for citation in result["citations"]:
            print(f"- {citation['filename']}, page {citation['page_number']}")
            print(f"  {citation['quote']}")
    elif result["citations_valid"]:
        print(
            "\nSources: none. The files do not support an answer, so nothing was cited."
        )
    else:
        print("\nThe draft answer was withheld because a citation could not be verified.")
        for problem in result["citation_problems"]:
            print(f"- {problem}")
    print(f"\nLocal model: {result['model']}")
    print(f"Time: {result['generation_seconds']:.1f}s for the answer")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask one question about every PDF in a folder."
    )
    parser.add_argument(
        "--docs",
        default=DEFAULT_DOCS_DIR,
        help="Folder of PDF files. Defaults to the fictional sample set.",
    )
    parser.add_argument(
        "--question",
        help="Ask this question and exit. Omit it to type a question.",
    )
    parser.add_argument(
        "--cache",
        default="embeddings.json",
        help="Where to save embeddings for this folder.",
    )
    args = parser.parse_args(argv)

    try:
        index = build_index(args.docs, cache_file=args.cache)
    except (LocalModelError, FileNotFoundError, RuntimeError, requests.RequestException) as exc:
        print(exc)
        return 1

    print(f"Indexed {len(index.files)} PDF file(s):")
    for name in index.files:
        print(f"- {name}")

    question = args.question
    if not question:
        try:
            question = input("\nAsk a question about these files: ")
        except EOFError:
            print(
                "No question was entered. Example: "
                'python3 curator.py --question "What is the fixed fee in the Harborline Markets engagement letter?"'
            )
            return 1

    try:
        result = answer_question(question, index)
    except (LocalModelError, ValueError, RuntimeError, requests.RequestException) as exc:
        print(exc)
        return 1

    print_answer(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
