"""Ask questions about local PDFs with Ollama and a persistent document cache."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from io import BytesIO
from pathlib import Path
from time import perf_counter

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from common import cosine_similarity, embedding_vector


ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / "knowledge_test"
CACHE_DIR = ROOT / ".curator_index"
CACHE_SCHEMA = 1
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = os.environ.get("CURATOR_LLM", "qwen2.5:3b")
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
CANDIDATE_COUNT = int(os.environ.get("RAG_CANDIDATE_COUNT", "10"))
FINAL_CONTEXT_COUNT = int(os.environ.get("RAG_FINAL_CONTEXT_COUNT", "6"))
# On knowledge_test/, both no-answer questions scored below -1.7 and all eight
# answerable questions scored above 0.3. -0.7 sits in that gap. A top rerank
# score under it returns "I don't know." and does not call Qwen.
RERANK_ABSTAIN_BELOW = -0.7
MAX_CITATIONS = 2


def make_chunk_record(sentences, chunk_index):
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
        "page_numbers": [segment["page_number"] for segment in page_segments],
        "page_segments": page_segments,
        "text": chunk_text,
    }


def build_page_aware_chunks(pdf_reader):
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
            page_chunks.append(make_chunk_record(current_sentences, len(page_chunks)))

            # Carry whole sentences into the next chunk as overlap.
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
        page_chunks.append(make_chunk_record(current_sentences, len(page_chunks)))

    return page_chunks


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def chunk_contains_quote(chunk, quote, page_number=None):
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        return False
    for segment in chunk.get("page_segments", []):
        if page_number is not None and segment["page_number"] != page_number:
            continue
        if normalized_quote in normalize_text(segment["text"]):
            return True
        source_path = chunk.get("source_path")
        page_text = pdf_page_text(source_path, segment["page_number"]) if source_path else ""
        cleaned = normalize_text(clean_passage_text(segment["text"], page_text))
        if cleaned and normalized_quote in cleaned:
            return True
    return False


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
        chunk = selected_by_id.get(chunk_id)

        if chunk is None:
            problems.append(f"Unknown or unselected chunk ID: {chunk_id}")
            continue
        if page_number not in chunk["page_numbers"]:
            problems.append(f"Page {page_number} is not part of {chunk_id}.")
            continue
        if not chunk_contains_quote(chunk, quote, page_number):
            problems.append(
                f"Quote was not found on page {page_number} in {chunk_id}."
            )

    return not problems, problems


def show_ranking(title, ranked_chunks, reranked=False):
    print(f"\n--- {title} ---")
    for rank, result in enumerate(ranked_chunks, start=1):
        if reranked:
            reranker_score, cosine_score, _index, chunk = result
            score_text = (
                f"Reranker: {reranker_score:.4f} | Cosine: {cosine_score:.4f}"
            )
        else:
            cosine_score, _index, chunk = result
            score_text = f"Cosine: {cosine_score:.4f}"
        pages = ", ".join(map(str, chunk["page_numbers"]))
        print(f"{rank}. {chunk['chunk_id']} | Pages: {pages} | {score_text}")


_PDF_READERS = {}
_PAGE_TEXT = {}


def pdf_page_text(pdf_path, page_number):
    """Original page text, used only to restore punctuation for the answer."""
    path = str(pdf_path)
    cached = _PAGE_TEXT.setdefault(path, {})
    if page_number in cached:
        return cached[page_number]
    try:
        reader = _PDF_READERS.get(path)
        if reader is None:
            reader = PdfReader(path)
            _PDF_READERS[path] = reader
    except (OSError, PdfReadError) as error:
        raise ValueError(f"Could not read PDF text from {path}.") from error
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError(
            f"Page {page_number} is outside {path} ({len(reader.pages)} pages)."
        )
    text = reader.pages[page_number - 1].extract_text() or ""
    cached[page_number] = text
    return text


def sentence_pieces(page_text):
    """Same '. ' split as chunking, remembering which periods the split removed.

    A period that belongs to a heading line such as '6. HR records' is marked
    so the answer text can drop that number without dropping sentence periods.
    """
    original = page_text or ""
    flat = original.replace("\n", " ")
    parts = flat.split(". ")
    last = len(parts) - 1
    pieces = []
    cursor = 0
    for index, part in enumerate(parts):
        lost_period = index < last
        end = cursor + len(part)
        heading_period = False
        if lost_period and original[end:end + 2] == ". ":
            number_start = end
            while number_start > 0 and original[number_start - 1].isdigit():
                number_start -= 1
            number = original[number_start:end]
            line_start = original.rfind("\n", 0, number_start) + 1
            if number and len(number) <= 2 and original[line_start:number_start].strip() == "":
                heading_period = True
        text = part.strip()
        if text:
            pieces.append((text, lost_period, heading_period))
        cursor = end + (2 if lost_period else 0)
    return pieces


def find_sentence_span(pieces, collapsed):
    norms = [" ".join(text.split()) for text, _lost_period, _heading in pieces]
    for start in range(len(norms)):
        acc = []
        for end in range(start, len(norms)):
            acc.append(norms[end])
            joined = " ".join(acc)
            if joined == collapsed:
                return start, end + 1
            if len(joined) > len(collapsed):
                break
    return None


def clean_passage_text(segment_text, page_text):
    """Punctuation for the Qwen prompt. Retrieval text stays unchanged.

    Periods, colons, and semicolons stay. A heading number from a line such as
    '6. HR records' is dropped, including when it was glued to the line above.
    """
    collapsed = " ".join(str(segment_text).split())
    if not page_text:
        return collapsed
    pieces = sentence_pieces(page_text)
    span = find_sentence_span(pieces, collapsed)
    if span is None:
        return collapsed
    start, end = span
    kept = []
    for text, lost_period, heading_period in pieces[start:end]:
        body = " ".join(text.split())
        if heading_period:
            tokens = body.split()
            if tokens and tokens[-1].isdigit() and len(tokens[-1]) <= 2:
                body = " ".join(tokens[:-1])
                lost_period = False
        if not body:
            continue
        if lost_period and not body.endswith("."):
            body += "."
        kept.append(body)
    return " ".join(kept)


def select_context(ranked_chunks, reranked=False):
    selected_chunks = [
        result[3] if reranked else result[2]
        for result in ranked_chunks[:FINAL_CONTEXT_COUNT]
    ]
    evidence_by_id = {}
    seen = set()
    for chunk in selected_chunks:
        source_path = chunk.get("source_path")
        for segment in chunk["page_segments"]:
            page_text = ""
            if source_path:
                page_text = pdf_page_text(source_path, segment["page_number"])
            quote = clean_passage_text(segment["text"], page_text)
            page_number = segment["page_number"]
            key = (page_number, quote.casefold())
            if key in seen or not any(character.isalpha() for character in quote):
                continue
            seen.add(key)
            evidence_id = f"E{len(evidence_by_id) + 1}"
            evidence_by_id[evidence_id] = {
                "chunk_id": chunk["chunk_id"],
                "page_number": page_number,
                "quote": quote,
                "source_file": chunk.get("source_file", ""),
            }

    return selected_chunks, evidence_by_id


def ollama_request(endpoint, payload):
    try:
        response = requests.post(f"{OLLAMA_URL}/api/{endpoint}", json=payload, timeout=300)
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise RuntimeError("Start Ollama and run the command again.") from error
    return response.json()


def embed(text):
    result = ollama_request("embed", {"model": EMBED_MODEL, "input": text})
    return embedding_vector(result, EMBED_MODEL)


def chat_answer(system_prompt, prompt, response_format=None, max_tokens=300):
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    if response_format is not None:
        payload["format"] = response_format
    result = ollama_request("chat", payload)
    return result["message"]["content"]


def file_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def read_index(path):
    try:
        with path.open(encoding="utf-8") as handle:
            index = json.load(handle)
    except (FileNotFoundError, ValueError):
        return {}
    if (
        isinstance(index, dict)
        and index.get("schema") == CACHE_SCHEMA
        and index.get("model") == EMBED_MODEL
        and isinstance(index.get("chunks"), list)
        and isinstance(index.get("embeddings"), list)
        and len(index["chunks"]) == len(index["embeddings"]) > 0
    ):
        return index
    return {}


def save_index(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(index, handle)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_or_build_index(pdf_path):
    """Reuse the saved passages and embeddings for this PDF whenever possible."""
    file_key = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{file_key}.json"
    index = read_index(cache_path)
    signature = file_signature(pdf_path)

    def bind_source(index):
        resolved = str(pdf_path.resolve())
        for chunk in index["chunks"]:
            chunk["source_path"] = resolved
        return index

    if index.get("stat") == signature:
        print(f"Index: reused {len(index['chunks'])} passages from {pdf_path.name}.")
        return bind_source(index)

    contents = pdf_path.read_bytes()
    if file_signature(pdf_path) != signature:
        raise RuntimeError("The PDF changed while it was being read. Run the command again.")
    digest = hashlib.sha256(contents).hexdigest()

    if index.get("sha256") == digest:
        index["stat"] = signature
        save_index(cache_path, index)
        print(f"Index: reused {len(index['chunks'])} passages from {pdf_path.name}.")
        return bind_source(index)

    chunks = build_page_aware_chunks(PdfReader(BytesIO(contents)))
    if not chunks:
        raise ValueError("No text could be extracted. Choose a PDF with selectable text.")
    print(f"Index: building {len(chunks)} passages from {pdf_path.name}...")
    index = {
        "schema": CACHE_SCHEMA,
        "model": EMBED_MODEL,
        "stat": signature,
        "sha256": digest,
        "chunks": chunks,
        "embeddings": [embed(chunk["text"]) for chunk in chunks],
    }
    save_index(cache_path, index)
    print(f"Index: built {len(chunks)} passages and saved them in {CACHE_DIR.name}/.")
    return bind_source(index)


def list_pdfs(folder):
    folder = Path(folder)
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_corpus(pdf_paths):
    """One index over every PDF. Retrieval and chunking stay the same per file."""
    chunks = []
    embeddings = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        index = load_or_build_index(pdf_path)
        for chunk in index["chunks"]:
            tagged = dict(chunk)
            tagged["source_file"] = pdf_path.name
            chunks.append(tagged)
        embeddings.extend(index["embeddings"])
    return {"chunks": chunks, "embeddings": embeddings}


def load_reranker():
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(RERANKER_MODEL, backend="onnx", device="cpu")
    except Exception:
        return CrossEncoder(RERANKER_MODEL, backend="torch", device="cpu")


def rank_candidates(question, index):
    question_vector = embed(question)
    results = [
        (cosine_similarity(question_vector, vector), chunk_index, chunk)
        for chunk_index, (chunk, vector) in enumerate(zip(index["chunks"], index["embeddings"]))
    ]
    return sorted(results, key=lambda item: item[0], reverse=True)[:CANDIDATE_COUNT]


def rerank_candidates(question, candidates, reranker):
    pairs = [(question, chunk["text"]) for _score, _index, chunk in candidates]
    scores = [float(score) for score in reranker.predict(pairs, batch_size=8)]
    if len(scores) != len(candidates) or any(not math.isfinite(score) for score in scores):
        raise ValueError("The reranker returned invalid scores.")
    results = [
        (score, cosine_score, chunk_index, chunk)
        for score, (cosine_score, chunk_index, chunk) in zip(scores, candidates)
    ]
    return sorted(results, key=lambda item: item[0], reverse=True)


def generate_answer(question, ranked, pdf_name, reranked=False, quiet=False):
    selected_chunks, evidence = select_context(ranked, reranked=reranked)
    context = "\n\n".join(f"[{key}] {item['quote']}" for key, item in evidence.items())
    system_prompt = (
        "Copy, word for word, every sentence that states a fact the question asks for. "
        "Use the general or standard rule, not an exception, a target, or the first date you see. "
        "If one sentence states the standard rule and also an exception, copy that whole sentence. "
        "Cite the excerpt you copied. Use a second excerpt only if the answer needs both. "
        "If no passage states the fact explicitly, return 'I don't know.' and an empty evidence_ids list."
        " Answer only if an excerpt states the answer for the exact item asked (same term; e.g. QST is not HST). Otherwise reply exactly 'I don't know.' with empty evidence_ids."
        " For tables, use only the row matching the question's item."
    )
    user_prompt = f"Question:\n{question}\n\nEvidence:\n{context}" if context else ""

    def finish(answer, citations):
        if not quiet:
            print(f"Answer: {answer}")
            seen = []
            for citation in citations:
                source = citation.get("source_file") or pdf_name
                key = (source, citation["page_number"])
                if key in seen:
                    continue
                seen.append(key)
                print(f"Source: {source}, page {citation['page_number']}")
        return {
            "answer": answer,
            "citations": citations,
            "system_prompt": system_prompt if context else "",
            "prompt": user_prompt,
            "passages": selected_chunks,
            "evidence": evidence,
        }

    if not context:
        return finish("I don't know.", [])

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "evidence_ids": {
                "type": "array",
                "maxItems": MAX_CITATIONS,
                "items": {"type": "string", "enum": list(evidence)},
            },
        },
        "required": ["answer", "evidence_ids"],
    }
    output = chat_answer(system_prompt, user_prompt, schema, max_tokens=128)
    try:
        result = json.loads(output)
    except json.JSONDecodeError:
        result = None

    if not isinstance(result, dict):
        return finish("I don't know. (The model returned an invalid response.)", [])

    answer = result.get("answer")
    ids = result.get("evidence_ids")
    if (
        not isinstance(answer, str)
        or not answer.strip()
        or not isinstance(ids, list)
        or len(ids) > MAX_CITATIONS
        or any(not isinstance(key, str) or key not in evidence for key in ids)
    ):
        return finish("I don't know. (The model returned an invalid answer or citation.)", [])

    citations = [evidence[key] for key in dict.fromkeys(ids)]
    valid, _problems = validate_citations(answer, citations, selected_chunks)
    if not valid:
        return finish("I don't know. (Citation validation failed.)", [])

    return finish(answer.strip(), citations)


def ask_question(question, index, pdf_name, reranker, compare=False, quiet=False):
    started = perf_counter()
    candidates = rank_candidates(question, index)
    search_seconds = perf_counter() - started
    reranked = None
    rerank_seconds = None
    if reranker is not None:
        started = perf_counter()
        try:
            reranked = rerank_candidates(question, candidates, reranker)
        except (RuntimeError, ValueError) as error:
            print(f"Reranker unavailable for this question. Using cosine search. {error}")
        rerank_seconds = perf_counter() - started

    if compare and not quiet:
        show_ranking("Cosine candidates", candidates)
        if reranked is not None:
            show_ranking("Reranked candidates", reranked, reranked=True)
    started = perf_counter()
    top_score = reranked[0][0] if reranked else None
    if top_score is not None and top_score < RERANK_ABSTAIN_BELOW:
        if not quiet:
            print(f"Top rerank score {top_score:.4f} is below {RERANK_ABSTAIN_BELOW}.")
            print("Answer: I don't know.")
        result = {
            "answer": "I don't know.",
            "citations": [],
            "system_prompt": "",
            "prompt": "",
            "passages": [],
            "evidence": {},
        }
    elif compare:
        if not quiet:
            print("\nCosine-only answer")
        result = generate_answer(question, candidates, pdf_name, quiet=quiet)
        if reranked is not None:
            if not quiet:
                print("\nReranked answer")
            result = generate_answer(
                question, reranked, pdf_name, reranked=True, quiet=quiet
            )
    elif reranked is None:
        result = generate_answer(question, candidates, pdf_name, quiet=quiet)
    else:
        result = generate_answer(
            question, reranked, pdf_name, reranked=True, quiet=quiet
        )
    answer_seconds = perf_counter() - started
    timings = [f"search {search_seconds:.1f}s"]
    if rerank_seconds is not None:
        timings.append(f"rerank {rerank_seconds:.1f}s")
    timings.append(f"answer {answer_seconds:.1f}s")
    if not quiet:
        print("Time: " + ", ".join(timings))
    result["candidates"] = candidates
    result["reranked"] = reranked
    result["search_seconds"] = search_seconds
    result["rerank_seconds"] = rerank_seconds
    result["answer_seconds"] = answer_seconds
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        type=Path,
        help="Index every PDF in this folder. Defaults to knowledge_test/.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Index one PDF instead of the folder.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--index", action="store_true", help="Prepare the cache and exit.")
    mode.add_argument("--question", help="Answer one document question and exit.")
    parser.add_argument("--compare", action="store_true", help="Show both search rankings and answers.")
    args = parser.parse_args(argv)
    if CANDIDATE_COUNT <= 0 or FINAL_CONTEXT_COUNT <= 0:
        parser.error("Candidate and context counts must be positive.")
    if args.folder is not None and args.pdf is not None:
        parser.error("Pass --folder or --pdf, not both.")

    started = perf_counter()
    try:
        if args.pdf is not None:
            pdf_path = args.pdf.expanduser().resolve()
            if not pdf_path.is_file():
                parser.error(f"PDF not found at {pdf_path}.")
            index = load_or_build_index(pdf_path)
            source_name = pdf_path.name
            idle_prompt = "\nAsk a question about the PDF. Type exit to quit."
        else:
            folder = (args.folder if args.folder is not None else DEFAULT_FOLDER).expanduser().resolve()
            if not folder.is_dir():
                parser.error(f"PDF folder not found at {folder}.")
            pdfs = list_pdfs(folder)
            if not pdfs:
                parser.error(f"No PDFs found in {folder}.")
            index = load_corpus(pdfs)
            source_name = folder.name
            idle_prompt = "\nAsk a question about the PDFs. Type exit to quit."
    except (OSError, PdfReadError, ValueError, RuntimeError, requests.RequestException) as error:
        parser.exit(1, f"Could not prepare the documents: {error}\n")
    print(f"Index time: {perf_counter() - started:.1f}s")

    if args.index:
        return 0

    reranker = None
    reranker_attempted = False
    if args.question is None:
        print(idle_prompt)

    while True:
        try:
            entry = args.question if args.question is not None else input("\nYou> ")
            entry = entry.strip()
            if entry.casefold() in {"exit", "quit"}:
                return 0
            if not entry:
                if args.question is not None:
                    parser.error("Enter a question after --question.")
                continue

            if not reranker_attempted:
                reranker_attempted = True
                print("Loading the reranker for the first question...", flush=True)
                started = perf_counter()
                try:
                    reranker = load_reranker()
                except Exception as error:
                    print(f"Reranker could not load. Using cosine search. {error}")
                print(f"Reranker load time: {perf_counter() - started:.1f}s")
            ask_question(entry, index, source_name, reranker, args.compare)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
            print(f"Could not answer: {error}")
            if args.question is not None:
                return 1

        if args.question is not None:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
