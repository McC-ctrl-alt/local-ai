"""Ask questions about local PDFs with Ollama and a persistent document cache.

Parse, chunk, embed, index, cosine top 10, cross-encoder rerank, and the
-0.7 cutoff live in engine.py. This file asks Qwen and prints the answer.
eval/run_eval.py imports this module and does not need to change.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import engine
from engine import (
    CACHE_DIR,
    CACHE_SCHEMA,
    CANDIDATE_COUNT,
    CHAT_MODEL,
    EMBED_MODEL,
    FINAL_CONTEXT_COUNT,
    OLLAMA_URL,
    RERANK_ABSTAIN_BELOW,
    RERANKER_MODEL,
    build_page_aware_chunks,
    chat_answer,
    embed,
    list_pdfs,
    load_corpus,
    load_or_build_index,
    load_reranker,
    make_chunk_record,
    rank_candidates,
    rerank_below_cutoff,
    rerank_candidates,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = ROOT / "knowledge_test"
MAX_CITATIONS = 2


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


def generate_answer(question, ranked, pdf_name, reranked=False, quiet=False):
    selected_chunks, evidence = select_context(ranked, reranked=reranked)
    context = "\n\n".join(f"[{key}] {item['quote']}" for key, item in evidence.items())
    system_prompt = (
        "Copy, word for word, every sentence that states a fact the question asks for. "
        "Use the general or standard rule, not an exception, a target, or the first date you see. "
        "If one sentence states the standard rule and also an exception, copy that whole sentence. "
        "Cite the excerpt you copied. Use a second excerpt only if the answer needs both. "
        "If no passage states the fact explicitly, return 'I don't know.' and an empty evidence_ids list."
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
    output = engine.chat_answer(system_prompt, user_prompt, schema, max_tokens=128)
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
    if rerank_below_cutoff(top_score):
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
