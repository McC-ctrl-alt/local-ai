"""Score curator_cached.py on eval/questions.yaml and write eval/results.csv."""

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import curator_cached as curator


NUMBER_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "5": "five",
    "7": "seven",
    "13": "thirteen",
    "15": "fifteen",
    "20": "twenty",
    "30": "thirty",
    "75": "seventy five",
}
KEYWORD_STOP = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "the", "and", "for",
    "with", "from", "that", "this", "after", "before", "must", "into",
    "about", "have", "has", "been", "their", "them", "than", "then",
}
ABSTAIN_HINTS = (
    "not stated",
    "not in the",
    "do not state",
    "does not state",
    "does not say",
    "do not say",
    "cannot tell",
    "can't tell",
    "no information",
    "not specified",
    "not provided",
)
CSV_FIELDS = [
    "id",
    "kind",
    "file",
    "page",
    "question",
    "expected",
    "fact_in_text",
    "cosine_page_rank",
    "keyword_page_rank",
    "rerank_rank",
    "in_rerank_top6",
    "rerank_moved",
    "answer",
    "cited_pages",
    "search_s",
    "rerank_s",
    "answer_s",
    "label",
    "citation_ok",
    "needs_human",
]


def loosen(text):
    folded = curator.normalize_text(text)
    folded = folded.replace(" per cent", " %").replace(" percent", " %")
    folded = folded.replace("%", " % ")
    folded = folded.replace("-", " ").replace("+", " ")
    cleaned = []
    for character in folded:
        if character.isalnum() or character.isspace() or character == "%":
            cleaned.append(character)
        else:
            cleaned.append(" ")
    return " ".join(token for token in "".join(cleaned).split() if token not in {"the", "a", "an"})


def fact_forms(fact):
    base = loosen(fact)
    forms = [base]
    for digit, word in NUMBER_WORDS.items():
        token = f" {digit} "
        padded = f" {base} "
        if token in padded:
            forms.append(padded.replace(token, f" {word} ", 1).strip())
    return forms


def phrase_in(text, fact):
    folded = f" {loosen(text)} "
    return any(f" {form} " in folded or loosen(text) == form for form in fact_forms(fact))


def page_text(index, source_file, page_number):
    parts = []
    for chunk in index["chunks"]:
        if chunk.get("source_file") != source_file:
            continue
        for segment in chunk.get("page_segments", []):
            if segment["page_number"] == page_number:
                parts.append(segment["text"])
    return " ".join(parts)


def corpus_text(index):
    return " ".join(chunk["text"] for chunk in index["chunks"])


def fact_present(index, item):
    """Whether the gold fact is in the extracted passage text for its page."""
    if item["kind"] == "no-answer":
        blob = curator.normalize_text(corpus_text(index))
        probes = [curator.normalize_text(probe) for probe in item.get("absent_probes", [])]
        return any(probe and probe in blob for probe in probes)
    source = item.get("file") or ""
    page = item.get("page")
    if not source or page is None:
        return False
    blob = page_text(index, source, int(page))
    needles = item.get("page_facts") or item.get("facts") or []
    return all(curator.normalize_text(fact) in curator.normalize_text(blob) for fact in needles)


def keyword_candidates(question, index):
    tokens = [
        token
        for token in curator.normalize_text(question).split()
        if token not in KEYWORD_STOP and len(token) > 2
    ]
    scored = []
    for chunk_index, chunk in enumerate(index["chunks"]):
        text = curator.normalize_text(chunk["text"])
        score = sum(text.count(token) for token in tokens)
        scored.append((score, chunk_index, chunk))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return scored[: curator.CANDIDATE_COUNT]


def page_rank(ranked, source_file, page_number, reranked=False, require_positive=False):
    if not source_file or page_number is None:
        return None
    page_number = int(page_number)
    for rank, result in enumerate(ranked, start=1):
        score = result[0]
        chunk = result[3] if reranked else result[2]
        if require_positive and score <= 0:
            continue
        if chunk.get("source_file") == source_file and page_number in chunk.get("page_numbers", []):
            return rank
    return None


def is_idk(answer):
    text = curator.normalize_text(answer)
    return text.startswith("i don't know") or text.startswith("i do not know")


def looks_unresolved(answer):
    text = curator.normalize_text(answer)
    return any(hint in text for hint in ABSTAIN_HINTS)


def score_answer(item, answer):
    """Return a label and whether a person should score the answer."""
    if item["kind"] == "no-answer":
        if is_idk(answer):
            return "correct-IDK", False
        return "wrong", looks_unresolved(answer)

    facts = item.get("facts") or []
    hits = [phrase_in(answer, fact) for fact in facts]
    if facts and all(hits):
        return "correct", False
    if any(hits):
        return "partial", True
    if is_idk(answer):
        return "wrong", False
    fact_tokens = set()
    for fact in facts:
        fact_tokens.update(token for token in loosen(fact).split() if len(token) >= 4 or any(character.isdigit() for character in token))
    answer_tokens = set(loosen(answer).split())
    overlap = fact_tokens & answer_tokens
    unclear = bool(overlap) or looks_unresolved(answer)
    return "wrong", unclear


def citation_matches(item, citations):
    if item["kind"] == "no-answer":
        return not citations
    expected = (item.get("file") or "", int(item["page"]))
    cited = {
        (citation.get("source_file") or "", citation.get("page_number"))
        for citation in citations
    }
    return expected in cited


def format_citations(citations):
    seen = []
    for citation in citations:
        label = f"{citation.get('source_file') or 'unknown'} p.{citation.get('page_number')}"
        if label not in seen:
            seen.append(label)
    return "; ".join(seen)


def format_rank(rank):
    return "" if rank is None else str(rank)


def seconds(value):
    return "" if value is None else f"{value:.3f}"


def load_questions(path):
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    questions = document.get("questions") if isinstance(document, dict) else document
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {path}.")
    return questions


def evaluate(folder, questions):
    index = curator.load_corpus(curator.list_pdfs(folder))
    print("Loading the reranker...", flush=True)
    reranker = curator.load_reranker()
    rows = []
    for item in questions:
        present = fact_present(index, item)
        has_page = bool(item.get("file")) and item.get("page") is not None
        outcome = curator.ask_question(
            item["question"], index, item.get("file") or "", reranker, compare=False, quiet=True
        )
        if has_page:
            keyword = keyword_candidates(item["question"], index)
            keyword_rank = page_rank(
                keyword, item.get("file"), item.get("page"), require_positive=True
            )
            cosine_rank = page_rank(outcome["candidates"], item.get("file"), item.get("page"))
            rerank_rank = None
            if outcome["reranked"] is not None:
                rerank_rank = page_rank(
                    outcome["reranked"], item.get("file"), item.get("page"), reranked=True
                )
            in_top6 = rerank_rank is not None and rerank_rank <= curator.FINAL_CONTEXT_COUNT
            moved = (
                cosine_rank is not None
                and rerank_rank is not None
                and cosine_rank != rerank_rank
            )
            cosine_cell = format_rank(cosine_rank)
            keyword_cell = format_rank(keyword_rank)
            rerank_cell = format_rank(rerank_rank)
            top6_cell = "yes" if in_top6 else "no"
            moved_cell = "yes" if moved else "no"
        else:
            cosine_cell = keyword_cell = rerank_cell = top6_cell = moved_cell = "n/a"
        label, needs_human = score_answer(item, outcome["answer"])
        citation_ok = citation_matches(item, outcome["citations"])
        row = {
            "id": item["id"],
            "kind": item["kind"],
            "file": item.get("file") or "",
            "page": "" if item.get("page") is None else item["page"],
            "question": item["question"],
            "expected": item.get("expected", ""),
            "fact_in_text": "yes" if present else "no",
            "cosine_page_rank": cosine_cell,
            "keyword_page_rank": keyword_cell,
            "rerank_rank": rerank_cell,
            "in_rerank_top6": top6_cell,
            "rerank_moved": moved_cell,
            "answer": outcome["answer"],
            "cited_pages": format_citations(outcome["citations"]),
            "search_s": seconds(outcome["search_seconds"]),
            "rerank_s": seconds(outcome["rerank_seconds"]),
            "answer_s": seconds(outcome["answer_seconds"]),
            "label": label,
            "citation_ok": "yes" if citation_ok else "no",
            "needs_human": "yes" if needs_human else "no",
        }
        rows.append(row)
        print_question(row)
    return rows


def format_time_line(row):
    search = f"{float(row['search_s']):.1f} seconds"
    rerank = "n/a" if row["rerank_s"] == "" else f"{float(row['rerank_s']):.1f} seconds"
    answer = f"{float(row['answer_s']):.1f} seconds"
    return f"Time: Search {search}, Rerank: {rerank}, Answer: {answer}"


def print_question(row):
    print(f"\n=== {row['id']} ({row['kind']}) ===")
    print(f"expected fact in extracted text: {row['fact_in_text']}")
    print(
        "right page in candidates: "
        f"cosine {row['cosine_page_rank'] or 'no'} | keyword {row['keyword_page_rank'] or 'no'}"
    )
    print(f"in reranked top-{curator.FINAL_CONTEXT_COUNT}: {row['in_rerank_top6']} (rank {row['rerank_rank'] or 'none'})")
    print(f"rerank moved it: {row['rerank_moved']}")
    print(f"answer: {row['answer']}")
    print(f"cited pages: {row['cited_pages'] or 'none'}")
    print(format_time_line(row))
    print(f"label: {row['label']} | citation ok: {row['citation_ok']} | needs human: {row['needs_human']}")


def average(rows, key):
    values = [float(row[key]) for row in rows if row[key] != ""]
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize(rows):
    counts = {label: 0 for label in ("correct", "partial", "wrong", "correct-IDK")}
    for row in rows:
        if row["label"] in counts:
            counts[row["label"]] += 1
    cited = sum(row["citation_ok"] == "yes" for row in rows)
    search_avg = average(rows, "search_s")
    rerank_avg = average(rows, "rerank_s")
    answer_avg = average(rows, "answer_s")
    lines = [
        f"correct: {counts['correct']}",
        f"partial: {counts['partial']}",
        f"wrong: {counts['wrong']}",
        f"correct-IDK: {counts['correct-IDK']}",
        f"citation accuracy: {cited}/{len(rows)} ({(100 * cited / len(rows)) if rows else 0:.0f}%)",
        (
            "avg seconds: "
            f"search {search_avg:.1f}, rerank {rerank_avg:.1f}, "
            f"answer {answer_avg:.1f}, total {search_avg + rerank_avg + answer_avg:.1f}"
        ),
    ]
    for row in rows:
        lines.append(f"{row['id']}: {format_time_line(row)}")
    flagged = [row for row in rows if row["needs_human"] == "yes"]
    if flagged:
        lines.append("needs human scoring:")
        for row in flagged:
            lines.append(f"- {row['id']} [{row['label']}] {row['answer']}")
    else:
        lines.append("needs human scoring: none")
    return "\n".join(lines)


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "knowledge_test")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.csv")
    args = parser.parse_args(argv)
    questions = load_questions(args.questions.expanduser().resolve())
    rows = evaluate(args.folder.expanduser().resolve(), questions)
    write_results(args.output, rows)
    summary = summarize(rows)
    summary_path = args.output.with_name("summary.txt")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print("\n=== summary ===")
    print(summary)
    print(f"\nWrote {args.output}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
