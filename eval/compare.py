"""Compare an eval results CSV to the baseline.

Prints correct, wrong, IDK, and citation-hit counts, plus average search,
rerank, and answer times. For each question id in both files, prints the
baseline and new top rerank score, the delta, and the label. A question is
marked CHANGED when that score moves by more than 0.01 or the label
changes. Exits 1 when any of those counts gets worse, or when any
no-answer question gets an answer. That check does not depend on what the
baseline said. Average times and the per-question lines are printed and do
not change the exit status.
"""

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "eval" / "results_baseline_mac.csv"
DEFAULT_RESULTS = ROOT / "eval" / "results.csv"


def is_idk(answer):
    """Same abstention check as eval/run_eval.py."""
    text = " ".join(str(answer).casefold().split())
    return text.startswith("i don't know") or text.startswith("i do not know")


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}.")
    missing = {"id", "kind", "answer", "label", "citation_ok", "search_s", "rerank_s", "answer_s"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}.")
    return rows


def count_labels(rows):
    counts = {"correct": 0, "partial": 0, "wrong": 0, "IDK": 0}
    for row in rows:
        label = row["label"]
        if label == "correct-IDK":
            counts["IDK"] += 1
        elif label in counts:
            counts[label] += 1
        else:
            raise ValueError(f"Unknown label {label!r} on {row['id']}.")
    counts["citation hits"] = sum(row["citation_ok"] == "yes" for row in rows)
    return counts


def average(rows, key):
    values = []
    for row in rows:
        raw = (row.get(key) or "").strip()
        if raw == "":
            continue
        values.append(float(raw))
    if not values:
        return None
    return sum(values) / len(values)


def signed(value, digits):
    text = f"{value:+.{digits}f}"
    if text.startswith("+") and float(value) == 0:
        return f"{0:.{digits}f}"
    return text


def format_seconds(value):
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_score(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def parse_score(row):
    raw = (row.get("top_rerank_score") or "").strip()
    if raw == "":
        return None
    return float(raw)


def worse_counts(baseline, results):
    problems = []
    if results["correct"] < baseline["correct"]:
        problems.append(f"correct {results['correct']} < baseline {baseline['correct']}")
    if results["wrong"] > baseline["wrong"]:
        problems.append(f"wrong {results['wrong']} > baseline {baseline['wrong']}")
    if results["IDK"] < baseline["IDK"]:
        problems.append(f"IDK {results['IDK']} < baseline {baseline['IDK']}")
    if results["citation hits"] < baseline["citation hits"]:
        problems.append(
            f"citation hits {results['citation hits']} < baseline {baseline['citation hits']}"
        )
    return problems


def answered_no_answer(baseline_rows, result_rows):
    """Any no-answer question this file answers, baseline or not."""
    by_id = {row["id"]: row for row in result_rows}
    problems = []
    seen = set()
    for row in list(baseline_rows) + list(result_rows):
        if row["kind"] != "no-answer" or row["id"] in seen:
            continue
        seen.add(row["id"])
        current = by_id.get(row["id"])
        if current is None:
            problems.append(f"{row['id']} is a no-answer question and is missing")
            continue
        if not is_idk(current["answer"]):
            problems.append(
                f"{row['id']} is a no-answer question and was answered: {current['answer']}"
            )
    return problems


def print_counts(baseline, results):
    print(f"{'':16} {'baseline':>10} {'new':>8} {'delta':>8}")
    for name in ("correct", "partial", "wrong", "IDK", "citation hits"):
        delta = results[name] - baseline[name]
        print(f"{name:16} {baseline[name]:10d} {results[name]:8d} {signed(delta, 0):>8}")


def print_times(baseline_rows, result_rows):
    print()
    print(f"{'avg seconds':16} {'baseline':>10} {'new':>8} {'delta':>8}")
    for key, label in (("search_s", "search"), ("rerank_s", "rerank"), ("answer_s", "answer")):
        before = average(baseline_rows, key)
        after = average(result_rows, key)
        if before is None or after is None:
            delta = "n/a"
        else:
            delta = signed(after - before, 3)
        print(f"{label:16} {format_seconds(before):>10} {format_seconds(after):>8} {delta:>8}")


def print_questions(baseline_rows, result_rows):
    """Top rerank score and label for each question id present in both files."""
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    result_by_id = {row["id"]: row for row in result_rows}
    print()
    print("top rerank score:")
    print(f"{'id':28} {'baseline':>10} {'new':>8} {'delta':>8}  label")
    seen = set()
    printed = False
    for row in baseline_rows:
        question_id = row["id"]
        if question_id in seen or question_id not in result_by_id:
            continue
        seen.add(question_id)
        printed = True
        before_row = baseline_by_id[question_id]
        after_row = result_by_id[question_id]
        before = parse_score(before_row)
        after = parse_score(after_row)
        if before is None or after is None:
            delta_text = "n/a"
            moved = False
        else:
            delta_text = signed(after - before, 4)
            moved = abs(float(delta_text)) > 0.01
        label_text = f"{before_row['label']} -> {after_row['label']}"
        mark = ""
        if moved or before_row["label"] != after_row["label"]:
            mark = " CHANGED"
        print(
            f"{question_id:28} {format_score(before):>10} {format_score(after):>8} "
            f"{delta_text:>8}  {label_text}{mark}"
        )
    if not printed:
        print("  (none)")


def print_no_answer(baseline_rows, result_rows):
    by_id = {row["id"]: row for row in result_rows}
    print()
    print("no-answer questions:")
    seen = False
    for row in baseline_rows:
        if row["kind"] != "no-answer":
            continue
        seen = True
        current = by_id.get(row["id"])
        new_answer = "(missing)" if current is None else current["answer"]
        print(f"  {row['id']}: {row['answer']} -> {new_answer}")
    if not seen:
        print("  (none)")


def compare(baseline_rows, result_rows):
    baseline_counts = count_labels(baseline_rows)
    result_counts = count_labels(result_rows)
    problems = worse_counts(baseline_counts, result_counts)
    problems.extend(answered_no_answer(baseline_rows, result_rows))
    return baseline_counts, result_counts, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    baseline_path = args.baseline.expanduser().resolve()
    results_path = args.results.expanduser().resolve()
    try:
        baseline_rows = load_rows(baseline_path)
        result_rows = load_rows(results_path)
        baseline_counts, result_counts, problems = compare(baseline_rows, result_rows)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Could not compare: {error}", file=sys.stderr)
        return 2

    print(f"baseline: {baseline_path}")
    print(f"results:  {results_path}")
    print()
    print_counts(baseline_counts, result_counts)
    print_times(baseline_rows, result_rows)
    print_questions(baseline_rows, result_rows)
    print_no_answer(baseline_rows, result_rows)
    print()
    if problems:
        print("fail:")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print("pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
