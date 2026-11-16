import csv
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare", ROOT / "eval" / "compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

FIELDS = [
    "id",
    "kind",
    "answer",
    "label",
    "citation_ok",
    "search_s",
    "rerank_s",
    "answer_s",
]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in FIELDS}
            full.update(row)
            writer.writerow(full)


class CompareTests(unittest.TestCase):
    def test_fails_when_a_no_answer_question_is_answered_even_if_baseline_was_too(self):
        rows = [
            {
                "id": "hr-parental-weeks",
                "kind": "no-answer",
                "answer": "75%",
                "label": "wrong",
                "citation_ok": "no",
            },
            {
                "id": "hr-vacation-entitlement",
                "kind": "fact",
                "answer": "15 working days",
                "label": "correct",
                "citation_ok": "yes",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            baseline = folder / "baseline.csv"
            results = folder / "results.csv"
            write_csv(baseline, rows)
            write_csv(results, rows)
            with redirect_stdout(io.StringIO()):
                code = compare.main([str(results), "--baseline", str(baseline)])
        self.assertEqual(code, 1)

    def test_passes_when_every_no_answer_question_abstains(self):
        baseline = [
            {
                "id": "hr-parental-weeks",
                "kind": "no-answer",
                "answer": "75%",
                "label": "wrong",
                "citation_ok": "no",
            },
            {
                "id": "hr-vacation-entitlement",
                "kind": "fact",
                "answer": "15 working days",
                "label": "correct",
                "citation_ok": "yes",
            },
        ]
        results = [
            {
                "id": "hr-parental-weeks",
                "kind": "no-answer",
                "answer": "I don't know.",
                "label": "correct-IDK",
                "citation_ok": "yes",
            },
            {
                "id": "hr-vacation-entitlement",
                "kind": "fact",
                "answer": "15 working days",
                "label": "correct",
                "citation_ok": "yes",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            baseline_path = folder / "baseline.csv"
            results_path = folder / "results.csv"
            write_csv(baseline_path, baseline)
            write_csv(results_path, results)
            with redirect_stdout(io.StringIO()):
                code = compare.main([str(results_path), "--baseline", str(baseline_path)])
        self.assertEqual(code, 0)

    def test_still_fails_when_a_count_gets_worse(self):
        baseline = [
            {
                "id": "hr-vacation-entitlement",
                "kind": "fact",
                "answer": "15 working days",
                "label": "correct",
                "citation_ok": "yes",
            }
        ]
        results = [
            {
                "id": "hr-vacation-entitlement",
                "kind": "fact",
                "answer": "20 working days",
                "label": "wrong",
                "citation_ok": "yes",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            baseline_path = folder / "baseline.csv"
            results_path = folder / "results.csv"
            write_csv(baseline_path, baseline)
            write_csv(results_path, results)
            with redirect_stdout(io.StringIO()):
                code = compare.main([str(results_path), "--baseline", str(baseline_path)])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
