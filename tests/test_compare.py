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
    "top_rerank_score",
]


def question(question_id, label, score):
    return {
        "id": question_id,
        "kind": "fact",
        "answer": "a",
        "label": label,
        "citation_ok": "yes",
        "top_rerank_score": score,
    }


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

    def test_per_question_scores_match_by_id_without_changing_the_result(self):
        baseline = [
            question("q-score", "correct", "1.0000"),
            question("q-tiny", "correct", "1.0000"),
            question("q-boundary", "correct", "2.0000"),
            question("q-down", "correct", "4.0000"),
            question("q-over", "correct", "5.0000"),
            question("q-exact", "partial", "1.0000"),
            question("q-label", "wrong", "1.0000"),
            question("q-baseline-only", "wrong", "9.0000"),
        ]
        results = [
            question("q-label", "partial", "1.0000"),
            question("q-results-only", "correct", "2.0000"),
            question("q-exact", "correct", "1.0100"),
            question("q-over", "correct", "5.0101"),
            question("q-down", "correct", "3.9900"),
            question("q-boundary", "correct", "2.0100"),
            question("q-tiny", "correct", "1.0050"),
            question("q-score", "correct", "0.9800"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            baseline_path = folder / "baseline.csv"
            results_path = folder / "results.csv"
            write_csv(baseline_path, baseline)
            write_csv(results_path, results)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = compare.main([str(results_path), "--baseline", str(baseline_path)])
        self.assertEqual(code, 0)
        lines = {}
        for line in buffer.getvalue().splitlines():
            if line.startswith("q-"):
                lines[line.split()[0]] = line
        self.assertEqual(
            list(lines),
            ["q-score", "q-tiny", "q-boundary", "q-down", "q-over", "q-exact", "q-label"],
        )
        self.assertIn("1.0000", lines["q-score"])
        self.assertIn("0.9800", lines["q-score"])
        self.assertIn("-0.0200", lines["q-score"])
        self.assertIn("correct -> correct", lines["q-score"])
        self.assertIn("CHANGED", lines["q-score"])
        self.assertIn("+0.0050", lines["q-tiny"])
        self.assertNotIn("CHANGED", lines["q-tiny"])
        self.assertIn("+0.0100", lines["q-boundary"])
        self.assertNotIn("CHANGED", lines["q-boundary"])
        self.assertIn("-0.0100", lines["q-down"])
        self.assertNotIn("CHANGED", lines["q-down"])
        self.assertIn("+0.0101", lines["q-over"])
        self.assertIn("CHANGED", lines["q-over"])
        self.assertIn("partial -> correct", lines["q-exact"])
        self.assertIn("CHANGED", lines["q-exact"])
        self.assertIn("wrong -> partial", lines["q-label"])
        self.assertIn("0.0000", lines["q-label"])
        self.assertIn("CHANGED", lines["q-label"])
        self.assertNotIn("q-baseline-only", buffer.getvalue())
        self.assertNotIn("q-results-only", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
