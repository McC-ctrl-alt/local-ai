"""Demo CLI tests. Ollama is not called."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import demo
import policy_rag
from policy_rag import load_policy


def outcome_for(policy, department_id, allowed=True):
    department = policy["departments"][department_id]
    citation = {
        "source": "docs/hr/remote-work-policy.pdf",
        "page_number": 2,
        "quote": "An employee may work remotely for up to 3 days each week.",
        "chunk_id": "chunk-000",
    }
    return {
        "answer": "Remote work is up to 3 days each week." if allowed else "Refused by HR-01: no.",
        "citations": [citation] if allowed else [],
        "decisions": [
            {
                "department_id": department_id,
                "reason": department["description"],
                "department": department,
                "rule": department["rules"][0],
                "allowed": allowed,
            }
        ],
    }


class AskTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_unknown_role_is_rejected_before_retrieval(self):
        with patch("policy_rag.consult") as consult:
            with self.assertRaises(ValueError) as caught:
                demo.ask("When is the voucher due?", "intern")
        consult.assert_not_called()
        message = str(caught.exception)
        self.assertIn("intern", message)
        self.assertIn("tax_analyst", message)
        self.assertIn("hr_partner", message)
        self.assertIn("counsel", message)

    def test_ask_returns_the_answer_citation_and_trail_for_any_question(self):
        outcome = outcome_for(self.policy, "hr")
        with patch("policy_rag.consult", return_value=outcome) as consult:
            result = demo.ask("  What is the remote work limit?  ", " people_ops ")
        self.assertEqual(consult.call_args.args[1], "people_ops")
        self.assertEqual(consult.call_args.args[2], "What is the remote work limit?")
        self.assertEqual(result["role"], "people_ops")
        self.assertEqual(result["answer"], outcome["answer"])
        self.assertEqual(result["citations"][0]["file"], "remote-work-policy.pdf")
        self.assertEqual(result["citations"][0]["page"], 2)
        self.assertIn("HR-01 allow", result["trail"]["rules"])
        self.assertIn("HR-02 docs/hr only", result["trail"]["rules"])
        self.assertTrue(result["trail"]["departments"][0].startswith("hr — "))
        self.assertEqual(
            result["trail"]["sources"],
            ["remote-work-policy.pdf, page 2"],
        )

    def test_three_roles_are_passed_through_unchanged(self):
        questions = {
            "tax_analyst": "When is the estimated tax voucher due?",
            "hr_partner": "How long is parental leave?",
            "counsel": "What marking does a legal email need?",
        }
        for role, question in questions.items():
            department = "tax" if role.startswith("tax") else "hr" if role.startswith("hr") else "legal"
            outcome = outcome_for(self.policy, department)
            with patch("policy_rag.consult", return_value=outcome) as consult:
                result = demo.ask(question, role)
            self.assertEqual(consult.call_args.args[1:], (role, question))
            self.assertEqual(result["role"], role)
            self.assertEqual(result["question"], question)

    def test_ask_runs_cosine_rerank_and_one_qwen_call(self):
        chunks = []
        for index in range(12):
            text = f"fact {index} about the estimated tax voucher"
            chunks.append(
                {
                    "chunk_id": f"chunk-{index:03d}-abc",
                    "source": "docs/tax/estimated-tax-calendar.pdf",
                    "page_numbers": [2],
                    "page_segments": [{"page_number": 2, "text": text}],
                    "sentences": [{"page_number": 2, "text": text}],
                    "text": text,
                }
            )

        class Reranker:
            def __init__(self):
                self.pairs = None
                self.batch_size = None

            def predict(self, pairs, batch_size=8):
                self.pairs = list(pairs)
                self.batch_size = batch_size
                return [0.5 for _pair in pairs]

        def embed(text):
            if text.startswith("fact "):
                return [float(text.split()[1]), 1.0]
            return [1.0, 0.0]

        reranker = Reranker()
        body = json.dumps(
            {
                "answer": "fact 11 about the estimated tax voucher.",
                "evidence_ids": ["E1"],
            }
        )
        policy_rag._RERANKER = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with (
                    patch("engine.embed", side_effect=embed),
                    patch("engine.load_reranker", return_value=reranker),
                    patch(
                        "engine.ollama_request",
                        return_value={"message": {"content": body}},
                    ) as request,
                    patch(
                        "policy_rag.load_department_chunks",
                        return_value=([Path("estimated-tax-calendar.pdf")], chunks),
                    ),
                    patch("policy_rag.folder_hash", return_value="abc"),
                    patch("policy_rag.CACHE_PATH", Path(tmp) / "cache.json"),
                ):
                    result = demo.ask(
                        "When is the second quarterly estimated tax voucher due?",
                        "tax_partner",
                    )
        finally:
            policy_rag._RERANKER = None

        self.assertEqual(request.call_count, 1)
        self.assertEqual(len(reranker.pairs), 10)
        self.assertEqual(reranker.batch_size, 8)
        self.assertEqual(result["role"], "tax_partner")
        self.assertEqual(result["answer"], "fact 11 about the estimated tax voucher.")
        self.assertEqual(result["citations"][0]["file"], "estimated-tax-calendar.pdf")
        self.assertEqual(result["citations"][0]["page"], 2)
        self.assertIn("TAX-01 allow", result["trail"]["rules"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_one_shot_prints_answer_citation_and_trail(self):
        outcome = outcome_for(self.policy, "legal")
        with patch("policy_rag.consult", return_value=outcome):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = demo.main(
                    [
                        "--role",
                        "general_counsel",
                        "--question",
                        "What does the privilege protocol require?",
                    ]
                )
        self.assertEqual(code, 0)
        output = buffer.getvalue()
        self.assertIn(f"Answer: {outcome['answer']}", output)
        self.assertIn("Citation: remote-work-policy.pdf, page 2", output)
        self.assertIn("Role: general_counsel", output)
        self.assertIn("Rules:", output)
        self.assertIn("Departments:", output)
        self.assertIn("Sources: remote-work-policy.pdf, page 2", output)

    def test_unknown_role_flag_exits(self):
        with patch("policy_rag.consult") as consult:
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    demo.main(["--role", "intern", "--question", "Hello"])
        self.assertEqual(caught.exception.code, 2)
        consult.assert_not_called()

    def test_interactive_prompts_for_role_then_stops_on_a_blank_question(self):
        outcome = outcome_for(self.policy, "tax")
        stdin = io.StringIO("nope\ntax_analyst\nWhen is the voucher due?\n\n")
        with (
            patch("sys.stdin", stdin),
            patch("policy_rag.consult", return_value=outcome) as consult,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = demo.main([])
        self.assertEqual(code, 0)
        consult.assert_called_once()
        self.assertEqual(consult.call_args.args[1], "tax_analyst")
        self.assertEqual(consult.call_args.args[2], "When is the voucher due?")
        self.assertIn("Unknown role 'nope'", buffer.getvalue())
        self.assertIn("Citation:", buffer.getvalue())

    def test_question_without_a_role_prompts_then_asks_that_question(self):
        outcome = outcome_for(self.policy, "legal")
        stdin = io.StringIO("counsel\n\n")
        with (
            patch("sys.stdin", stdin),
            patch("policy_rag.consult", return_value=outcome) as consult,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = demo.main(
                    ["--question", "What marking does the subject line need?"]
                )
        self.assertEqual(code, 0)
        consult.assert_called_once()
        self.assertEqual(consult.call_args.args[1], "counsel")
        self.assertEqual(
            consult.call_args.args[2],
            "What marking does the subject line need?",
        )
        self.assertIn("Role: counsel", buffer.getvalue())

    def test_role_flag_without_a_question_skips_the_role_prompt(self):
        outcome = outcome_for(self.policy, "hr", allowed=False)
        outcome["answer"] = "Refused by HR-01: HR documents may be read only by the hr_partner and people_ops roles."
        stdin = io.StringIO("How many weeks of leave?\n\n")
        with (
            patch("sys.stdin", stdin),
            patch("policy_rag.consult", return_value=outcome) as consult,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = demo.main(["--role", "counsel"])
        self.assertEqual(code, 0)
        consult.assert_called_once()
        self.assertEqual(consult.call_args.args[1], "counsel")
        self.assertIn("Citation: none", buffer.getvalue())
        self.assertIn("HR-01 refuse", buffer.getvalue())
        self.assertIn("Sources: none", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
