import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

from policy_rag import (
    build_page_aware_chunks,
    decide_access,
    is_abstention,
    load_department_chunks,
    load_policy,
    order_reranked,
    parse_route,
    refusal_text,
    run_question,
    split_access,
    validate_citations,
)
from scripts.build_docs import DOCUMENTS, build_pdf

ROOT = Path(__file__).resolve().parents[1]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_three_departments_each_have_folder_roles_and_rules(self):
        departments = self.policy["departments"]
        self.assertEqual(set(departments), {"tax", "legal", "hr"})
        for department in departments.values():
            self.assertTrue(department["folder"].startswith("docs/"))
            self.assertGreaterEqual(len(department["allowed_roles"]), 2)
            self.assertGreaterEqual(len(department["rules"]), 3)
            self.assertTrue(any(rule["kind"] == "access" for rule in department["rules"]))

    def test_denied_role_is_refused_with_the_access_rule(self):
        allowed, rule = decide_access(self.policy["departments"]["legal"], "tax_analyst")
        self.assertFalse(allowed)
        text = refusal_text(rule)
        self.assertIn("LEG-01", text)
        self.assertEqual(text, f"Refused by {rule['id']}: {rule['text']}")
        self.assertIn("counsel", rule["text"])

    def test_allowed_role_passes(self):
        allowed, rule = decide_access(self.policy["departments"]["hr"], "hr_partner")
        self.assertTrue(allowed)
        self.assertEqual(rule["id"], "HR-01")

    def test_reranker_score_reorders_cosine_candidates(self):
        candidates = [
            (0.9, {"chunk_id": "a"}),
            (0.2, {"chunk_id": "b"}),
        ]
        ordered = order_reranked(candidates, [-1.0, 3.5])
        self.assertEqual([item[2]["chunk_id"] for item in ordered], ["b", "a"])
        self.assertEqual(ordered[0][1], 0.2)

    def test_denied_question_cites_the_rule_and_opens_nothing(self):
        with (
            patch(
                "policy_rag.route_question",
                return_value=[("legal", "legal advice")],
            ),
            patch("policy_rag.load_department_chunks") as load_chunks,
            patch("policy_rag.get_reranker") as get_reranker,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = run_question(
                    self.policy,
                    "tax_analyst",
                    "What marking must the subject line include when an email seeks legal advice?",
                    "refusal",
                )
        self.assertEqual(status, 0)
        load_chunks.assert_not_called()
        get_reranker.assert_not_called()
        output = buffer.getvalue()
        self.assertIn("Role: tax_analyst", output)
        self.assertIn("LEG-01 refuse", output)
        self.assertIn("Refused by LEG-01:", output)
        self.assertIn("Cited: none", output)

    def test_router_can_return_one_or_more_departments(self):
        routes = parse_route(
            '{"departments": [{"department": "Tax", "reason": "Estimated tax date."}]}',
            {"tax", "legal", "hr"},
        )
        self.assertEqual(routes, [("tax", "Estimated tax date.")])
        routes = parse_route(
            json.dumps(
                {
                    "departments": [
                        {"department": "legal", "reason": "Contracts."},
                        {"department": "tax", "reason": "Meals."},
                        {"department": "legal", "reason": "Again."},
                    ]
                }
            ),
            {"tax", "legal", "hr"},
        )
        self.assertEqual(routes, [("legal", "Contracts."), ("tax", "Meals.")])
        with self.assertRaises(ValueError):
            parse_route(
                '{"departments": [{"department": "finance", "reason": "no"}]}',
                {"tax"},
            )

    def test_each_routed_department_is_allowed_or_refused(self):
        decisions, allowed, denied = split_access(
            self.policy,
            "counsel",
            [("legal", "contract review"), ("tax", "meal deduction")],
        )
        self.assertEqual([item["department_id"] for item in allowed], ["legal"])
        self.assertEqual([item["department_id"] for item in denied], ["tax"])
        self.assertEqual(denied[0]["rule"]["id"], "TAX-01")
        self.assertEqual(
            [item["department_id"] for item in decisions],
            ["legal", "tax"],
        )

    def test_mixed_route_opens_only_allowed_folders(self):
        chunk = {
            "chunk_id": "chunk-000-abc",
            "source": "docs/legal/contract-review-standard.pdf",
            "page_numbers": [2],
            "page_segments": [
                {
                    "page_number": 2,
                    "text": "A standard vendor contract is reviewed within 10 business days",
                }
            ],
            "sentences": [
                {
                    "page_number": 2,
                    "text": "A standard vendor contract is reviewed within 10 business days",
                }
            ],
            "text": "A standard vendor contract is reviewed within 10 business days",
        }
        answer = {
            "answer": "Review takes 10 business days.",
            "delivered": "Review takes 10 business days.",
            "citations": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": 2,
                    "quote": chunk["text"],
                    "source": chunk["source"],
                }
            ],
            "valid": True,
            "problems": [],
        }
        with (
            patch(
                "policy_rag.route_question",
                return_value=[
                    ("legal", "contract review standard"),
                    ("tax", "client meal deduction"),
                ],
            ),
            patch(
                "policy_rag.load_department_chunks",
                return_value=([Path("contract.pdf")], [chunk]),
            ) as load_chunks,
            patch("policy_rag.folder_hash", return_value="abc"),
            patch("policy_rag.embeddings_for", return_value=([[1.0, 0.0]], True)),
            patch("policy_rag.rank_by_cosine", return_value=[(0.8, chunk)]),
            patch(
                "policy_rag.rerank_candidates",
                return_value=[(2.0, 0.8, chunk)],
            ),
            patch("policy_rag.answer_question", return_value=answer) as answer_call,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                status = run_question(
                    self.policy,
                    "counsel",
                    "How many business days, and what meal percentage?",
                    "cross-dept",
                )
        self.assertEqual(status, 0)
        load_chunks.assert_called_once()
        self.assertTrue(str(load_chunks.call_args[0][0]).endswith("docs/legal"))
        answer_call.assert_called_once()
        sent_chunks = answer_call.call_args[0][2]
        self.assertEqual(list(sent_chunks), ["E1"])
        output = buffer.getvalue()
        self.assertIn("Refused by TAX-01:", output)
        self.assertIn("Not opened: docs/tax, docs/hr", output)
        self.assertIn("contract-review-standard.pdf p. 2", output)
        self.assertIn("cosine: chunk, page, score", output)

    def test_sample_facts_sit_on_page_two_of_six_pdfs(self):
        expected = {
            "docs/tax/estimated-tax-calendar.pdf": "June 15",
            "docs/tax/meals-deduction-memo.pdf": "50 percent",
            "docs/legal/privilege-protocol.pdf": "Privileged and Confidential",
            "docs/legal/contract-review-standard.pdf": "10 business days",
            "docs/hr/parental-leave-policy.pdf": "16 weeks",
            "docs/hr/remote-work-policy.pdf": "3 days",
        }
        self.assertEqual(len(DOCUMENTS), 6)
        for path, fact in expected.items():
            reader = PdfReader(str(ROOT / path))
            self.assertGreaterEqual(len(reader.pages), 2)
            self.assertIn(fact, reader.pages[1].extract_text())
            self.assertIn("FICTIONAL SAMPLE", reader.pages[0].extract_text())

    def test_chunks_keep_the_source_file_and_only_that_department(self):
        pdfs, chunks = load_department_chunks(ROOT / "docs" / "tax")
        self.assertEqual(len(pdfs), 2)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertTrue(chunk["source"].startswith("docs/tax/"))
            self.assertIn(2, chunk["page_numbers"])
            self.assertTrue(chunk["chunk_id"].startswith("chunk-"))

    def test_citation_must_match_the_cited_page(self):
        reader = PdfReader(str(ROOT / "docs" / "tax" / "estimated-tax-calendar.pdf"))
        chunks = build_page_aware_chunks(reader, "docs/tax/estimated-tax-calendar.pdf")
        segment = next(
            segment
            for chunk in chunks
            for segment in chunk["page_segments"]
            if "June 15" in segment["text"]
        )
        chunk = next(chunk for chunk in chunks if segment in chunk["page_segments"])
        citation = {
            "chunk_id": chunk["chunk_id"],
            "page_number": segment["page_number"],
            "quote": segment["text"],
        }
        valid, problems = validate_citations(
            "The second quarterly voucher is due on June 15.",
            [citation],
            chunks,
        )
        self.assertTrue(valid, problems)
        valid, problems = validate_citations(
            "The voucher is due on July 4.",
            [citation],
            chunks,
        )
        self.assertFalse(valid)
        self.assertTrue(problems)
        june = next(
            item for chunk in chunks for item in chunk["sentences"] if "June 15" in item["text"]
        )
        june_chunk = next(
            chunk for chunk in chunks if june in chunk["sentences"]
        )
        june_citation = {
            "chunk_id": june_chunk["chunk_id"],
            "page_number": june["page_number"],
            "quote": june["text"],
        }
        valid, problems = validate_citations(
            "The second quarterly voucher is due on June 15.",
            [june_citation],
            chunks,
        )
        self.assertTrue(valid, problems)
        valid, _problems = validate_citations(
            "The second quarterly voucher is due on April 15.",
            [june_citation],
            chunks,
        )
        self.assertFalse(valid)
        valid, _problems = validate_citations("I don't know.", [], chunks)
        self.assertTrue(valid)

    def test_bare_abstention_is_recognized(self):
        self.assertTrue(is_abstention("I don't know."))
        self.assertTrue(is_abstention("I don't know"))
        self.assertFalse(
            is_abstention(
                "Review takes 10 business days, and I don't know the meal rate."
            )
        )

    def test_builder_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.pdf"
            path.write_bytes(build_pdf([["Page one."], ["The stipend is 750 dollars."]]))
            reader = PdfReader(str(path))
            self.assertIn("750 dollars", reader.pages[1].extract_text())


class DemoFileTests(unittest.TestCase):
    def test_five_demo_questions_cover_the_routes(self):
        questions = json.loads((ROOT / "demo_questions.json").read_text())
        self.assertEqual(len(questions), 5)
        self.assertEqual(
            [item["id"] for item in questions],
            ["routed", "routed-hr", "cross-dept", "refusal", "unknown"],
        )
        roles = {item["role"] for item in questions}
        self.assertTrue(roles <= {"tax_analyst", "hr_partner", "counsel", "people_ops"})


if __name__ == "__main__":
    unittest.main()
