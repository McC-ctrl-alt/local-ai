"""Policy tests. Ollama is mocked. Retrieval still reads the Northstar PDFs."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import engine
import policy_rag
from policy_rag import load_policy

# Paraphrase of the HR page 1 onboarding row. It does not say "Week 1".
QUESTION = (
    "During a new hire's opening week on the job, what does their manager "
    "need to confirm?"
)


class ScoreOnWeek:
    def __init__(self, week_score=4.0, other_score=-2.0):
        self.pairs = None
        self.batch_size = None
        self.week_score = week_score
        self.other_score = other_score

    def predict(self, pairs, batch_size=8):
        self.pairs = list(pairs)
        self.batch_size = batch_size
        return [
            self.week_score if "Week 1" in passage else self.other_score
            for _question, passage in pairs
        ]


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._cache = patch("engine.CACHE_DIR", Path(cls._tmp.name) / "cache")
        cls._cache.start()

    @classmethod
    def tearDownClass(cls):
        cls._cache.stop()
        cls._tmp.cleanup()

    def setUp(self):
        policy_rag._RERANKER = None
        self.endpoints = []
        self.chat_payloads = []

    def tearDown(self):
        policy_rag._RERANKER = None

    def test_each_role_has_one_named_rule(self):
        roles = load_policy()["roles"]
        self.assertEqual(
            set(roles),
            {"hr_partner", "legal_counsel", "tax_analyst", "managing_partner"},
        )
        self.assertEqual(roles["hr_partner"]["departments"], ["hr"])
        self.assertEqual(roles["legal_counsel"]["departments"], ["legal"])
        self.assertEqual(roles["tax_analyst"]["departments"], ["tax"])
        self.assertEqual(set(roles["managing_partner"]["departments"]), {"hr", "legal", "tax"})
        for spec in roles.values():
            self.assertTrue(spec["rule"]["id"])
            self.assertTrue(spec["rule"]["text"])
        self.assertEqual(roles["hr_partner"]["rule"]["id"], "HR-01")

    def _fake_request(self, endpoint, payload):
        self.endpoints.append(endpoint)
        if endpoint == "embed":
            text = payload["input"]
            vector = [1.0, 0.0] if text == QUESTION or "Week 1" in text else [0.0, 1.0]
            return {"embeddings": [vector]}
        if endpoint == "chat":
            self.chat_payloads.append(payload)
            user = payload["messages"][1]["content"]
            evidence_id = "E1"
            for line in user.splitlines():
                if line.startswith("[E") and "Week 1" in line:
                    evidence_id = line.split("]", 1)[0][1:]
                    break
            body = json.dumps(
                {
                    "answer": (
                        "In the first week the manager confirms goals, client "
                        "assignment, required training, accessibility and "
                        "accommodation needs, and the 30/60/90-day check-ins."
                    ),
                    "evidence_ids": [evidence_id],
                }
            )
            return {"message": {"content": body}}
        raise AssertionError(endpoint)

    def ask(self, role, reranker):
        with (
            redirect_stdout(io.StringIO()),
            patch("engine.ollama_request", side_effect=self._fake_request),
            patch("engine.load_reranker", return_value=reranker),
        ):
            return policy_rag.consult(QUESTION, role)

    def test_hr_partner_paraphrase_cites_hr_page_one(self):
        self.assertNotIn("Week 1", QUESTION)
        self.assertNotIn("onboarding", QUESTION.casefold())
        reranker = ScoreOnWeek()
        result = self.ask("hr_partner", reranker)

        self.assertEqual(engine.CANDIDATE_COUNT, 10)
        self.assertEqual(engine.RERANK_ABSTAIN_BELOW, -0.7)
        self.assertEqual(
            engine.RERANKER_MODEL,
            os.environ.get("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        )
        self.assertEqual(len(reranker.pairs), 10)
        self.assertEqual(reranker.batch_size, 8)
        self.assertEqual(len(self.chat_payloads), 1)
        payload = self.chat_payloads[0]
        self.assertEqual(payload["model"], engine.CHAT_MODEL)
        self.assertEqual(
            payload["options"],
            {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 128},
        )
        self.assertIn(QUESTION, payload["messages"][1]["content"])
        cited = [
            citation
            for citation in result["citations"]
            if citation.get("source_file") == "HR_Department.pdf"
            and citation.get("page_number") == 1
        ]
        self.assertTrue(cited)
        self.assertEqual(result["trail"]["role"], "hr_partner")
        self.assertEqual(result["trail"]["rule"]["id"], "HR-01")
        self.assertIn("hr", result["trail"]["departments"])
        self.assertTrue(
            any(source == "HR_Department.pdf p. 1" for source in result["trail"]["sources"])
        )
        self.assertNotIn("can't access", result["answer"])

    def test_tax_analyst_is_refused_on_the_same_question(self):
        reranker = ScoreOnWeek()
        result = self.ask("tax_analyst", reranker)
        rule = load_policy()["roles"]["hr_partner"]["rule"]

        self.assertNotIn("chat", self.endpoints)
        self.assertIn("embed", self.endpoints)
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["stage"], "refused")
        self.assertIn(rule["id"], result["answer"])
        self.assertIn(rule["text"], result["answer"])
        self.assertIn("tax_analyst", result["answer"])
        self.assertEqual(result["trail"]["role"], "tax_analyst")
        self.assertEqual(result["trail"]["rule"], {"id": rule["id"], "text": rule["text"]})
        self.assertEqual(result["trail"]["departments"], ["hr"])
        self.assertEqual(result["trail"]["sources"], [])

    def test_score_below_cutoff_skips_qwen(self):
        reranker = ScoreOnWeek(week_score=-2.0, other_score=-2.0)
        result = self.ask("hr_partner", reranker)

        self.assertEqual(len(reranker.pairs), 10)
        self.assertNotIn("chat", self.endpoints)
        self.assertEqual(result["answer"], "I don't know.")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["stage"], "cutoff")
        self.assertIsNone(result["trail"]["rule"])
        self.assertNotIn("can't access", result["answer"])


if __name__ == "__main__":
    unittest.main()
