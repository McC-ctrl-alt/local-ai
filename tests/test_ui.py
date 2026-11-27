"""Demo server tests. consult is mocked, so no model is loaded."""

import json
import threading
import unittest
from http.client import HTTPConnection
from unittest.mock import patch

import ui


ANSWER = (
    "Manager Confirm goals, client assignment, required training, "
    "accessibility and accommodation needs, and the 30/60/90-day check-ins."
)


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ui.make_server(0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection(ui.HOST, self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        content_type = response.getheader("Content-Type")
        connection.close()
        return status, content_type, raw

    def test_server_listens_on_localhost_only(self):
        self.assertEqual(self.httpd.server_address[0], "127.0.0.1")
        self.assertEqual(ui.HOST, "127.0.0.1")

    def test_get_root_serves_the_offline_page(self):
        status, content_type, raw = self.request("GET", "/")
        page = raw.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("Viewing as", page)
        self.assertIn("hr_partner", page)
        self.assertIn("legal_counsel", page)
        self.assertIn("tax_analyst", page)
        self.assertIn("managing_partner", page)
        self.assertIn('const USER_NAME = "Liam"', page)
        self.assertIn("Week 1 onboarding", page)
        self.assertIn("vacation entitlement", page)
        self.assertIn("Ontario HST", page)
        self.assertIn("What should a manager confirm during Week 1?", page)
        self.assertIn("What is the standard vacation entitlement for a full year?", page)
        self.assertIn("What HST rate applies to taxable services supplied in Ontario?", page)
        self.assertIn("Why this answer", page)
        self.assertIn("Searching 3 departments…", page)
        self.assertIn("Running locally · no internet", page)
        self.assertIn("curator-theme", page)
        self.assertIn("#1c1c1e", page)
        self.assertIn("#3b6ef5", page)
        self.assertIn("#1f4d3a", page)
        self.assertNotIn("http://", page)
        self.assertNotIn("https://", page)
        self.assertNotIn("@import", page)

    def test_post_ask_returns_consult_json_and_calls_nothing_else(self):
        result = {
            "answer": ANSWER,
            "citations": [
                {
                    "source_file": "HR_Department.pdf",
                    "page_number": 1,
                    "quote": "Confirm goals, client assignment, required training",
                    "chunk_id": "chunk-002",
                }
            ],
            "trail": {
                "role": "hr_partner",
                "rule": {
                    "id": "HR-01",
                    "text": "HR documents may be read only by the hr_partner and the managing_partner.",
                },
                "departments": ["hr"],
                "sources": ["HR_Department.pdf p. 1"],
            },
            "stage": "answered",
        }
        question = "During a new hire's opening week on the job, what does their manager need to confirm?"
        with (
            patch("policy_rag.consult", return_value=result) as consult,
            patch("engine.embed", side_effect=AssertionError("embed")),
            patch("engine.chat_answer", side_effect=AssertionError("chat")),
            patch("engine.load_reranker", side_effect=AssertionError("reranker")),
            patch("engine.load_corpus", side_effect=AssertionError("corpus")),
        ):
            status, content_type, raw = self.request(
                "POST",
                "/ask",
                body=json.dumps({"question": question, "role": "hr_partner"}).encode(),
                headers={"Content-Type": "application/json"},
            )
        payload = json.loads(raw.decode("utf-8"))
        consult.assert_called_once_with(question, "hr_partner")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(payload["answer"], ANSWER)
        self.assertTrue(payload["answer"].startswith("Manager "))
        self.assertEqual(payload["citations"], result["citations"])
        self.assertEqual(payload["trail"], result["trail"])
        self.assertEqual(payload["stage"], "answered")

    def test_post_ask_does_not_call_consult_for_a_bad_body(self):
        with patch("policy_rag.consult") as consult:
            status, _content_type, raw = self.request(
                "POST",
                "/ask",
                body=b"{",
                headers={"Content-Type": "application/json"},
            )
        consult.assert_not_called()
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(raw.decode("utf-8")))


if __name__ == "__main__":
    unittest.main()
