"""Demo UI tests. Ollama is not called."""

import unittest
from unittest.mock import patch

import demo_ui
import engine
import policy_rag


class DemoUiTests(unittest.TestCase):
    def setUp(self):
        self.app = demo_ui.create_app(prewarm=False)
        self.client = self.app.test_client()

    def test_import_and_index_renders_roles_from_policy(self):
        with patch.object(engine, "chat_answer") as chat:
            response = self.client.get("/")
        chat.assert_not_called()
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Local AI demo", html)
        for role in policy_rag.known_roles():
            self.assertIn(role, html)
        self.assertIn("Why this answer", html)
        self.assertIn("127.0.0.1", html)

    def test_ask_returns_the_mocked_answer_citation_and_trail(self):
        result = {
            "question": "How long is parental leave?",
            "role": "hr_partner",
            "answer": "Parental leave is 16 weeks.",
            "citations": [
                {
                    "file": "parental-leave-policy.pdf",
                    "page": 3,
                    "quote": "Parental leave is 16 weeks.",
                    "source": "docs/hr/parental-leave-policy.pdf",
                }
            ],
            "trail": {
                "role": "hr_partner",
                "rules": ["HR-01 allow", "HR-02 docs/hr only"],
                "departments": ["hr — HR folder. Parental leave policy and the remote work policy."],
                "sources": ["parental-leave-policy.pdf, page 3"],
            },
        }
        with patch("demo.ask", return_value=result) as ask:
            response = self.client.post(
                "/ask",
                json={
                    "role": "hr_partner",
                    "question": "How long is parental leave?",
                },
            )
        ask.assert_called_once_with("How long is parental leave?", "hr_partner")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), result)

    def test_ask_returns_400_when_ask_rejects_the_role(self):
        with patch(
            "demo.ask",
            side_effect=ValueError("Unknown role 'intern'. Roles in policy.yaml: tax_analyst."),
        ) as ask:
            response = self.client.post(
                "/ask",
                json={"role": "intern", "question": "When is the voucher due?"},
            )
        ask.assert_called_once()
        self.assertEqual(response.status_code, 400)
        self.assertIn("intern", response.get_json()["error"])

    def test_create_app_does_not_prewarm_by_default(self):
        with patch.object(engine, "chat_answer") as chat:
            demo_ui.create_app()
        chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
