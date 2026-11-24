"""Retrieval tests. Ollama and the cross-encoder are mocked, so no model is loaded."""

import json
import os
import unittest
from io import BytesIO
from unittest.mock import patch

from pypdf import PdfReader

import curator
import engine
from scripts.build_docs import build_pdf


def sentence(label, letter, count=200):
    return f"{label} " + (letter * count) + "."


class ChunkTests(unittest.TestCase):
    def test_curator_uses_the_engine_chunker(self):
        self.assertIs(curator.build_page_aware_chunks, engine.build_page_aware_chunks)
        self.assertIs(curator.rank_candidates, engine.rank_candidates)
        self.assertIs(curator.rerank_candidates, engine.rerank_candidates)
        self.assertIs(curator.embed, engine.embed)
        self.assertEqual(curator.RERANK_ABSTAIN_BELOW, -0.7)
        self.assertEqual(engine.RERANK_ABSTAIN_BELOW, -0.7)

    def test_pages_stay_on_the_chunk_and_the_default_record_is_unchanged(self):
        pdf = build_pdf([["Page one sentence."], ["Page two sentence."]])
        reader = PdfReader(BytesIO(pdf))
        plain = engine.build_page_aware_chunks(reader)
        tagged = engine.build_page_aware_chunks(reader, source="docs/tax/note.pdf", index_start=4)

        self.assertEqual(len(plain), 1)
        self.assertEqual(plain[0]["page_numbers"], [1, 2])
        self.assertNotIn("source", plain[0])
        self.assertNotIn("sentences", plain[0])
        self.assertTrue(plain[0]["chunk_id"].startswith("chunk-000-"))

        self.assertEqual([chunk["text"] for chunk in plain], [chunk["text"] for chunk in tagged])
        self.assertEqual(tagged[0]["source"], "docs/tax/note.pdf")
        self.assertTrue(tagged[0]["chunk_id"].startswith("chunk-004-"))
        self.assertEqual(
            plain[0]["chunk_id"].rsplit("-", 1)[-1],
            tagged[0]["chunk_id"].rsplit("-", 1)[-1],
        )
        self.assertEqual(
            [item["text"] for item in tagged[0]["sentences"]],
            ["Page one sentence", "Page two sentence"],
        )

    def test_pack_is_about_1000_characters_with_about_300_of_overlap(self):
        # Six sentences, each a bit over 200 characters. The fifth does not fit
        # in the 1000-character pack, and only the last sentence of the first
        # pack is short enough to carry forward as overlap.
        lines = [
            sentence("Alpha", "a"),
            sentence("Beta", "b"),
            sentence("Gamma", "c"),
            sentence("Delta", "d"),
            sentence("Epsilon", "e"),
            sentence("Zeta", "f"),
        ]
        reader = PdfReader(BytesIO(build_pdf([lines])))
        chunks = engine.build_page_aware_chunks(reader)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0]["text"].startswith("Alpha "))
        self.assertIn("Delta ", chunks[0]["text"])
        self.assertNotIn("Epsilon ", chunks[0]["text"])
        self.assertTrue(chunks[1]["text"].startswith("Delta "))
        self.assertIn("Zeta ", chunks[1]["text"])
        self.assertNotIn("Alpha ", chunks[1]["text"])
        self.assertLessEqual(len(chunks[0]["text"]), 1000)
        self.assertLessEqual(len(chunks[1]["text"]), 1000)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = []
        self.embeddings = []
        for index in range(12):
            self.chunks.append(
                {
                    "chunk_id": f"c{index}",
                    "text": f"sentence {index}",
                    "page_numbers": [1],
                    "page_segments": [{"page_number": 1, "text": f"sentence {index}"}],
                }
            )
            self.embeddings.append([float(index), 1.0])
        self.index = {"chunks": self.chunks, "embeddings": self.embeddings}

    def test_cosine_keeps_ten_and_the_reranker_reorders_those_ten(self):
        class Reranker:
            def __init__(self):
                self.pairs = None
                self.batch_size = None

            def predict(self, pairs, batch_size=8):
                self.pairs = list(pairs)
                self.batch_size = batch_size
                # Lowest score on the best cosine hit, so the order has to move.
                return [float(offset) for offset in range(len(pairs))]

        reranker = Reranker()
        with patch("engine.embed", return_value=[1.0, 0.0]) as embed:
            ranked = engine.rank_candidates("what is sentence 11", self.index)
            reranked = engine.rerank_candidates("what is sentence 11", ranked, reranker)

        embed.assert_called_once_with("what is sentence 11")
        self.assertEqual(len(ranked), 10)
        self.assertEqual(len(reranked), 10)
        self.assertEqual(len(reranker.pairs), 10)
        self.assertEqual(reranker.batch_size, 8)
        # Highest cosine is chunk 11. It is one of the ten, and the reranker
        # scores that pair lowest, so it must not stay first.
        cosine_ids = [chunk["chunk_id"] for _score, _index, chunk in ranked]
        self.assertEqual(cosine_ids[0], "c11")
        self.assertNotIn("c0", cosine_ids)
        self.assertNotIn("c1", cosine_ids)
        reranked_ids = [chunk["chunk_id"] for _score, _cosine, _index, chunk in reranked]
        self.assertEqual(reranked_ids[0], "c2")
        self.assertEqual(reranked_ids[-1], "c11")
        self.assertGreater(reranked[0][0], reranked[-1][0])

    def test_reranker_loads_the_cross_encoder_on_cpu(self):
        calls = []

        def factory(model, backend, device):
            calls.append((model, backend, device))
            if backend == "onnx":
                raise RuntimeError("onnx extra is not installed")
            return object()

        with patch("sentence_transformers.CrossEncoder", side_effect=factory):
            produced = engine.load_reranker()

        self.assertIsNotNone(produced)
        self.assertEqual(
            calls,
            [
                (engine.RERANKER_MODEL, "onnx", "cpu"),
                (engine.RERANKER_MODEL, "torch", "cpu"),
            ],
        )
        self.assertEqual(engine.RERANKER_MODEL, os.environ.get(
            "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
        ))


class AnswerCallTests(unittest.TestCase):
    def index(self):
        chunk = {
            "chunk_id": "chunk-000-abc",
            "text": "The standard vacation entitlement is 15 working days.",
            "page_numbers": [1],
            "page_segments": [
                {
                    "page_number": 1,
                    "text": "The standard vacation entitlement is 15 working days",
                }
            ],
            "source_file": "HR_Department.pdf",
        }
        return {"chunks": [chunk], "embeddings": [[1.0, 0.0]]}

    def test_cutoff_is_exclusive_of_minus_0_7(self):
        self.assertFalse(engine.rerank_below_cutoff(-0.7))
        self.assertFalse(engine.rerank_below_cutoff(0.32))
        self.assertFalse(engine.rerank_below_cutoff(None))
        self.assertTrue(engine.rerank_below_cutoff(-0.7000001))
        self.assertTrue(engine.rerank_below_cutoff(-1.78))

    def test_score_below_cutoff_skips_qwen(self):
        class Reranker:
            def predict(self, pairs, batch_size=8):
                self.pairs = list(pairs)
                return [-1.78]

        reranker = Reranker()
        with (
            patch("engine.embed", return_value=[1.0, 0.0]),
            patch("engine.chat_answer") as chat,
        ):
            result = curator.ask_question(
                "What is the employer RRSP match?",
                self.index(),
                "HR_Department.pdf",
                reranker,
                quiet=True,
            )

        chat.assert_not_called()
        self.assertEqual(len(reranker.pairs), 1)
        self.assertEqual(result["answer"], "I don't know.")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["prompt"], "")
        self.assertEqual(result["reranked"][0][0], -1.78)

    def test_score_above_cutoff_makes_one_qwen_call(self):
        class Reranker:
            def predict(self, pairs, batch_size=8):
                return [4.87]

        payload = json.dumps(
            {"answer": "I don't know.", "evidence_ids": []}
        )
        with (
            patch("engine.embed", return_value=[1.0, 0.0]),
            patch("engine.chat_answer", return_value=payload) as chat,
        ):
            result = curator.ask_question(
                "What is the standard vacation entitlement for a full year?",
                self.index(),
                "HR_Department.pdf",
                Reranker(),
                quiet=True,
            )

        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["answer"], "I don't know.")

    def test_chat_options_pin_temperature_seed_context_and_model(self):
        with patch(
            "engine.ollama_request",
            return_value={"message": {"content": "ok"}},
        ) as request:
            text = engine.chat_answer("system", "user", {"type": "object"}, max_tokens=128)

        self.assertEqual(text, "ok")
        endpoint, payload = request.call_args[0]
        self.assertEqual(endpoint, "chat")
        self.assertEqual(payload["model"], engine.CHAT_MODEL)
        self.assertEqual(
            engine.CHAT_MODEL, os.environ.get("CURATOR_LLM", "qwen2.5:3b")
        )
        self.assertEqual(
            payload["options"],
            {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 128},
        )
        self.assertEqual(payload["format"], {"type": "object"})


if __name__ == "__main__":
    unittest.main()
