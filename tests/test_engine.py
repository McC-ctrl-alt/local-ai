"""Retrieval tests. Ollama and the cross-encoder are mocked, so no model is loaded."""

import io
import json
import math
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfReader

import curator
import engine
from tests.build_pdf import build_pdf


def python_cosine(vector_a, vector_b):
    """The previous pure-Python score, used only to bound float drift."""
    length_a = math.hypot(*vector_a)
    length_b = math.hypot(*vector_b)
    dot_product = sum(left * right for left, right in zip(vector_a, vector_b))
    return dot_product / (length_a * length_b)


def sentence(label, letter, count=200):
    return f"{label} " + (letter * count) + "."


class ChunkTests(unittest.TestCase):
    def test_curator_uses_the_engine_chunker(self):
        self.assertIs(curator.build_page_aware_chunks, engine.build_page_aware_chunks)
        self.assertIs(curator.rank_candidates, engine.rank_candidates)
        self.assertIs(curator.rerank_candidates, engine.rerank_candidates)
        self.assertIs(curator.embed, engine.embed)
        self.assertIs(curator.list_pdfs, engine.list_pdfs)
        self.assertIs(curator.load_corpus, engine.load_corpus)
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
        question = [1.0, 0.0]
        for score, chunk_index, chunk in ranked:
            self.assertIs(type(score), float)
            self.assertIs(type(chunk_index), int)
            reference = python_cosine(question, self.embeddings[chunk_index])
            self.assertLess(abs(score - reference), 1e-6)
            self.assertEqual(chunk["chunk_id"], f"c{chunk_index}")

    def test_rank_scores_every_chunk_with_one_product_and_keeps_ties(self):
        chunks = []
        embeddings = []
        for name, vector in (
            ("first", [1.0, 0.0]),
            ("later", [1.0, 0.0]),
            ("other", [0.0, 1.0]),
        ):
            chunks.append(
                {
                    "chunk_id": name,
                    "text": name,
                    "page_numbers": [1],
                    "page_segments": [{"page_number": 1, "text": name}],
                }
            )
            embeddings.append(vector)
        index = {"chunks": chunks, "embeddings": embeddings}
        with patch("engine.embed", return_value=[1.0, 0.0]):
            ranked = engine.rank_candidates("same vector twice", index)
        self.assertEqual([item[2]["chunk_id"] for item in ranked], ["first", "later", "other"])
        self.assertEqual([item[1] for item in ranked], [0, 1, 2])
        self.assertAlmostEqual(ranked[0][0], 1.0, places=6)
        self.assertAlmostEqual(ranked[1][0], 1.0, places=6)
        self.assertLess(abs(ranked[2][0]), 1e-6)

    def test_rank_rejects_bad_vectors_before_scoring(self):
        def run(embeddings):
            index = {
                "chunks": [{"chunk_id": str(i), "text": "t", "page_numbers": [1], "page_segments": []} for i in range(len(embeddings))],
                "embeddings": embeddings,
            }
            with patch("engine.embed", return_value=[1.0, 0.0]):
                engine.rank_candidates("question", index)

        with self.assertRaises(ValueError) as mismatch:
            run([[1.0, 0.0, 0.0]])
        self.assertIn("same length", str(mismatch.exception))
        self.assertIn("got 2 and 3", str(mismatch.exception))
        with self.assertRaises(ValueError) as zeros:
            run([[0.0, 0.0]])
        self.assertIn("zero-length", str(zeros.exception))
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError) as non_finite:
                run([[bad, 1.0]])
            self.assertIn("non-finite", str(non_finite.exception))
        empty_index = {
            "chunks": [
                {
                    "chunk_id": "c",
                    "text": "t",
                    "page_numbers": [1],
                    "page_segments": [],
                }
            ],
            "embeddings": [[]],
        }
        with patch("engine.embed", return_value=[]):
            with self.assertRaises(ValueError) as empty:
                engine.rank_candidates("question", empty_index)
        self.assertIn("zero-length", str(empty.exception))

    def test_rerank_rejects_non_finite_scores(self):
        candidates = [(0.2, 0, {"text": "passage", "chunk_id": "c"})]

        class Scores:
            def __init__(self, values):
                self.values = values

            def predict(self, pairs, batch_size=8):
                self.batch_size = batch_size
                return self.values

        for values in ([float("nan")], [float("inf")], [float("-inf")]):
            with self.assertRaises(ValueError) as caught:
                engine.rerank_candidates("question", candidates, Scores(values))
            self.assertIn("invalid scores", str(caught.exception))
        reranked = engine.rerank_candidates("question", candidates, Scores([1.25]))
        self.assertIs(type(reranked[0][0]), float)
        self.assertEqual(reranked[0][0], 1.25)

    def test_reranker_loads_onnx_from_the_local_folder(self):
        calls = []
        sentinel = object()
        local_folder = "/cache/models/cross-encoder"

        def factory(model, *, backend, device, model_kwargs):
            calls.append((model, backend, device, model_kwargs))
            return sentinel

        with (
            patch(
                "huggingface_hub.snapshot_download",
                return_value=local_folder,
            ) as download,
            patch("sentence_transformers.CrossEncoder", side_effect=factory),
        ):
            produced = engine.load_reranker()

        self.assertIs(produced, sentinel)
        download.assert_called_once_with(engine.RERANKER_MODEL, local_files_only=True)
        self.assertEqual(
            calls,
            [
                (
                    local_folder,
                    "onnx",
                    "cpu",
                    {"file_name": "onnx/model.onnx"},
                )
            ],
        )
        self.assertEqual(engine.RERANKER_MODEL, os.environ.get(
            "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
        ))

    def test_reranker_raises_a_clear_error_when_the_local_model_is_missing(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        calls = []

        def factory(model, *, backend, device, model_kwargs):
            calls.append((model, backend, device, model_kwargs))
            raise OSError("offline")

        with (
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=LocalEntryNotFoundError("no local snapshot"),
            ) as download,
            patch("sentence_transformers.CrossEncoder", side_effect=factory),
        ):
            with self.assertRaises(RuntimeError) as caught:
                engine.load_reranker()

        message = str(caught.exception)
        self.assertIn("Run once with internet to download the model", message)
        self.assertIn(engine.RERANKER_MODEL, message)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        download.assert_called_once_with(engine.RERANKER_MODEL, local_files_only=True)
        self.assertEqual(
            calls,
            [
                (
                    engine.RERANKER_MODEL,
                    "onnx",
                    "cpu",
                    {"file_name": "onnx/model.onnx"},
                )
            ],
        )

    def test_reranker_raises_when_the_local_onnx_file_cannot_load(self):
        calls = []
        local_folder = "/cache/models/cross-encoder"

        def factory(model, *, backend, device, model_kwargs):
            calls.append((model, backend, device, model_kwargs))
            raise RuntimeError("onnx file is missing")

        with (
            patch(
                "huggingface_hub.snapshot_download",
                return_value=local_folder,
            ) as download,
            patch("sentence_transformers.CrossEncoder", side_effect=factory),
        ):
            with self.assertRaises(RuntimeError) as caught:
                engine.load_reranker()

        self.assertIn("Run once with internet to download the model", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        download.assert_called_once_with(engine.RERANKER_MODEL, local_files_only=True)
        self.assertEqual(
            calls,
            [
                (
                    local_folder,
                    "onnx",
                    "cpu",
                    {"file_name": "onnx/model.onnx"},
                )
            ],
        )


class CosineTests(unittest.TestCase):
    def test_rank_scores_match_the_old_formula_within_float_rounding(self):
        generator = random.Random(0)
        for width in (2, 8, 768):
            for _ in range(20):
                left = [generator.uniform(-2.0, 2.0) for _ in range(width)]
                right = [generator.uniform(-2.0, 2.0) for _ in range(width)]
                index = {
                    "chunks": [
                        {
                            "chunk_id": "c",
                            "text": "t",
                            "page_numbers": [1],
                            "page_segments": [],
                        }
                    ],
                    "embeddings": [right],
                }
                with patch("engine.embed", return_value=left):
                    ranked = engine.rank_candidates("question", index)
                score, chunk_index, chunk = ranked[0]
                self.assertIs(type(score), float)
                self.assertEqual(chunk_index, 0)
                self.assertEqual(chunk["chunk_id"], "c")
                self.assertLess(abs(score - python_cosine(left, right)), 1e-6)


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


class CorpusLayoutTests(unittest.TestCase):
    def test_northstar_pdfs_live_in_department_folders(self):
        folder = Path(__file__).resolve().parents[1] / "knowledge_test"
        pdfs = engine.list_pdfs(folder)
        by_name = {path.name: path.parent.name for path in pdfs}
        self.assertEqual(
            by_name,
            {
                "HR_Department.pdf": "hr",
                "Legal_Department.pdf": "legal",
                "Tax_Department.pdf": "tax",
            },
        )
        self.assertTrue(all(path.parent != folder for path in pdfs))

    def test_a_new_subfolder_pdf_is_tagged_and_a_changed_file_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            department = root / "hr"
            department.mkdir()
            path = department / "note.pdf"
            path.write_bytes(build_pdf([["The stipend is 750 dollars."]]))
            with (
                redirect_stdout(io.StringIO()),
                patch("engine.CACHE_DIR", root / "cache"),
                patch("engine.embed", return_value=[1.0, 0.0]) as embed,
            ):
                found = engine.list_pdfs(root)
                self.assertEqual(found, [path])
                index = engine.load_corpus(found)
                self.assertEqual(index["chunks"][0]["department"], "hr")
                self.assertEqual(index["chunks"][0]["source_file"], "note.pdf")
                self.assertIn("750", index["chunks"][0]["text"])
                built = embed.call_count
                self.assertGreater(built, 0)

                again = engine.load_corpus(engine.list_pdfs(root))
                self.assertEqual(embed.call_count, built)
                self.assertEqual(again["chunks"][0]["department"], "hr")
                self.assertIn("750", again["chunks"][0]["text"])

                path.write_bytes(build_pdf([["The stipend is 800 dollars."]]))
                later = path.stat().st_mtime_ns + 10_000_000_000
                os.utime(path, ns=(later, later))
                rebuilt = engine.load_corpus(engine.list_pdfs(root))
                self.assertGreater(embed.call_count, built)
                self.assertIn("800", rebuilt["chunks"][0]["text"])
                self.assertNotIn("750", rebuilt["chunks"][0]["text"])
                self.assertEqual(rebuilt["chunks"][0]["department"], "hr")


if __name__ == "__main__":
    unittest.main()
