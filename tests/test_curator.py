"""Checks that do not call Qwen.

Ollama is mocked here. A live model run is separate from these tests.
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import curator
from textpdf import write_text_pdf


def sample_chunk():
    return {
        "chunk_id": "harbor-chunk-000-aaa",
        "filename": "engagement-letter-harborline-markets.pdf",
        "page_numbers": [2],
        "page_segments": [
            {
                "page_number": 2,
                "text": "The fixed professional fee for this engagement is $18,500.",
            }
        ],
        "text": "The fixed professional fee for this engagement is $18,500.",
    }


def sample_evidence():
    chunk = sample_chunk()
    return {
        "E1": {
            "filename": chunk["filename"],
            "chunk_id": chunk["chunk_id"],
            "page_number": 2,
            "quote": chunk["text"],
        }
    }


class FakeReranker:
    def predict(self, pairs, batch_size=8):
        return [float(len(text)) for _question, text in pairs]


class ChunkAndFolderTests(unittest.TestCase):
    def test_chunks_keep_filename_and_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "policy.pdf"
            write_text_pdf(
                pdf_path,
                [
                    ["Alpha lives on the first page."],
                    ["Beta lives on the second page."],
                ],
            )
            from pypdf import PdfReader

            chunks = curator.build_page_aware_chunks(PdfReader(str(pdf_path)), pdf_path.name)
            self.assertTrue(chunks)
            self.assertTrue(all(chunk["filename"] == "policy.pdf" for chunk in chunks))
            by_page = {}
            for chunk in chunks:
                for segment in chunk["page_segments"]:
                    by_page.setdefault(segment["page_number"], []).append(segment["text"])
            self.assertIn("Alpha lives on the first page", " ".join(by_page[1]))
            self.assertIn("Beta lives on the second page", " ".join(by_page[2]))

    def test_folder_indexes_every_pdf_and_skips_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_text_pdf(folder / "a.pdf", [["Alpha fact is blue."]])
            write_text_pdf(folder / "b.PDF", [["Beta fact is green."]])
            (folder / "notes.txt").write_text("This note is not a source.")
            pdfs, chunks = curator.load_chunks(folder)
            self.assertEqual([path.name for path in pdfs], ["a.pdf", "b.PDF"])
            filenames = {chunk["filename"] for chunk in chunks}
            self.assertEqual(filenames, {"a.pdf", "b.PDF"})
            combined = " ".join(chunk["text"] for chunk in chunks)
            self.assertNotIn("not a source", combined)

    def test_empty_folder_explains_what_to_add(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                curator.load_chunks(tmp)
            self.assertIn("No PDF files", str(caught.exception))

    def test_embedding_cache_reuses_and_then_rebuilds(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cache_path = folder / "embeddings.json"
            write_text_pdf(folder / "a.pdf", [["Alpha fact is blue."]])
            calls = {"count": 0}

            def fake_embed(texts):
                calls["count"] += 1
                return [[1.0, 0.0] for _text in texts]

            with patch("curator.embed_texts", side_effect=fake_embed):
                first = curator.build_index(folder, cache_file=cache_path, use_reranker=False)
                second = curator.build_index(folder, cache_file=cache_path, use_reranker=False)
                self.assertEqual(calls["count"], 1)
                self.assertEqual(len(first.chunks), len(second.chunks))
                write_text_pdf(folder / "a.pdf", [["Alpha fact is now red."]])
                curator.build_index(folder, cache_file=cache_path, use_reranker=False)
                self.assertEqual(calls["count"], 2)


class CitationTests(unittest.TestCase):
    def test_real_quote_on_the_stated_page_passes(self):
        chunk = sample_chunk()
        citations = [sample_evidence()["E1"]]
        valid, problems = curator.validate_citations(
            "The fixed fee is $18,500.",
            citations,
            [chunk],
        )
        self.assertTrue(valid)
        self.assertEqual(problems, [])

    def test_wrong_file_or_page_or_quote_fails(self):
        chunk = sample_chunk()
        wrong_file = dict(sample_evidence()["E1"], filename="other.pdf")
        valid, problems = curator.validate_citations(
            "The fixed fee is $18,500.",
            [wrong_file],
            [chunk],
        )
        self.assertFalse(valid)
        self.assertTrue(problems)

        wrong_page = dict(sample_evidence()["E1"], page_number=9)
        valid, problems = curator.validate_citations(
            "The fixed fee is $18,500.",
            [wrong_page],
            [chunk],
        )
        self.assertFalse(valid)

        wrong_quote = dict(sample_evidence()["E1"], quote="The fee is one dollar.")
        valid, problems = curator.validate_citations(
            "The fee is one dollar.",
            [wrong_quote],
            [chunk],
        )
        self.assertFalse(valid)
        self.assertIn("Quote was not found", problems[0])

    def test_amount_must_appear_in_the_cited_passage(self):
        chunk = sample_chunk()
        citation = dict(sample_evidence()["E1"])
        citation["quote"] = "Fees for the Harborline Markets engagement are stated on this page."
        citation["text"] = citation["quote"]
        chunk["page_segments"] = [{"page_number": 2, "text": citation["quote"]}]
        chunk["text"] = citation["quote"]
        valid, problems = curator.validate_citations(
            "The fixed fee is $18,500.",
            [citation],
            [chunk],
        )
        self.assertFalse(valid)
        self.assertTrue(any("$18,500" in problem for problem in problems))

    def test_abstention_requires_no_citations(self):
        valid, problems = curator.validate_citations("I don't know.", [], [])
        self.assertTrue(valid)
        self.assertEqual(problems, [])
        valid, _problems = curator.validate_citations(
            "I don't know.",
            [sample_evidence()["E1"]],
            [sample_chunk()],
        )
        self.assertFalse(valid)


class MockedQwenTests(unittest.TestCase):
    def test_generation_model_stays_qwen(self):
        self.assertEqual(curator.GENERATION_MODEL, "qwen2.5:3b")

    def _chat_response(self, payload):
        return {
            "message": {"content": json.dumps(payload)},
            "total_duration": 10,
        }

    def test_valid_qwen_reply_keeps_filename_and_page(self):
        calls = []

        def fake_post(endpoint, payload, timeout):
            calls.append((endpoint, payload, timeout))
            return self._chat_response(
                {"evidence_ids": ["E1"], "answer": "The fixed fee is $18,500."}
            )

        with patch("curator.ollama_post", side_effect=fake_post):
            result = curator.generate_answer(
                "What is the fixed fee?",
                sample_evidence(),
                [sample_chunk()],
            )

        self.assertEqual(len(calls), 1)
        endpoint, payload, _timeout = calls[0]
        self.assertEqual(endpoint, "chat")
        self.assertEqual(payload["model"], "qwen2.5:3b")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["num_predict"], 256)
        self.assertEqual(payload["messages"][0]["content"], "Use only the supplied evidence.")
        self.assertIn('answer "I don\'t know."', payload["messages"][1]["content"])
        self.assertTrue(result["citations_valid"])
        self.assertEqual(
            result["citations"][0]["filename"],
            "engagement-letter-harborline-markets.pdf",
        )
        self.assertEqual(result["citations"][0]["page_number"], 2)
        self.assertEqual(result["delivered_answer"], "The fixed fee is $18,500.")

    def test_bad_evidence_id_is_withheld(self):
        def fake_post(endpoint, payload, timeout):
            return self._chat_response(
                {"evidence_ids": ["E99"], "answer": "The fixed fee is $18,500."}
            )

        with patch("curator.ollama_post", side_effect=fake_post):
            result = curator.generate_answer(
                "What is the fixed fee?",
                sample_evidence(),
                [sample_chunk()],
            )
        self.assertFalse(result["citations_valid"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(
            result["delivered_answer"],
            "I don't know. (Citation validation failed.)",
        )

    def test_abstention_with_extra_ids_is_still_clean(self):
        def fake_post(endpoint, payload, timeout):
            return self._chat_response(
                {"evidence_ids": ["E1"], "answer": "I don't know."}
            )

        with patch("curator.ollama_post", side_effect=fake_post):
            result = curator.generate_answer(
                "What was revenue in 2025?",
                sample_evidence(),
                [sample_chunk()],
            )
        self.assertTrue(result["citations_valid"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["delivered_answer"], "I don't know.")

    def test_explicit_abstention_has_no_citation(self):
        def fake_post(endpoint, payload, timeout):
            return self._chat_response(
                {"evidence_ids": [], "answer": "I don't know."}
            )

        with patch("curator.ollama_post", side_effect=fake_post):
            result = curator.generate_answer(
                "What was revenue in 2025?",
                sample_evidence(),
                [sample_chunk()],
            )
        self.assertTrue(result["citations_valid"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["delivered_answer"], "I don't know.")

    def test_answer_question_calls_qwen_once(self):
        calls = []

        def fake_post(endpoint, payload, timeout):
            calls.append(endpoint)
            if endpoint == "embed":
                return {"embeddings": [[1.0, 0.0]]}
            return self._chat_response(
                {"evidence_ids": ["E1"], "answer": "The fixed fee is $18,500."}
            )

        chunk = sample_chunk()
        index = curator.DocumentIndex(
            folder="sample_documents",
            files=[chunk["filename"]],
            chunks=[chunk],
            embeddings=[[1.0, 0.0]],
            reranker=FakeReranker(),
        )
        with patch("curator.ollama_post", side_effect=fake_post):
            result = curator.answer_question("What is the fixed fee?", index)
        self.assertEqual(calls.count("chat"), 1)
        self.assertEqual(calls.count("embed"), 1)
        self.assertEqual(result["model"], "qwen2.5:3b")
        self.assertIn(chunk["filename"], result["citations"][0]["filename"])


class SampleDocumentTests(unittest.TestCase):
    def test_saved_quotes_match_the_named_file_and_page(self):
        cases_path = ROOT / "evaluation_cases.jsonl"
        cases = [
            json.loads(line)
            for line in cases_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(cases), 4)
        chunks = []
        for pdf_path in curator.list_pdfs(ROOT / "sample_documents"):
            from pypdf import PdfReader

            chunks.extend(
                curator.build_page_aware_chunks(PdfReader(str(pdf_path)), pdf_path.name)
            )
        self.assertGreaterEqual(len({chunk["filename"] for chunk in chunks}), 6)
        for case in cases:
            matched = [
                chunk
                for chunk in chunks
                if chunk["filename"] == case["filename"]
                and curator.chunk_contains_quote(
                    chunk, case["supporting_quote"], case["page_number"]
                )
            ]
            self.assertTrue(matched, case["question"])

    def test_revenue_is_not_in_the_sample_files(self):
        from pypdf import PdfReader

        text = []
        for pdf_path in curator.list_pdfs(ROOT / "sample_documents"):
            reader = PdfReader(str(pdf_path))
            text.extend((page.extract_text() or "") for page in reader.pages)
        combined = " ".join(text).casefold()
        self.assertNotIn("revenue", combined)
        self.assertIn("fictional", combined)


class CommandLineTests(unittest.TestCase):
    def test_cli_prints_one_cited_answer(self):
        chunk = sample_chunk()
        index = curator.DocumentIndex(
            folder="sample_documents",
            files=[chunk["filename"]],
            chunks=[chunk],
            embeddings=[[1.0, 0.0]],
            reranker=FakeReranker(),
        )
        result = {
            "delivered_answer": "The fixed fee is $18,500.",
            "citations_valid": True,
            "citations": [sample_evidence()["E1"]],
            "citation_problems": [],
            "model": "qwen2.5:3b",
            "generation_seconds": 1.25,
        }
        buffer = io.StringIO()
        with patch("curator.build_index", return_value=index), patch(
            "curator.answer_question", return_value=result
        ), redirect_stdout(buffer):
            code = curator.main(
                ["--docs", "sample_documents", "--question", "What is the fixed fee?"]
            )
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertEqual(output.count("\nAnswer:"), 1)
        self.assertIn(
            "engagement-letter-harborline-markets.pdf, page 2",
            output,
        )
        self.assertIn("qwen2.5:3b", output)

    def test_missing_folder_does_not_call_qwen(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = curator.main(
                ["--docs", "/tmp/curator-missing-folder", "--question", "Hello"]
            )
        self.assertEqual(code, 1)
        self.assertIn("Document folder not found", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
