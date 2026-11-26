"""Shared retrieval engine for curator.py and policy_rag.py.

Parse, chunk, embed, index, cosine top 10, cross-encoder rerank, and the
-0.7 abstain cutoff live here. PDFs in subfolders are included, and each
chunk is tagged with that subfolder as its department. curator.py imports
this module, so eval/run_eval.py and eval/compare.py stay on this path.
Tuning a model, the candidate count, or the cutoff happens in this file.
"""

import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".curator_index"
CACHE_SCHEMA = 1
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = os.environ.get("CURATOR_LLM", "qwen2.5:3b")
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
)
CANDIDATE_COUNT = int(os.environ.get("RAG_CANDIDATE_COUNT", "10"))
FINAL_CONTEXT_COUNT = int(os.environ.get("RAG_FINAL_CONTEXT_COUNT", "6"))
# On knowledge_test/, both no-answer questions scored below -1.7 and all eight
# answerable questions scored above 0.3. -0.7 sits in that gap. A top rerank
# score under it returns "I don't know." and does not call Qwen.
RERANK_ABSTAIN_BELOW = -0.7


def rerank_below_cutoff(top_score):
    """Return whether this top cross-encoder score should skip the answer call."""
    return top_score is not None and top_score < RERANK_ABSTAIN_BELOW


def chat_options(max_tokens):
    """Ollama options shared by every Qwen call.

    temperature 0 and seed 42 keep the call repeatable. num_ctx 8192 stops
    Ollama's default context from truncating the prompt.
    """
    return {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 8192,
        "num_predict": max_tokens,
    }


def make_chunk_record(sentences, chunk_index, source=None):
    chunk_text = " ".join(sentence for sentence, _page_number in sentences)
    page_sentences = {}
    for sentence, page_number in sentences:
        page_sentences.setdefault(page_number, []).append(sentence)
    page_segments = [
        {
            "page_number": page_number,
            "text": " ".join(page_sentences[page_number]),
        }
        for page_number in sorted(page_sentences)
    ]
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:10]
    record = {
        "chunk_id": f"chunk-{chunk_index:03d}-{content_hash}",
        "page_numbers": [segment["page_number"] for segment in page_segments],
        "page_segments": page_segments,
        "text": chunk_text,
    }
    # curator.py leaves these off so cached chunk records stay as they were.
    # policy_rag.py passes a source path and also keeps each sentence.
    if source is not None:
        record["source"] = source
        record["sentences"] = [
            {"page_number": page_number, "text": sentence}
            for sentence, page_number in sentences
        ]
    return record


def build_page_aware_chunks(pdf_reader, source=None, index_start=0):
    page_sentences = []
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\n", " ")
        page_sentences.extend(
            (sentence.strip(), page_number)
            for sentence in page_text.split(". ")
            if sentence.strip()
        )

    page_chunks = []
    current_sentences = []

    for sentence, page_number in page_sentences:
        current_text = " ".join(item[0] for item in current_sentences)
        if current_sentences and len(current_text) + 1 + len(sentence) > 1000:
            page_chunks.append(
                make_chunk_record(
                    current_sentences,
                    index_start + len(page_chunks),
                    source,
                )
            )

            # Carry whole sentences into the next chunk as overlap.
            overlap = []
            overlap_length = 0
            for previous_sentence in reversed(current_sentences):
                previous_length = len(previous_sentence[0])
                if overlap_length + previous_length > 300:
                    break
                overlap.insert(0, previous_sentence)
                overlap_length += previous_length
            current_sentences = overlap

        current_sentences.append((sentence, page_number))

    if current_sentences:
        page_chunks.append(
            make_chunk_record(
                current_sentences,
                index_start + len(page_chunks),
                source,
            )
        )

    return page_chunks


def ollama_request(endpoint, payload):
    try:
        response = requests.post(f"{OLLAMA_URL}/api/{endpoint}", json=payload, timeout=300)
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise RuntimeError("Start Ollama and run the command again.") from error
    return response.json()


def embedding_vector(payload: dict[str, Any], model: str) -> list[float]:
    """Return the first embedding from an Ollama /api/embed response."""
    try:
        vector = payload["embeddings"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"Ollama returned no embedding from {model}.") from error
    if isinstance(vector, (str, bytes)) or len(vector) == 0:
        raise ValueError(f"Ollama returned an empty embedding from {model}.")
    return vector


def embed(text):
    result = ollama_request("embed", {"model": EMBED_MODEL, "input": text})
    return embedding_vector(result, EMBED_MODEL)


def chat_answer(system_prompt, prompt, response_format=None, max_tokens=300):
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": chat_options(max_tokens),
    }
    if response_format is not None:
        payload["format"] = response_format
    result = ollama_request("chat", payload)
    return result["message"]["content"]


def file_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]


def read_index(path):
    try:
        with path.open(encoding="utf-8") as handle:
            index = json.load(handle)
    except (FileNotFoundError, ValueError):
        return {}
    if (
        isinstance(index, dict)
        and index.get("schema") == CACHE_SCHEMA
        and index.get("model") == EMBED_MODEL
        and isinstance(index.get("chunks"), list)
        and isinstance(index.get("embeddings"), list)
        and len(index["chunks"]) == len(index["embeddings"]) > 0
    ):
        return index
    return {}


def save_index(path, index):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(index, handle)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_or_build_index(pdf_path):
    """Reuse the saved passages and embeddings for this PDF whenever possible."""
    file_key = hashlib.sha256(str(pdf_path).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{file_key}.json"
    index = read_index(cache_path)
    signature = file_signature(pdf_path)

    def bind_source(index):
        resolved = str(pdf_path.resolve())
        for chunk in index["chunks"]:
            chunk["source_path"] = resolved
        return index

    if index.get("stat") == signature:
        print(f"Index: reused {len(index['chunks'])} passages from {pdf_path.name}.")
        return bind_source(index)

    contents = pdf_path.read_bytes()
    if file_signature(pdf_path) != signature:
        raise RuntimeError("The PDF changed while it was being read. Run the command again.")
    digest = hashlib.sha256(contents).hexdigest()

    if index.get("sha256") == digest:
        index["stat"] = signature
        save_index(cache_path, index)
        print(f"Index: reused {len(index['chunks'])} passages from {pdf_path.name}.")
        return bind_source(index)

    chunks = build_page_aware_chunks(PdfReader(BytesIO(contents)))
    if not chunks:
        raise ValueError("No text could be extracted. Choose a PDF with selectable text.")
    print(f"Index: building {len(chunks)} passages from {pdf_path.name}...")
    index = {
        "schema": CACHE_SCHEMA,
        "model": EMBED_MODEL,
        "stat": signature,
        "sha256": digest,
        "chunks": chunks,
        "embeddings": [embed(chunk["text"]) for chunk in chunks],
    }
    save_index(cache_path, index)
    print(f"Index: built {len(chunks)} passages and saved them in {CACHE_DIR.name}/.")
    return bind_source(index)


def list_pdfs(folder):
    """Every PDF in this folder and its subfolders, in path order.

    A department is a subfolder. Dropping a new PDF into one is enough for
    the next load to pick it up. Nothing in this function names a department.
    """
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def load_corpus(pdf_paths):
    """One index over every PDF. Retrieval and chunking stay the same per file.

    ``department`` is the PDF's parent folder name, so a file in
    ``knowledge_test/hr/`` is department ``hr``. ``load_or_build_index`` still
    reuses the saved embeddings until the file's stat signature changes.
    """
    chunks = []
    embeddings = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        index = load_or_build_index(pdf_path)
        for chunk in index["chunks"]:
            tagged = dict(chunk)
            tagged["source_file"] = pdf_path.name
            tagged["department"] = pdf_path.parent.name
            chunks.append(tagged)
        embeddings.extend(index["embeddings"])
    return {"chunks": chunks, "embeddings": embeddings}


def load_reranker():
    """Load the ONNX cross-encoder on CPU.

    The cached snapshot is resolved first, so a later run does not look the
    model up by name. With no local copy, that same ONNX load is tried online
    once. A failed ONNX load raises. PyTorch is not a fallback.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from sentence_transformers import CrossEncoder

    def load_onnx(model_path):
        return CrossEncoder(
            model_path,
            backend="onnx",
            device="cpu",
            model_kwargs={"file_name": "onnx/model.onnx"},
        )

    def fail(error):
        raise RuntimeError(
            f"Could not load the ONNX reranker '{RERANKER_MODEL}'. "
            "Run once with internet to download the model."
        ) from error

    try:
        local_folder = snapshot_download(RERANKER_MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        try:
            return load_onnx(RERANKER_MODEL)
        except Exception as error:
            fail(error)

    try:
        return load_onnx(local_folder)
    except Exception as error:
        fail(error)


def rank_candidates(question, index):
    """Cosine top 10. One normalised matrix-vector product scores every chunk."""
    question_vector = embed(question)
    chunks = index["chunks"]
    embeddings = index["embeddings"]
    count = min(len(chunks), len(embeddings))
    if count == 0:
        return []

    query = np.asarray(question_vector, dtype=np.float64)
    width = int(query.shape[0])
    rows = []
    for vector in embeddings[:count]:
        if len(vector) != width:
            raise ValueError(
                "cosine_similarity requires two vectors of the same length, "
                f"got {width} and {len(vector)}."
            )
        rows.append(vector)
    matrix = np.asarray(rows, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    for row_norm in row_norms:
        row_norm = float(row_norm)
        if query_norm == 0.0 or row_norm == 0.0:
            raise ValueError("cosine_similarity cannot score a zero-length vector.")
        if not np.isfinite(query_norm) or not np.isfinite(row_norm):
            raise ValueError("cosine_similarity cannot score a non-finite vector.")

    scores = (matrix / row_norms[:, np.newaxis]) @ (query / query_norm)
    if not np.all(np.isfinite(scores)):
        raise ValueError("cosine_similarity cannot score a non-finite vector.")

    # Same order as a stable sort by score, highest first. Ties keep the
    # earlier chunk.
    order = np.lexsort((np.arange(count), -scores))[:CANDIDATE_COUNT]
    return [
        (float(scores[chunk_index]), int(chunk_index), chunks[chunk_index])
        for chunk_index in order
    ]


def rerank_candidates(question, candidates, reranker):
    pairs = [(question, chunk["text"]) for _score, _index, chunk in candidates]
    scores = [float(score) for score in reranker.predict(pairs, batch_size=8)]
    if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
        raise ValueError("The reranker returned invalid scores.")
    results = [
        (score, cosine_score, chunk_index, chunk)
        for score, (cosine_score, chunk_index, chunk) in zip(scores, candidates)
    ]
    return sorted(results, key=lambda item: item[0], reverse=True)
