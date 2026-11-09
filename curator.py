#Project Curator

from pypdf import PdfReader
from sentence_transformers import CrossEncoder
import requests
import math
import os #checks where the cache file exists
import json #saves and loads the cache
import hashlib #lets us detect whether the PDF has changed
import time #lets us measure how long the embedding and Q&A process takes

#Section 1: Read the PDF and build chunks.

# Extract each PDF page separately so every chunk keeps its page provenance.
PDF_PATH = "Bitcoin Thesis Paper copy.pdf"
reader = PdfReader(PDF_PATH)

with open(PDF_PATH, "rb") as pdf_file:
    pdf_hash = hashlib.sha256(pdf_file.read()).hexdigest()


def make_chunk_record(sentences, chunk_index):
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
    return {
        "chunk_id": f"chunk-{chunk_index:03d}-{content_hash}",
        "page_numbers": [segment["page_number"] for segment in page_segments],
        "page_segments": page_segments,
        "text": chunk_text,
    }

def build_page_aware_chunks(pdf_reader):
    # Keep sentence/page pairs while chunking, including the sentence overlap.
    page_sentences = []
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\n", " ")
        page_sentences.extend(
            (sentence.strip(), page_number)
            for sentence in page_text.split(". ") #splits at each period - might not be useful for decimals or tables containing periods
            if sentence.strip()
        )

    page_chunks = []
    current_sentences = []

    for sentence, page_number in page_sentences:
        current_text = " ".join(item[0] for item in current_sentences)
        if current_sentences and len(current_text) + 1 + len(sentence) > 1000:
            page_chunks.append(make_chunk_record(current_sentences, len(page_chunks)))

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
        page_chunks.append(make_chunk_record(current_sentences, len(page_chunks)))

    return page_chunks

#Section 2: Cache validation and embedding creation

chunks = build_page_aware_chunks(reader)
print(f"Number of chunks: {len(chunks)}")

cache_file = "embeddings.json"
cache_schema_version = 3
embedding_model = "nomic-embed-text"

# Old caches contain plain strings and cannot supply page citations, so require
# the new schema before reusing cached chunks and embeddings.
if os.path.exists(cache_file):
    with open(cache_file, "r") as cache_handle:
        cache = json.load(cache_handle)
else:
    cache = {}

cache_is_compatible = (
    cache.get("cache_schema_version") == cache_schema_version
    and cache.get("pdf_hash") == pdf_hash
    and cache.get("embedding_model") == embedding_model
    and len(cache.get("chunks", [])) == len(cache.get("embeddings", []))
    and all(
        isinstance(chunk, dict)
        and {"chunk_id", "page_numbers", "page_segments", "text"} <= chunk.keys()
        for chunk in cache.get("chunks", [])
    )
)

if cache_is_compatible:
    print("Loading cached page-aware chunks and embeddings.")
    chunks = cache["chunks"]
    chunk_embeddings = cache["embeddings"]
else:
    print("Creating embeddings for the PDF.")
    chunk_embeddings = []

    for chunk in chunks:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": embedding_model, "input": chunk["text"]},
        )
        response.raise_for_status()
        chunk_embeddings.append(response.json()["embeddings"][0])

    cache = {
        "cache_schema_version": cache_schema_version,
        "pdf_hash": pdf_hash,
        "embedding_model": embedding_model,
        "chunks": chunks,
        "embeddings": chunk_embeddings,
    }

    with open(cache_file, "w") as cache_handle:
        json.dump(cache, cache_handle)

    print("Page-aware embeddings saved to cache.")


#cosine similarity measures the space between two non-zero vectors.
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    return dot_product / (magnitude_a * magnitude_b)

#the above code packages the mathematical operator into a reusable function (def)
#so that every time we compared two vectors, we can simply say: cosine_similarity(vector_a, vector_b)

# Keep candidate count and final context size separate so we can compare
# cosine ordering and reranked ordering fairly.
CANDIDATE_COUNT = int(os.environ.get("RAG_CANDIDATE_COUNT", "10"))
FINAL_CONTEXT_COUNT = int(os.environ.get("RAG_FINAL_CONTEXT_COUNT", "6"))
RERANKER_MODEL = os.environ.get(
    "RAG_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
EVALUATION_CASES_FILE = "evaluation_cases.jsonl"


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def find_evaluation_case(question):
    if not os.path.exists(EVALUATION_CASES_FILE):
        return None
    with open(EVALUATION_CASES_FILE, "r") as cases_file:
        for line in cases_file:
            if not line.strip():
                continue
            case = json.loads(line)
            if normalize_text(case.get("question", "")) == normalize_text(question):
                return case
    return None


def chunk_contains_quote(chunk, quote, page_number=None):
    normalized_quote = normalize_text(quote)
    if not normalized_quote:
        return False
    for segment in chunk.get("page_segments", []):
        if page_number is not None and segment["page_number"] != page_number:
            continue
        if normalized_quote in normalize_text(segment["text"]):
            return True
    return False


def find_quote_rank(ranked_chunks, quote, reranked=False):
    for rank, result in enumerate(ranked_chunks, start=1):
        chunk = result[3] if reranked else result[2]
        if chunk_contains_quote(chunk, quote):
            return rank, chunk["chunk_id"]
    return None, None


def validate_citations(answer, citations, selected_chunks):
    if normalize_text(answer).startswith("i don't know"):
        return not citations, []
    if not isinstance(citations, list) or not citations:
        return False, ["Answer has no citation."]

    selected_by_id = {chunk["chunk_id"]: chunk for chunk in selected_chunks}
    problems = []
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        page_number = citation.get("page_number")
        quote = citation.get("quote", "")
        chunk = selected_by_id.get(chunk_id)

        if chunk is None:
            problems.append(f"Unknown or unselected chunk ID: {chunk_id}")
            continue
        if page_number not in chunk["page_numbers"]:
            problems.append(f"Page {page_number} is not part of {chunk_id}.")
            continue
        if not chunk_contains_quote(chunk, quote, page_number):
            problems.append(
                f"Quote was not found on page {page_number} in {chunk_id}."
            )

    return not problems, problems


def generate_answer(evidence_by_id, selected_chunks, label):
    context = "\n\n".join(
        f"[{evidence_id}] {source['quote']}"
        for evidence_id, source in evidence_by_id.items()
    )
 
    prompt = f"""Answer the question in one sentence using only the numbered excerpts.
Choose the evidence before writing your answer. Return only the evidence IDs
needed to support it. If the evidence is partial, qualify the answer.
If the excerpts provide no useful evidence, answer "I don't know." with no IDs.
 
Question:
{question}
 
Evidence:
{context}

"""
    answered_format = {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "enum": list(evidence_by_id)},
            },
            "answer": {"type": "string", "minLength": 1},
        },
        "required": ["evidence_ids", "answer"],
    }

    abstention_format = {
        "type": "object",
        "properties": {
            "evidence_ids": {"type": "array", "maxItems": 0},
            "answer": {"const": "I don't know."},
        },
        "required": ["evidence_ids", "answer"],
    }

    response_format = (
        {"anyOf": [answered_format, abstention_format]}
        if evidence_by_id
        else abstention_format
    )

    started_at = time.perf_counter()

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:3b",
            "format": response_format,
            "messages": [
                {
                    "role": "system",
                    "content": "Use only the supplied evidence.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
    )

    response.raise_for_status()
    data = response.json()
    generation_seconds = time.perf_counter() - started_at
 
    try:
        parsed = json.loads(data["message"]["content"])

    except json.JSONDecodeError:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    answer = parsed.get("answer", "")
    evidence_ids = parsed.get("evidence_ids")
    citations = []

    if not isinstance(answer, str) or not answer.strip():
        citations_valid = False
        citation_problems = ["Qwen did not return an answer."]
    elif not isinstance(evidence_ids, list) or any(
        not isinstance(evidence_id, str) or evidence_id not in evidence_by_id
        for evidence_id in evidence_ids
    ):
        citations_valid = False
        citation_problems = ["Qwen returned invalid evidence IDs."]
    else:
        answer = answer.strip()
        citations = [
            evidence_by_id[evidence_id]
            for evidence_id in dict.fromkeys(evidence_ids)
        ]
        citations_valid, citation_problems = validate_citations(
            answer, citations, selected_chunks
        )
 
    delivered_answer = (
        answer if citations_valid else "I don't know. (Citation validation failed.)"
    )
 
    print(f"\n--- {label} ---")
    print(f"Answer: {delivered_answer}")

    for citation in citations:
        print(
            f"Citation: p. {citation['page_number']} | {citation['chunk_id']} | "
            f"“{citation['quote']}”"
        )

    print(f"Citation validation: {'PASS' if citations_valid else 'FAIL'}")

    for problem in citation_problems:
        print(f"Citation issue: {problem}")

    print(f"Generation time: {generation_seconds:.2f}s")
 
    return {
        "model_answer": answer,
        "delivered_answer": delivered_answer,
        "citations": citations,
        "citations_valid": citations_valid,
        "citation_problems": citation_problems,
        "ollama": data,
        "generation_seconds": generation_seconds,
    }

question = input("\nAsk a question about the Bitcoin paper: ")

#Section #: Embed the question and retrieve the cadidates.
question_response = requests.post(
    "http://localhost:11434/api/embed",
    json={"model": embedding_model, "input": question},
)
question_response.raise_for_status()
question_embedding = question_response.json()["embeddings"][0]

# Stage 1: cosine similarity quickly selects a small candidate set.
retrieval_start = time.perf_counter()
results = []
for i, chunk in enumerate(chunks):
    score = cosine_similarity(question_embedding, chunk_embeddings[i])
    results.append((score, i, chunk))

results.sort(key=lambda result: result[0], reverse=True)
retrieval_seconds = time.perf_counter() - retrieval_start
candidates = results[:CANDIDATE_COUNT]

# Stage 2: compare each question/chunk pair and reorder the cosine candidates.
reranker_load_start = time.perf_counter()
try:
    reranker = CrossEncoder(
        RERANKER_MODEL,
        backend="onnx",
        device="cpu",
        local_files_only=False,
    )
except Exception as exc:
    raise RuntimeError(
        f"Reranker model '{RERANKER_MODEL}' is not available locally. "
        "Download it before running offline."
    ) from exc
reranker_load_seconds = time.perf_counter() - reranker_load_start

pairs = [(question, chunk["text"]) for _cosine, _i, chunk in candidates]
rerank_start = time.perf_counter()
reranker_scores = reranker.predict(pairs, batch_size=8)
rerank_seconds = time.perf_counter() - rerank_start

reranked_results = sorted(
    [
        (float(reranker_score), cosine_score, chunk_index, chunk)
        for reranker_score, (cosine_score, chunk_index, chunk)
        in zip(reranker_scores, candidates)
    ],
    key=lambda result: result[0],
    reverse=True,
)


def show_ranking(title, ranked_chunks, reranked=False):
    print(f"\n--- {title} ---")
    for rank, result in enumerate(ranked_chunks, start=1):
        if reranked:
            reranker_score, cosine_score, _index, chunk = result
            score_text = (
                f"Reranker: {reranker_score:.4f} | Cosine: {cosine_score:.4f}"
            )
        else:
            cosine_score, _index, chunk = result
            score_text = f"Cosine: {cosine_score:.4f}"
        pages = ", ".join(map(str, chunk["page_numbers"]))
        print(f"{rank}. {chunk['chunk_id']} | Pages: {pages} | {score_text}")


def select_context(ranked_chunks, reranked=False):
    selected_chunks = [
        result[3] if reranked else result[2]
        for result in ranked_chunks[:FINAL_CONTEXT_COUNT]
    ]
 
    evidence_by_id = {}
    for chunk in selected_chunks:
        for segment in chunk["page_segments"]:
            if segment["text"].strip():
                evidence_id = f"E{len(evidence_by_id) + 1}"
                evidence_by_id[evidence_id] = {
                    "chunk_id": chunk["chunk_id"],
                    "page_number": segment["page_number"],
                    "quote": segment["text"],
                }
 
    return selected_chunks, evidence_by_id


show_ranking("Cosine candidates", candidates)
show_ranking("Reranked candidates", reranked_results, reranked=True)

evaluation_case = find_evaluation_case(question)
if evaluation_case:
    gold_quote = evaluation_case["supporting_quote"]
    cosine_rank, cosine_chunk_id = find_quote_rank(candidates, gold_quote)
    reranked_rank, reranked_chunk_id = find_quote_rank(
        reranked_results, gold_quote, reranked=True
    )
    print("\n--- Gold passage retrieval check ---")
    print(
        f"Cosine top 10: rank {cosine_rank} ({cosine_chunk_id})"
        if cosine_rank is not None
        else "Cosine top 10: supporting quote not found"
    )
    print(
        f"Reranked top 10: rank {reranked_rank} ({reranked_chunk_id})"
        if reranked_rank is not None
        else "Reranked top 10: supporting quote not found"
    )
    print(f"Gold answer: {evaluation_case['gold_answer']}")

baseline_chunks, baseline_context = select_context(candidates)
reranked_chunks, reranked_context = select_context(
    reranked_results, reranked=True
)

print(f"\n--- Exact context sent to Qwen ---\n{reranked_context}")

baseline_result = generate_answer(
    baseline_context, baseline_chunks, "Cosine-only answer"
)
reranked_result = generate_answer(
    reranked_context, reranked_chunks, "Cross-encoder reranked answer"
)

print("\n--- Stage timings ---")
print(f"Cosine retrieval: {retrieval_seconds:.4f}s")
print(f"Reranker model load: {reranker_load_seconds:.2f}s")
print(f"Cross-encoder scoring: {rerank_seconds:.4f}s")
for label, result in (
    ("Cosine-only", baseline_result),
    ("Reranked", reranked_result),
):
    ollama = result["ollama"]
    print(
        f"{label} generation: {result['generation_seconds']:.2f}s "
        f"(Ollama {ollama.get('total_duration', 0) / 1e9:.2f}s)"
    )
