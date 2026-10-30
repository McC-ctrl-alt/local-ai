#Project 1: Personal Document Q&A

#Goal:
#Build a simple RAG (Retrieval-Augmented Generation) application that can answer questions
# using information contained in my own documents.

#Current stage:
#PDF -> Text

#Key concept:
#pypdf allows Python to read/extract text from PDF files.

#Chunking:
#When I ask the question on double spending, instead of looking for the answer on one page, the LLM will use
# chunking to break the text into smaller pieces and search for the meaning across all pages. 
# this will allow the LLM to combine information from multiple pages to answer the question.

from pypdf import PdfReader
from sentence_transformers import CrossEncoder
import requests
import math
import os #checks where the cache file exists
import json #saves and loads the cache
import hashlib #lets us detect whether the PDF has changed
import time #lets us measure how long the embedding and Q&A process takes


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
            for sentence in page_text.split(". ")
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


chunks = build_page_aware_chunks(reader)
print(f"Number of chunks: {len(chunks)}")

# region Embedding and retrieval notes
#this converts the chunks into data that we can use for embeddings. This now shows us how many chunks and how many characters per chunk.
#it still remains as strings of text but we can now use these chunks to create embeddings and store them in a vector database for retrieval.

#the problem with the first basic chunk method (where we had no overlap or designated chunk size is that it can lead to issues when trying to retrieve information from the chunks).
#1. Chunks are inconsistent in size, some are too long, some are too short.
#2. we're breaking context - we have seperated sentences that may be related to each other, but are now in different chunks.
#3. sentence boundaries aren't enough. Real documents aren't just collections of independent sentences. 
#meaning often depends on surrounding sentences, headings, paragraphs, tables, lists, and sections

#What the code does up to the chunking stage:
#1. It reads the PDF file and extracts the text from it.
#2. It cleans the text by removing line breaks and splitting it into chunks based on periods    
#3. It prints out each chunk with its corresponding number for verification.
#4. The chunks can then be used for further processing, such as indexing or searching for answers to questions.

#now on to embedding:
#embedding's role is to represent the meaning of text numerically so we can compare questions to their answers in the chunks
#we will use "nomic-embed-text" through Ollama. It's a dedicated embedding model and is better than a general chat model

#Process:
#Embedding: Represent this text numerically
#Similarity: Which numerical representations are closest to this question?
#LLM: Given the question + retrieved chunks, formulate an answer

#I am not asking the LLM to "search the PDF"
#I am setting up a retriveal system to find relevant chunks of text that can help answer the question. 
#The LLM will then use those chunks to generate an answer.

#The below is a demo of how we embed one chunk:

#text = chunk[0]  # just using the first chunk for demonstration

#response = requests.post(
#    "http://localhost:11434/api/embed",
#    json={
#        "model": "nomic-embed-text",
#        "input": text
#    }
#) 

#this chunk of text is sent to the embedding model to get its numerical representation
# Python -> HTTP request -> Ollama -> nomic-embed text -> vector -> Python
#JSON is a data format that allows us to send and receive data between the client (Python) and the server (Ollama).

#embedding = response.json()["embeddings"][0] #pulls the numerical vector out of Ollama's response and stores it in the variable "embedding"

#print(f"Embedding length: {len(embedding)}")
#print(embedding[:10]) # print the first 10 dimensions of the embedding for verification

#the important part of this is that I can see the relationship between vectors and the text they represent.
#This is the key to retrieval-augmented generation. 
#The LLM can use these embeddings to find relevant chunks of text that can help answer a question.

#the 768 would mean that this particular model represents the text using 768 numerical dimensions
#a numerical dimension is a single number in the vector that represents some aspect of the meaning of the text.

#Right now we only have 1 vector for 1 chunk of text (that is why it is 768 dimensions).

#the next concept is similarity search 

#let's bring in another vector for comparison. We will run a question and calculate its similarity
# endregion

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
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
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


def generate_answer(context, selected_chunks, label):
    prompt = f"""Answer in one sentence using only the supplied evidence.
If the evidence is insufficient, answer "I don't know." and return no citations.

Return one valid JSON object with:
- "answer": your one-sentence answer
- "citations": a list of objects with "chunk_id", integer "page_number",
  and "quote" copied exactly from that page's supplied text

Cite only chunks in the context. Do not invent page numbers or quotes.

Context:
{context}

Question:
{question}
"""
    started_at = time.perf_counter()
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:3b",
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Use only supplied evidence. Do not invent facts, pages, "
                        "chunk IDs, or quotations. Abstain if evidence is insufficient."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
    )
    response.raise_for_status()
    data = response.json()
    raw_answer = data["message"]["content"]
    generation_seconds = time.perf_counter() - started_at

    try:
        parsed = json.loads(raw_answer)
    except json.JSONDecodeError:
        parsed = {"answer": raw_answer, "citations": []}

    answer = str(parsed.get("answer", "")).strip()
    citations = parsed.get("citations", [])
    citations_valid, citation_problems = validate_citations(
        answer, citations, selected_chunks
    )
    delivered_answer = answer
    if not citations_valid:
        delivered_answer = "I don't know. (Citation validation failed.)"

    print(f"\n--- {label} ---")
    print(f"Answer: {delivered_answer}")
    if citations:
        for citation in citations:
            print(
                f"Citation: p. {citation.get('page_number')} | "
                f"{citation.get('chunk_id')} | "
                f"“{citation.get('quote', '')}”"
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
        device="cpu",
        local_files_only=True,
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
    selected = ranked_chunks[:FINAL_CONTEXT_COUNT]
    selected_chunks = [
        result[3] if reranked else result[2]
        for result in selected
    ]
    context = "\n\n".join(
        f"[chunk_id={chunk['chunk_id']}]\n"
        + "\n".join(
            f"[page={segment['page_number']}] {segment['text']}"
            for segment in chunk["page_segments"]
        )
        for chunk in selected_chunks
    )
    return selected_chunks, context


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
