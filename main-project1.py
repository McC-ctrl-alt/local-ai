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
    page_numbers = sorted({page_number for _sentence, page_number in sentences})
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:10]
    return {
        "chunk_id": f"chunk-{chunk_index:03d}-{content_hash}",
        "page_numbers": page_numbers,
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
cache_schema_version = 2
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
        and {"chunk_id", "page_numbers", "text"} <= chunk.keys()
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

# Stage 2: the cross-encoder reads each question/chunk pair jointly and
# produces a relevance score used to reorder only the cosine candidates.
try:
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        local_files_only=True,
    )
except Exception as exc:
    raise RuntimeError(
        f"Reranker model '{RERANKER_MODEL}' is not available locally. "
        "Download it before running offline; see the setup notes."
    ) from exc

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


def show_ranking(title, ranked_chunks, use_reranker_score=False):
    print(f"\n--- {title} ---")
    for rank, result in enumerate(ranked_chunks, start=1):
        if use_reranker_score:
            reranker_score, cosine_score, _index, chunk = result
            score_text = (
                f"Reranker: {reranker_score:.4f} | Cosine: {cosine_score:.4f}"
            )
        else:
            cosine_score, _index, chunk = result
            score_text = f"Cosine: {cosine_score:.4f}"
        pages = ", ".join(map(str, chunk["page_numbers"]))
        print(
            f"{rank}. {chunk['chunk_id']} | Pages: {pages} | {score_text}"
        )


def context_from(ranked_chunks, reranked=False):
    selected = ranked_chunks[:FINAL_CONTEXT_COUNT]
    if reranked:
        selected_chunks = [result[3] for result in selected]
    else:
        selected_chunks = [result[2] for result in selected]
    context = "\n\n".join(
        f"[{chunk['chunk_id']} | PDF pages {', '.join(map(str, chunk['page_numbers']))}]\n"
        f"{chunk['text']}"
        for chunk in selected_chunks
    )
    return selected_chunks, context


def generate_answer(context, label):
    prompt = f"""Answer in one sentence using only the supplied evidence.
If the evidence is insufficient, say "I don't know."
Cite the supporting chunk ID and PDF page number.

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
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Use only the supplied context. Do not invent information. "
                        "Include a citation to the exact supporting chunk and page."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
    )
    response.raise_for_status()
    data = response.json()
    answer = data["message"]["content"]
    elapsed_seconds = time.perf_counter() - started_at

    print(f"\n--- {label} answer ---")
    print(answer)
    print(
        f"Time: {elapsed_seconds:.2f}s "
        f"(Ollama reported {data.get('total_duration', 0) / 1e9:.2f}s)"
    )
    return answer, data, elapsed_seconds


show_ranking("Cosine candidates", candidates)
show_ranking("Reranked candidates", reranked_results, use_reranker_score=True)

baseline_chunks, baseline_context = context_from(candidates)
reranked_chunks, reranked_context = context_from(
    reranked_results, reranked=True
)

baseline_answer, baseline_data, baseline_generation_seconds = generate_answer(
    baseline_context, "Cosine-only"
)
reranked_answer, reranked_data, reranked_generation_seconds = generate_answer(
    reranked_context, "Cross-encoder reranked"
)

print("\n--- Stage timings ---")
print(f"Cosine retrieval: {retrieval_seconds:.4f}s")
print(f"Cross-encoder reranking: {rerank_seconds:.4f}s")
print(
    "Baseline generation: "
    f"{baseline_generation_seconds:.2f}s "
    f"(prompt {baseline_data.get('prompt_eval_duration', 0) / 1e9:.2f}s; "
    f"generation {baseline_data.get('eval_duration', 0) / 1e9:.2f}s)"
)
print(
    "Reranked generation: "
    f"{reranked_generation_seconds:.2f}s "
    f"(prompt {reranked_data.get('prompt_eval_duration', 0) / 1e9:.2f}s; "
    f"generation {reranked_data.get('eval_duration', 0) / 1e9:.2f}s)"
)
