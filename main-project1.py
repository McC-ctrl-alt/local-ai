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
import requests
import math
import os #checks where the cache file exists
import json #saves and loads the cache
import hashlib #lets us detect whether the PDF has changed
import time #lets us measure how long the embedding and Q&A process takes


#extract text from a PDF file:
#make sure the pdf is in the same folder as the script or provide the full path to the PDF file.
#PDFReader reads the PDF and gives the program access to its pages
reader = PdfReader("Bitcoin Thesis Paper copy.pdf") 

full_text = ""

for page in reader.pages:
    full_text += page.extract_text() + "\n" 

#print(repr(full_text[:1000]))
# ^print the first 1000 characters of the extracted text for verification

#extract_text() converts the page's contents into string
#.pages: returns a list of Page objects, one for each page in the PDF.
#"\n" adds a line break after each page's text to separate them.

#clean up the extracted text:
clean_text = full_text.replace("\n", " ")

#split the document into sentences
sentences = clean_text.split(". ")

#create chunks with overlap
chunks = []
current_chunk = []

#the code below does 3 things:
#creates chunks of text that are less than 1000 characters long
#adds overlap of 300 characters from the previous chunk to the next chunk
#makes sure the chunks start with complete sentences and not cut off in the middle of a sentence
#the specific line that makes sure the chunk starts with complete sentences is: 
# "overlap.insert(0, previous_sentence)"

for sentence in sentences:
    sentence = sentence.strip()  # remove leading/trailing whitespace
    if len(" ".join(current_chunk)) + len(sentence) < 1000:  # adjust the chunk size as needed
        current_chunk.append(sentence)
    else:
        #save the completed chunk and start a new one
        chunks.append(" ".join(current_chunk))

        #build overlap from complete sentences
        overlap = []
        overlap_length = 0

        for previous_sentence in reversed(current_chunk):
            if overlap_length + len(previous_sentence) > 300:
                break

            overlap.insert(0, previous_sentence)  # add to the beginning of the overlap list
            overlap_length += len(previous_sentence)

        # start the new chunk with the overlap and the current sentence
        current_chunk = overlap + [sentence]  

#add the final chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))

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

# Identify the current PDF contents
with open("Bitcoin Thesis Paper copy.pdf", "rb") as f: 
    pdf_hash = hashlib.sha256(f.read()).hexdigest()

# Load the cache if it matches the current PDF
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# the code below checks if the cache exists and if the hash of the current PDF matches the hash stored in the cache. 
# If they match, it loads the cached chunks and embeddings. 
# If not, it creates new embeddings for the PDF and saves them to the cache.
if cache.get("pdf_hash") == pdf_hash:
    print("Loading cached chunks and embeddings.")
    chunks = cache["chunks"]
    chunk_embeddings = cache["embeddings"]

else:
    print("Creating embeddings for the PDF.")
    chunk_embeddings = []

    for chunk in chunks:
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": chunk
            }
        )
        response.raise_for_status()
        chunk_embedding = response.json()["embeddings"][0]
        chunk_embeddings.append(chunk_embedding)

    cache = {
        "pdf_hash": pdf_hash,
        "chunks": chunks,
        "embeddings": chunk_embeddings
    }

    with open(cache_file, "w") as f:
        json.dump(cache, f)

    print("Embeddings saved to cache.")

def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    return dot_product / (magnitude_a * magnitude_b)

#the above code packages the mathematical operator into a reusable function (def)
#so that every time we compared two vectors, we can simply say: cosine_similarity(vector_a, vector_b)

question = input("\nAsk a question about the Bitcoin paper: ")

question_response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": question
    }
)

question_embedding = question_response.json()["embeddings"][0]

# Compare the question against the stored chunk embeddings
results = [] #this creates an empty list where we will store (score, chunk number, chunk text)

for i, chunk in enumerate(chunks): #this doesn't call Ollama's embedding API anymore - it will just reuse the vectors already created.
    score = cosine_similarity(question_embedding, chunk_embeddings[i])

    results.append((score, i, chunk)) #then store the result and repeat

results.sort(reverse=True) #this puts the highest similarity scores first
top_results = results[:6]

for score, i, chunk in top_results: 
    print(f"\n--- CHUNK {i} | Similarity: {score:.4f} ---")
    print(chunk)

#the cosine calculation gives us a number between -1 and 1 that tells us how similar the two vectors are.

#cosine similarity asks: are these two vectors pointing in the same direction?

#a score close to 1 means high similarity; close to 0 means low similiarity

#RAG generation:
#Retrieval found the most relevant chunks
#We now give those chunks + the question to the LLM
#The LLM does not search the document; it uses the retrieval context to answer

#top_results = results[:5]

context = "\n\n".join(
    chunk for score, i, chunk in top_results
)

prompt = f"""Answer in one sentence using only the supplied evidence.

Context:
{context}

Question:
{question}
"""

start_time = time.perf_counter()

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:3b",
        "messages": [
            {
                "role": "system",
                "content": "Answer using only the provided context. Do not invent information."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
)

data = response.json()

print(f"\n---Prompt---\n{prompt}")

print("\n--- PERFORMANCE METRICS ---")
print(f"Total Ollama time: {data.get('total_duration', 0) / 1e9:.2f}s")
print(f"Model loading: {data.get('load_duration', 0) / 1e9:.2f}s")
print(f"Prompt processing: {data.get('prompt_eval_duration', 0) / 1e9:.2f}s")
print(f"Answer generation: {data.get('eval_duration', 0) / 1e9:.2f}s")
print(f"Output tokens: {data.get('eval_count', 0)}")

generation_time = (time.perf_counter()) - (start_time) 

answer = response.json()["message"]["content"]

print("\n--- ANSWER ---")
print(answer)
print(f"\nGeneration time: {generation_time: .2f} seconds")
#what this adds: PDF -> chunks -> embeddings -> similarity search -> top 5 chunks -> Qwen3 -> answer
