"""
Advanced RAG Pipeline (built from scratch)
Techniques used:
1. LLM-based smart chunking (instead of fixed-size splitting)
2. Chunk pre-processing (headline + summary + original text)
3. Reranking of retrieved chunks
4. Query rewriting before retrieval

Requires a .env file with your OPENAI_API_KEY, and a knowledge-base/
folder containing subfolders (e.g. products/, employees/, contracts/,
company/) of .md files.
"""

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm import tqdm
from litellm import completion

# Config
openai = OpenAI()
load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

DB_NAME = "preprocessed_db"
COLLECTION_NAME = "docs"

# Local sentence transformer embedding model
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

KNOWLEDGE_BASE_PATH = Path("knowledge-base")
AVERAGE_CHUNK_SIZE = 500
RETRIEVAL_K = 10

# Data models
class Result(BaseModel):
    """A single retrieved/stored piece of content, similar to LangChain's Document."""
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    """One LLM generated chunk of a document."""
    headline: str = Field(description="A brief heading for this chunk, likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk, unchanged")

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

class RankOrder(BaseModel):
    """Relevance ordering returned by the reranker LLM call."""
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# Step 1: Load documents from the knowledge base
def fetch_documents() -> list[dict]:
    """Homemade version of LangChain's DirectoryLoader, reads every .md file,
    grouped by its parent folder name (used as the doc type)."""
    documents = []
    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                documents.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})
    print(f"Loaded {len(documents)} documents")
    return documents

# Step 2: Turn documents into LLM-generated chunks
def make_prompt(document: dict) -> str:
    """Builds the chunking instructions for a single document."""
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""

def make_messages(document: dict) -> list[dict]:
    return [{"role": "user", "content": make_prompt(document)}]

def process_document(document: dict) -> list[Result]:
    """Sends one document to the LLM and converts the response into Result objects."""
    messages = make_messages(document)
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents: list[dict]) -> list[Result]:
    """Processes every document into chunks. (Can be parallelized with
    multiprocessing.Pool if you hit rate limits, run sequentially here.)"""
    chunks = []
    for doc in tqdm(documents):
        chunks.extend(process_document(doc))
    return chunks

# Step 3: Embed and store chunks in Chroma
def create_embeddings(chunks: list[Result]) -> None:
    chroma = PersistentClient(path=DB_NAME)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)

    texts = [chunk.page_content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

# Retrieval: reranking + query rewriting
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Asks the LLM to reorder retrieved chunks by relevance to the question."""
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]

def fetch_context_unranked(question: str, collection) -> list[Result]:
    """Embeds the question and retrieves the top-K nearest chunks from Chroma."""
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=doc, metadata=meta))
    return chunks

def fetch_context(question: str, collection) -> list[Result]:
    """Retrieve then rerank."""
    chunks = fetch_context_unranked(question, collection)
    return rerank(question, chunks)

# Answering: build the prompt, call the LLM
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

def make_rag_messages(question: str, history: list[dict], chunks: list[Result]) -> list[dict]:
    """Builds the final message list for the answering LLM call, citing sources."""
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]

def rewrite_query(question: str, history: list[dict] = []) -> str:
    """Rewrites the user's question into a short, targeted search query."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a single, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
Don't mention the company name unless it's a general question about the company.
IMPORTANT: Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content

def answer_question(question: str, collection, history: list[dict] = None) -> tuple[str, list[Result]]:
    """Full RAG pipeline for one question: rewrite -> retrieve -> rerank -> answer."""
    history = history or []
    query = rewrite_query(question, history)
    chunks = fetch_context(query, collection)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks

# Running the pipeline
def build_knowledge_base() -> None:
    """One-time ingestion: load docs, chunk them, embed and store them."""
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)

if __name__ == "__main__":
    build_knowledge_base()

    # ask a question through the full RAG pipeline
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    answer, sources = answer_question("Who won the IIOTY award?", collection)
    print(answer)