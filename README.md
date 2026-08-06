# RAG From Scratch

An Advanced Retrieval-Augmented Generation (RAG) pipeline built without LangChain, using native Python for maximum flexibility.

## What it does

The pipeline answers questions about a company's internal documents by combining several RAG techniques:

1. **LLM-based smart chunking**  instead of splitting documents by fixed character count, an LLM reads each document and splits it into meaningful, overlapping chunks.
2. **Chunk pre-processing**  each chunk is stored with a headline, summary and original text for better retrieval.
3. **Vector search (Chroma)**  chunks are embedded and stored in a local Chroma vector database for similarity search.
4. **Query rewriting**  user questions are rewritten into short, targeted search queries before retrieval.
5. **Reranking**  retrieved chunks are re-ordered by an LLM for relevance before being used as context.
6. **Answer generation**  the LLM answers the user's question using the most relevant chunks, citing sources.


## Libraries used
 
- `openai`  embeddings and LLM calls
- `python-dotenv`  loading API keys from `.env`
- `pydantic`  data models for chunks and structured LLM outputs
- `chromadb`  local vector database for storing and searching chunks
- `tqdm`  progress bars during document processing
- `litellm`  unified interface for calling LLM completions


## Notes
- `knowledge-base` folder and all its sub-folders and `.md` files were made using an LLM.
- `preprocessed_db/` (the Chroma vector store) is generated automatically on first run and is not included in the repo.
- Requires a local sentence-transformer model (`all-MiniLM-L6-v2`), downloaded automatically on first run.
