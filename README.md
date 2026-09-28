RegForg : (RAG => OLLAMA + LangChain + Pinecone + FastApi).

An END-to-End RAG pipline for building a private, local knowledgebase system using PDF DOCUMENTS, OLLAMA-QWEN3-embedding-8B, Lnagchain, Pinecone.

////////----PROJECT OVERVIEW----///////

This project implements a local Retrieval-Augmented Generation (RAG) pipeline that allows an AI application to search and retrieve relevant information from private documents.

Instead of sending documents to an external embedding API, this project uses Ollama to run the embedding model locally.

The current pipeline uses:
Python — Core programming language
LangChain — Document processing and retrieval framework
PDFLoader — PDF document ingestion
RecursiveCharacterTextSplitter — Document chunking
Ollama — Local model execution
Qwen3 Embedding 8B — Local embedding model
Pinecone — Persistent vector database
Vector Similarity Search — Semantic retrieval

The embedding model generates 4096-dimensional vectors, which are stored in Pinecone DataBase and later used to retrieve documents relevant to a user's query.


////////----PROJECT STRUCTURE----///////

