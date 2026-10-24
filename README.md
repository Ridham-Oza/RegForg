### RegForg : (RAG => OLLAMA + LangChain + Pinecone + FastApi).

An END-to-End RAG pipline for building a private, local knowledgebase system using PDF DOCUMENTS, OLLAMA-QWEN3-embedding-8B, Lnagchain, Pinecone.

### ////////----PROJECT OVERVIEW----///////

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


### ////////----PROJECT STRUCTURE----///////

RegForg/
│
├── Data/
│   └── documents/
│       └── MachineLearning-3.pdf
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   └── test_questions.py
│
├── src/
│   ├── embeddings/
│   │   └── ollama_embeddings.py
│   ├── generation/
│   │   └── llm.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── ui/
│   │   └── gradio_app.py
│   └── utils/
│       ├── logger.py
│       └── metrics.py
│
├── config.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md


## Architecture

```text
PDF
 ↓
Document Loader
 ↓
Chunking + Cleaning
 ↓
Qwen3-Embedding 8B
 ↓
4096-dimensional vectors
 ↓
Pinecone
 ↓
Semantic Retrieval
 ↓
Top-K Context
 ↓
Qwen3 8B
 ↓
Grounded Answer
 ↓
Gradio UI
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Embeddings | Qwen3-Embedding 8B |
| Generation | Qwen3 8B |
| Local inference | Ollama |
| Vector database | Pinecone |
| Retrieval | Cosine similarity |
| Frontend | Gradio |
| Evaluation | Recall@K, Hit@K, MRR, similarity, latency |

---

# Step 1 — Prerequisites

Install:

- Python 3.12
- Ollama
- Git
- Pinecone account/API key

Check Python:

```bash
python3 --version
```

Check Ollama:

```bash
ollama --version
```

---

# Step 2 — Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RegForg
```

If you already have the project:

```bash
cd RegForg
```

---

# Step 3 — Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

# Step 4 — Install Dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Otherwise:

```bash
pip install langchain langchain-community langchain-core
pip install langchain-ollama langchain-text-splitters
pip install pinecone ollama gradio pypdf python-dotenv
```

---

# Step 5 — Start Ollama

Check the Ollama API:

```bash
curl http://localhost:11434/api/tags
```

Pull the required models:

```bash
ollama pull qwen3-embedding:8b
ollama pull qwen3:8b
```

Verify:

```bash
ollama list
```

You should see:

```text
qwen3:8b
qwen3-embedding:8b
```

---

# Step 6 — Configure `.env`

Create:

```bash
touch .env
```

Example:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=regforg
PINECONE_NAMESPACE=regforg
PINECONE_DIMENSION=4096
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

If your `config.py` defines model variables, configure them there or in `.env` according to your implementation:

```env
EMBEDDING_MODEL=qwen3-embedding:8b
LLM_MODEL=qwen3:8b
TOP_K=5
```

Never commit `.env`.

---

# Step 7 — Add the PDF

Put your document here:

```text
Data/documents/
```

Example:

```text
Data/documents/MachineLearning-3.pdf
```

---

# Step 8 — Test Embeddings

```bash
python -c 'from src.embeddings.ollama_embeddings import QwenEmbeddings; e=QwenEmbeddings(); v=e.embed_query("What is machine learning?"); print("Embedding successful:", len(v)==4096); print("Dimension:", len(v)); print("First 10:", v[:10])'
```

Expected:

```text
Embedding successful: True
Dimension: 4096
```

---

# Step 9 — Test Chunking

```bash
python -m src.ingestion.chunker
```

The current project produced:

```text
Pages loaded: 152
Chunks created: 374

```


The chunking stage also removes unwanted short/header-only PDF chunks.

---

# Step 10 — Index the PDF

If you are re-indexing after changing chunking, first remove the old namespace vectors:

```bash
python -c 'from pinecone import Pinecone; from config import PINECONE_API_KEY,PINECONE_INDEX_NAME,PINECONE_NAMESPACE; pc=Pinecone(api_key=PINECONE_API_KEY); index=pc.Index(PINECONE_INDEX_NAME); print("Before:",index.describe_index_stats()); index.delete(delete_all=True,namespace=PINECONE_NAMESPACE); print("Old vectors deleted."); print("After:",index.describe_index_stats())'
```

Then:

```bash
python -m src.ingestion.indexer
```

Completed project result:

```text
Chunks indexed: 374
```

---

# Step 11 — Verify Pinecone

```bash
python -c 'from pinecone import Pinecone; from config import PINECONE_API_KEY,PINECONE_INDEX_NAME; pc=Pinecone(api_key=PINECONE_API_KEY); index=pc.Index(PINECONE_INDEX_NAME); print(index.describe_index_stats())'
```

The namespace should contain the indexed vectors.

---

# Step 12 — Test Retrieval

```bash
python -m src.retrieval.retriever
```

Or:

```bash
python -c 'from src.retrieval.retriever import PineconeRetriever; r=PineconeRetriever(top_k=5); result=r.retrieve("What is machine learning?"); print("Retrieved:",len(result["matches"])); print("Retrieval time:",round(result["retrieval_time"],4),"sec"); [(print(i+1, "Score:",round(m["score"],6),"Page:",m["page"])) for i,m in enumerate(result["matches"])]'
```

The retriever returns:

```text
matches
embedding_time
pinecone_time
retrieval_time
top_k
```

---

# Step 13 — Test LLM Generation

```bash
python -m src.generation.llm
```

This performs:

```text
Question
 ↓
Qwen Embedding
 ↓
Pinecone Retrieval
 ↓
Context
 ↓
Qwen3 8B
 ↓
Answer
```

---

# Step 14 — Run the Complete RAG Pipeline

```bash
python -m src.pipeline.rag_pipeline
```

The pipeline returns:

```text
answer
documents
retrieval_time
generation_time
total_time
num_chunks
```

---

# Step 15 — Run Retrieval Evaluation

```bash
python -m evaluation.evaluate_retrieval
```

The project evaluates:

1. Recall@5
2. Hit@5
3. MRR
4. Average similarity
5. Average retrieval latency
6. Average end-to-end retrieval latency

## Recorded Evaluation

```text
Recall@5              : 1.0000
Hit@5                 : 1.0000
MRR                   : 1.0000
Average similarity    : 0.7201
Average retrieval     : 1.7314 sec
Average end-to-end    : 1.7317 sec
```

These results depend on the evaluation dataset, document, models, and runtime environment.

---

# Step 16 — Start the Gradio Frontend

```bash
python -m src.ui.gradio_app
```

Open:

```text
http://127.0.0.1:7860
```

If port 7860 is busy:

```bash
GRADIO_SERVER_PORT=7861 python -m src.ui.gradio_app
```

Then open:

```text
http://127.0.0.1:7861
```

---
---

# Troubleshooting

## 1. Ollama connection error

```text
Failed to connect to Ollama
```

Check:

```bash
ollama list
```

and:

```bash
curl http://localhost:11434/api/tags
```

---

## 2. Model not found

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:8b
```

---

## 3. Embedding dimension mismatch

The current embedding model returns:

```text
4096
```

Therefore Pinecone must be configured as:

```env
PINECONE_DIMENSION=4096
```

---


## 4. Gradio port error

If 7860 is already occupied:

```bash
GRADIO_SERVER_PORT=7861 python -m src.ui.gradio_app
```

---

# RAG Evaluation Metrics

## Recall@K

Measures the proportion of relevant results retrieved in the top K.

```text
Recall@K =
relevant retrieved /
total relevant
```

## Hit@K

Returns 1 when at least one relevant result appears in the top K.

## MRR

Measures how highly the first relevant result appears:

```text
MRR = 1 / rank
```

## Average Similarity

Average semantic similarity of retrieved chunks.

## Retrieval Latency

Time spent retrieving relevant chunks.

## End-to-End Latency

Total retrieval-side request duration.

---

# What This Project Demonstrates

- Retrieval-Augmented Generation
- Semantic search
- Embeddings
- Vector databases
- Local LLM inference
- PDF processing
- Context construction
- Retrieval evaluation
- Latency measurement
- Gradio application development
- Modular Python architecture

---

# Quick Start

Once everything has already been configured:

```bash
cd RegForg
source .venv/bin/activate

ollama list

python -m src.retrieval.retriever
python -m src.pipeline.rag_pipeline
python -m evaluation.evaluate_retrieval
python -m src.ui.gradio_app
```

Open:

```text
http://127.0.0.1:7860
```

---

## Author

**Ridham Oza**

AI / ML | Data Science | Generative AI | RAG | LLM Applications



