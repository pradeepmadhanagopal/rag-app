# rag-app

A Retrieval-Augmented Generation (RAG) API that answers questions from PDF
documents, grounded strictly in their content. Built from scratch in Python —
chunking, embeddings, retrieval, and generation — then served over HTTP with
FastAPI.

## How it works

PDF → extract text (pypdf) → sliding-window chunking → embed chunks
(sentence-transformers, all-MiniLM-L6-v2) → cosine-similarity retrieval
(top-k + threshold) → grounded generation (Anthropic Claude).

Documents are indexed once at startup; each query embeds only the incoming
question. If no chunk clears the similarity threshold, the API declines to
answer without spending an LLM call.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=your-key' > .env
mkdir docs   # add your PDF here, update the filename in rag.py
fastapi dev app.py
```

Then open http://127.0.0.1:8000/docs for interactive API documentation.

## Endpoints

- `POST /ask` — `{"question": "..."}` → grounded answer from the loaded documents
- `GET /health` — service status

## Current parameters

| Parameter | Value | Notes |
|---|---|---|
| Chunk size | 900 chars | sliding window |
| Overlap | 150 chars | preserves ideas cut at boundaries |
| Top-k | 3 | retrieval candidates |
| Similarity threshold | 0.65 | floor below which chunks are dropped |

These are deliberate starting values, to be tuned against retrieval evals
(RAGAS) in an upcoming iteration.

## Design decisions

- **Grounding with refusal**: the prompt restricts answers to retrieved
  context and instructs the model to say when the context is insufficient —
  verified with out-of-scope negative tests.
- **Top-k + threshold combined**: rank caps cost; the score floor blocks
  confidently-retrieved garbage.
- **Empty-retrieval short-circuit**: no relevant chunks → no API call.
- **Secrets via environment**: key in `.env`, loaded with python-dotenv,
  excluded by `.gitignore`.

## Roadmap

- [x] End-to-end RAG pipeline
- [x] FastAPI service layer
- [x] Dockerise
- [x] Persistent vector store (ChromaDB)
- [ ] Retrieval evals (RAGAS) + parameter tuning
- [ ] Deployment with CI/CD

## Run with Docker

docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app

Note: image is currently ~8.6GB on disk (torch + CUDA libs); slimming via
CPU-only torch is a known optimisation on the roadmap.

**Threshold calibration**: migrating from in-memory cosine scoring to
  ChromaDB's cosine-distance scale silently shifted score ranges — irrelevant
  near-domain chunks moved from ~0.05 to ~0.59, breaking the original 0.25
  threshold. Recalibrated to 0.65 using the gap between genuine matches
  (0.76) and the irrelevant ceiling (0.59), verified with positive and
  negative test queries. Proper eval-set calibration (RAGAS) is the next step.