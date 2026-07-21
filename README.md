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
- [x] Retrieval evals (RAGAS) + parameter tuning
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


A 6-case eval suite (4 positive, 2 negative) measures retrieval quality on
every change: `python3 eval_retrieval.py`.

| Experiment | Hit rate | Notes |
|---|---|---|
| Baseline | 5/6 | scoring bug found & fixed first (see below) |
| + text cleaning | 5/6 | artifacts fixed; failing case unmoved — not the cause |
| + overlap 150→250 | 5/6 | failing case unmoved; degraded a passing case → reverted |

The eval harness itself caught a real bug on first run: ChromaDB 1.x silently
ignores the legacy `hnsw:space` metadata, so the collection used squared-L2
while scores were converted with the cosine formula — every retrieval failed
the threshold. Diagnosed from raw distances (L2² = 2·cosine-distance for
normalised embeddings) and fixed with the correct conversion.

### Known limitation: vocabulary mismatch

One eval case fails persistently: "What did Hubel and Wiesel discover?"
scores 0.56 against a chunk that is intact, clean, and contains the answer.
Root cause: the question's framing ("discover") shares little semantic
overlap with the slide's content (descriptions of cortex cell behaviour),
and rare proper nouns are diluted in dense embeddings — a known limitation
of pure semantic retrieval. Standard remedy is hybrid retrieval
(dense + BM25 keyword search), scoped as future work. Lowering the
threshold is not viable: negative-case chunks score up to 0.59.