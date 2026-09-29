# Car Manual Agent -- Hybrid RAG Engine

Answers user questions about the car owner's manual using a hybrid search
pipeline (BM25 + Embedding + RRF Fusion) optimized for edge devices like Jetson Orin.

This is a pure retrieval agent -- it does NOT use an LLM. It returns
matched documents directly and lets the orchestrator synthesize the final answer.

## Architecture

```
User Query
    |
    v
search_mode? --> bm25 -----------------> Top-K Results
              |                               |
              +--> embedding ----------->     |
              |                               v
              +--> hybrid (BM25+Emb) --> RRF Fusion --> Top-K Results
                                                            |
                                                        score >= threshold?
                                                        yes -> 1 document with highest score
                                                        no  -> top 3 documents with highest score
```

## Search Modes

| Mode | Method | Latency | Multilingual |
|---|---|---|---|
| `bm25` | TF-IDF keyword matching | <1ms | No |
| `embedding` | Cosine similarity (sentence-transformers) | ~15ms | Yes |
| `hybrid` (default) | BM25 + Embedding merged via RRF | ~16ms | Yes |

## Cache

All cached data is stored in a single `cache/` folder at the agent root:

- **Embedding model files** (Sentence-Transformers MiniLM weights, tokenizer, config)
- **Pre-computed embeddings** (`.npy` files, loaded in ~10ms instead of re-encoding in ~2-3s)
- **Hash files** (`.hash` files, MD5 of source data to detect changes)

On first run, the embedding model is automatically downloaded from Hugging Face
into `cache/`. Subsequent runs load from disk with no Internet required.

## Data

- `data/toyota.json` -- 100 Q&A pairs (2023 Toyota Corolla Cross)
- `data/mercedes.json` -- 160 Q&A pairs (Mercedes-Benz Actros/Antos/Arocs)
- `data/mmc.json` -- small Mitsubishi/Nissan executive Q&A dataset

## Configuration

All settings are in the root `.env` file. Agent-specific variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAR_MANUAL_PORT` | `8002` | Agent HTTP port |
| `CAR_MANUAL_BRAND` | `toyota` | Car brand to load (`toyota`, `mercedes`, or `mmc`) |
| `CAR_MANUAL_SEARCH_MODE` | `hybrid` | `bm25`, `embedding`, or `hybrid` |
| `CAR_MANUAL_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence-transformers model |
| `CAR_MANUAL_TOP_K` | `3` | Number of top results to return |
| `CAR_MANUAL_SCORE_THRESHOLD` | `0.7` | Below this score, returns 3 docs instead of 1 |

## Files

| File | Description |
|------|-------------|
| `main.py` | Uvicorn entry point, starts the A2A server on the configured port |
| `agent_executor.py` | A2A executor: runs RAG search, formats document output |
| `agent_card.py` | Agent identity, skills, and routing hints for the orchestrator |
| `qa_store.py` | Hybrid RAG engine (BM25 + Embedding + RRF + Cache) |
| `cache/` | Model files + pre-computed embeddings + hash files |
| `prompts.py` | Orchestrator routing description (PURPOSE / INPUT / OUTPUT / ROUTING) |
| `interactive_test.py` | Standalone REPL for testing search without the A2A server |
| `config.yaml` | YAML config (prepared for future migration, not yet loaded at runtime) |

For every configured dataset, including `mmc`, `agent_executor.py` searches the
loaded JSON directly. The JSON file is the single source of truth for available
questions and answers.

## Standalone Testing

Each file can be tested independently via `python <file>`:

```bash
cd agents/car_manual
uv venv --python 3.11
uv sync
uv pip install -e .

# Step 1: RAG engine internals
uv run python qa_store.py -q "tire pressure" --mode hybrid

# Step 2: Prompts and Agent Card
uv run python prompts.py
uv run python agent_card.py

# Step 3: Agent executor
uv run python agent_executor.py -q "How many airbags?"

# Step 4: Interactive REPL
uv run python interactive_test.py --brand toyota --mode hybrid
```

### Interactive REPL Commands

| Command | Description |
|---------|-------------|
| `:mode bm25\|embedding\|hybrid` | Switch search mode |
| `:threshold 0.5` | Change score threshold |
| `:topk 5` | Change number of results |
| `:brand mercedes` | Reload with different brand data |
| `:quit` or `:q` | Exit |

### Full A2A Server

```bash
uv run python main.py
```

Then test the running agent:

```bash
# Check agent card
curl http://localhost:8002/.well-known/agent-card.json

# Send a question via A2A protocol
curl -X POST http://localhost:8002/ \
    -H "Content-Type: application/json" \
    -d '{
      "jsonrpc": "2.0",
      "id": "1",
      "method": "message/send",
      "params": {
        "message": {
          "messageId": "m1",
          "role": "user",
          "parts": [{"kind": "text", "text": "How to open the trunk?"}]
        }
      }
    }'
```

### End-to-end via Orchestrator

Start the full system (see main project README), then:

```bash
curl -X POST http://localhost:8000/message \
    -H "Content-Type: application/json" \
    -d '{"message": "How do I open the trunk?", "session_id": "test"}'
```

The orchestrator should route this to the car_manual agent and return the answer.
