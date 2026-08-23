---
title: RAG Observability and Latency Analysis using BetterDB
description: An end-to-end Retrieval-Augmented Generation (RAG) pipeline instrumented with BetterDB for Redis / Valkey observability and latency inspection

date: May 19, 2026
modified: May 19, 2026

author: Arpon Kapuria
category: Dev Journal
tags: RAG, Observability
---

This project demonstrates an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline instrumented with `BetterDB` for `Redis/Valkey` **observability** and latency inspection.

The workflow covers:
- document ingestion and semantic retrieval
- Redis-backed caching and session memory
- local Valkey deployment using Docker
- BetterDB agent integration for observability
- latency attribution across the RAG pipeline
- bottleneck analysis for retrieval and inference stages

The stack combines FastAPI, Redis/Valkey, sentence-transformers embeddings, OpenRouter-hosted LLM inference, and BetterDB telemetry to analyze how different pipeline components contribute to overall response latency.

> Github Repository: [betterdb-rag-observability](https://github.com/arponkapuria/betterdb-rag-observability)

## Architecture

Valkey is deployed locally in this setup to minimize Redis network RTT and isolate inference latency from retrieval latency during observability experiments.

```text
PDF Upload → FastAPI (/ingest) → Sentence-Transformers Embeddings → Redis (local Valkey)
                                                                        ↓
User Query → FastAPI (/query) → Rate Limit Check                rag:doc:{sha256}
                              → Semantic Cache Check            semantic_cache:{md5}
                              → HGETALL Retrieval               rate_limit:user_{id}:*
                              → OpenRouter LLM Inference        langchain:memory:session:{id}
                              → Cache Store
                              → Session Write
                                   ↓
                            BetterDB Agent (Docker) → BetterDB Cloud Dashboard
```

## Workflow

### 2.1 Environment Setup

Create and synchronize the Python environment and dependencies using `uv`.

```bash
uv venv
source .venv/bin/activate

uv sync
```

The RAG pipeline implementation is located inside the `rag/` directory. The initial pipeline codebase was generated using Claude Code and later customized for the project requirements.

### 2.2 BetterDB Credentials

To connect BetterDB with the local Redis/Valkey instance:

1. Open the BetterDB dashboard
2. Navigate to: `BetterDB → Manage Connections → Add Connection (via Agent)`
3. Generate an API token
4. Copy:
   - BetterDB WebSocket URL
   - Agent token
   - Generated Docker command

### 2.3 Environment Variables

Create a `.env` file and configure the required credentials. Feel free to use any model provider instead of OpenRouter.

```env
BETTERDB_CLOUD_URL=<betterdb_websocket_url>
BETTERDB_TOKEN=<betterdb_agent_token>

OPENAI_API_KEY=<openrouter_api_key>

REDIS_URL=redis://localhost:6379
```

### 2.4 Local Valkey Setup

This project uses **Valkey**, the open-source fork of Redis, running locally through Docker. 

First create `docker-compose.yaml` and then start Valkey Container.

Pull the image from Docker Hub and start the container.

```bash
docker-compose up -d
```

#### Verify Container Health

Check whether the container is running correctly.

```bash
docker-compose ps
```

#### Observed Response

```text
NAME      STATUS
valkey    Up
```

### 2.5 Connect Valkey to BetterDB

#### Start BetterDB Agent:

Run the BetterDB agent container and connect it to the local Valkey instance.

```bash
docker run -d \
  --name betterdb-agent \
  -e VALKEY_HOST=host.docker.internal \
  -e VALKEY_PORT=6379 \
  -e BETTERDB_CLOUD_URL=<betterdb_websocket_url> \
  -e BETTERDB_TOKEN=<betterdb_agent_token> \
  betterdb/agent:latest
```

> `host.docker.internal` allows the Docker container to access the host machine’s local Redis/Valkey instance (`localhost:6379`).
>
> On Linux, replace it with: `172.17.0.1`

#### Verify BetterDB Agent Connection

Check container logs to confirm the agent successfully connected.

```bash
docker logs betterdb-agent
```

#### Expected log

```text
[Agent] WebSocket connected, sending hello
```

After successful connection:
- the connected Redis instance will appear inside the BetterDB dashboard
- the `betterdb-agent` container should appear in Docker


### 2.6 Model Configuration

#### Chat Model

The project uses `nvidia/nemotron-3-super-120b-a12b:free
` through OpenRouter using the OpenAI SDK.

Set the OpenRouter API key in `.env`:

```env
OPENAI_API_KEY=<openrouter_api_key>
```

#### Embedding Model

The embedding model used is `BAAI/bge-base-en-v1.5` via local inference using `sentence-transformers`.


### 2.7 Run the Application

#### Start the FastAPI Server:

```bash
# Run the application server using Uvicorn
uvicorn rag.main:app --reload --port 8000
```

This starts:
- the FastAPI backend
- the RAG pipeline
- retrieval and generation APIs

#### Verify API Endpoints:

After successful startup, open `http://localhost:8000/docs`. FastAPI Swagger UI should display all available API endpoints.

## Test Endpoints 

### 3.1 Health check (`/health`)

```bash
curl http://localhost:8000/health
```

#### Observed Response
```json
{"redis": "ok", "key_counts": {"rag:doc:": 0, "semantic_cache:": 0, "rate_limit:": 0, "langchain:memory:session:": 0}}
```

### 3.2 Ingest a PDF (`/ingest`)

Upload the PDF manually through the endpoint UI or use the command below.

```bash
curl -F "file=@Arpon-Kapuria-CV.pdf" \
     -H "X-User-ID: student1" \
     http://localhost:8000/ingest
```

#### Observed Response
```json
{"chunks_stored": 14, "source": "Arpon-Kapuria-CV.pdf", "keys_preview": ["rag:doc:abc...", ...]}
```

### 3.3 Check stats (`/stats`)

```bash
curl http://localhost:8000/stats
```

You'll get something like this -
```json
{
  "rag:doc": {
    "count": 14,
    "sample_keys": [
      "rag:doc:f25b6e2bf5cb18aa0432464a718d79f46a166b9429ddc80380dc154e523e2ed0",
      "rag:doc:a95c55da8669b990f483e06ad0747b6f78630fa7a9e29ba9c8001dc7981d8c56",
      "rag:doc:84c31d027bc86669fcc32c3b9899fa0dac3a531f42533ad8257fcf8f3e4c2d09"
    ],
    "sample_ttls": {
      "rag:doc:f25b6e2bf5cb18aa0432464a718d79f46a166b9429ddc80380dc154e523e2ed0": -1,
      "rag:doc:a95c55da8669b990f483e06ad0747b6f78630fa7a9e29ba9c8001dc7981d8c56": -1,
      "rag:doc:84c31d027bc86669fcc32c3b9899fa0dac3a531f42533ad8257fcf8f3e4c2d09": -1
    }
  },
  "semantic_cache": {
    "count": 0,
    "sample_keys": [],
    "sample_ttls": {}
  },
  "rate_limit": {
    "count": 0,
    "sample_keys": [],
    "sample_ttls": {}
  },
  "langchain:memory:session": {
    "count": 0,
    "sample_keys": [],
    "sample_ttls": {}
  },
  "_config": {
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "llm_model": "nvidia/nemotron-3-super-120b-a12b:free",
    "cache_threshold": 0.85,
    "rate_limit_minute": 10
  }
}
```

#### Observe in BetterDB:

- Open BetterDB → your connection → Key Analytics → Trigger Collections (if not updated)
- `rag:doc:*` = 14 keys, ~21 KB/key, **w/TTL = 0** (no expiry)

## Feature: MCP Server Debug

Ask any AI Agent plain-English questions — BetterDB answers from real Redis data. No dashboards needed. The MCP integration works with Claude Code, Cline, or any MCP-compatible AI agent. I am using `cline` with open source models from OpenRouter.

### 4.1 Setup MCP

```json
{
  "mcpServers": {
    "betterdb": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@betterdb/mcp"],
      "env": {
        "BETTERDB_URL": "https://local-test.app.betterdb.com/",
        "BETTERDB_TOKEN": "<your_token>"
      }
    }
  }
}
```

### 4.2 Questions to ask the Agent via MCP

```
"What are the slowest commands in the last 24h?"
"Show me memory breakdown by namespace"
"Who are the top clients by command count?"
"Show me any anomalies detected"
"What keys have no TTL?"
"Show me the complete incident timeline"
```

Each question maps to a BetterDB MCP tool call that returns real data.


## Feature: Semantic Cache TTL Bug

`semantic:cache:*` and `rag:doc:*` keys are written with no TTL — silent memory bloat.

Every semantically distinct query generates a new Redis hash entry. Without TTL expiration, semantic cache growth becomes effectively unbounded over long-running workloads.

### 5.1 Cache Miss
Runs the query for the first time, triggering retrieval and cache creation.

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'x-user-id: demo' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Where did Arpon Kapuria graduate from?", "session_id": "default"}'
```

#### Observed Response:
```json
{"response": "...", "cache_hit": false, "session_id": "default", "docs_used": 3, "latency_ms": 2932.9}
```

### 5.2 Cache Hit
Repeating the same query returns the cached response with lower latency.

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'x-user-id: demo' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Where did Arpon Kapuria graduate from?", "session_id": "default"}'
```

#### Observed Response:
```json
{"response": "...", "cache_hit": true, "session_id": "default", "docs_used": 0,"latency_ms": 402.8}
```

### 5.3 Semantic Cache Hit
A semantically similar query reuses the cached result through embedding similarity.

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'x-user-id: demo' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Arpon Kapuria'\''s graduation college?", "session_id": "default"}'
```

#### Observed Response:
```json
{"response": "...", "cache_hit": true, "session_id": "default", "docs_used": 0,"latency_ms": 373.5}
```

### 5.4 New Query Cache Miss
A different query bypasses the cache and triggers full retrieval again.

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'x-user-id: demo' \
  -H 'Content-Type: application/json' \
  -d '{"query": "Does he have any open source contributions?", "session_id": "default"}'
```

#### Observed Response:

```json
{"response": "...", "cache_hit": false, "session_id": "default", "docs_used": 3,"latency_ms": 7228.8}
```

### 5.5 Observe in BetterDB

- Key Analytics → `semantic:cache:*` → **w/TTL = 0**
- Memory grows with every new unique query
- No expiry = unbounded growth

### 5.6 Ask the Agent

```
"Why does my semantic cache have no TTL, and what happens to hit quality as data changes?"
```

#### The bug in code (`rag/pipeline.py:98`):

```python
# BUG — no expire
r.hset(cache_key(query), mapping={...})

# FIX
r.hset(cache_key(query), mapping={...})
r.expire(cache_key(query), 60 * 60 * 24 * 7)  # 7 days

# Fix rag:doc:* too
r.hset(rag_doc_key(chunk), mapping={...})
r.expire(rag_doc_key(chunk), 60 * 60 * 24 * 30)  # 30 days
```

## Feature: Agent Memory Runaway


`langchain:memory:session:*` hash grows with every query — no TTL, no size limit.

### 6.1 Session Memory Growth
Send multiple queries using the same `session_id` to simulate accumulating agent memory.

```bash
for Q in \
  "What technical skills he has?" \
  "What are his achievements?" \
  "Did he make any open source contributions?" \
  "What does his experience say about him?"; do

  echo -e "\nQuery: $Q"

  curl -s -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -H "x-user-id: demo" \
       -d "{\"query\": \"$Q\", \"session_id\": \"runaway-demo\"}" \
       | python3 -m json.tool
done
```

### 6.2 Observe in BetterDB

- Key Analytics → `langchain:memory:session:*`
- Key count stays **1** (same HASH key)
- `Memory grows` with each query `(850B → 2KB → 4KB → ...)`
- `w/TTL = 0` — grows forever


### 6.3 Inspect Session State
Check the stored Redis session memory and observe hash growth over time.

```bash
curl http://localhost:8000/stats
# Shows langchain:memory:session:runaway-demo key size
```

### 6.4 Ask the Agent

```
"Show agent memory observability — which session key is growing and what is the runaway pattern?"
```

#### The bug in code (`rag/pipeline.py:126`):

```python
# BUG — no expire, no size limit
r.hset(session_key(session_id), f"msg_{idx}", json.dumps(...))

# FIX — rolling window (keep last 20 messages)
pipe = r.pipeline()
pipe.hset(session_key(session_id), f"msg_{idx}", json.dumps(...))
pipe.expire(session_key(session_id), 7200)  # 2 hour TTL
pipe.execute()
# + Add max field count check to trim old messages
```

## Feature: Rate Limiter Abuse Detection

Demonstrates Redis-based rate limiting using the `INCR + EXPIRE` pattern.  

The script simulates repeated requests from the same user to trigger burst traffic, session growth, and observability events visible in BetterDB.


### 7.1. Normal Requests (Within Limit)

Sends 5 sequential requests from the same user.  
All requests should return `HTTP 200`.

```bash
for i in $(seq 1 5); do
  curl -s -o /dev/null \
       -w "Query $i: HTTP %{http_code}\n" \
       -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -H "X-User-ID: abuser" \
       -d '{"query": "test query '$i'", "session_id": "default"}'
done
```

#### Observed Response:

```bash
Query 1: HTTP 200
Query 2: HTTP 200
Query 3: HTTP 200
Query 4: HTTP 200
Query 5: HTTP 200
```

### 7.2 Burst Traffic Simulation

Launches 20 concurrent requests from the same user to intentionally exceed the configured rate limit.

Each `&` runs a request in the background, while `wait` blocks until all requests finish.

```bash
for i in $(seq 1 20); do
  curl -s -o /dev/null \
       -w "Query $i: HTTP %{http_code}\n" \
       -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" \
       -H "X-User-ID: abuser" \
       -d '{"query": "burst test '$i'", "session_id": "default"}' &
done

wait
echo "All requests completed"
```

#### Observed Response:

```bash
Query 1: HTTP 200
Query 2: HTTP 200
...
Query 10: HTTP 200
Query 11: HTTP 429  ← rate limit hit
Query 12: HTTP 429
...
```

The first requests succeed until the Redis rate limit threshold is exceeded. Subsequent requests are rejected with `HTTP 429`.

### 7.3 Observe in BetterDB

- Key Analytics → `rate:limit:user:abuser:*`
- `rate:limit:user:abuser:minute` — **w/TTL = 1** (60s window), value = 20
- `rate:limit:user:abuser:hour` — **w/TTL = 1** (3600s window)
- Click **Trigger Collection** immediately after burst to capture minute key before TTL expires

### 7.4 Ask the Agent

```
"Show me per-client INCR/EXPIRE command breakdown for rate_limit:* keys. Any burst patterns?"
```

#### The rate limit code (`rag/pipeline.py:49`):

```python
# Correct pattern — rate_limit keys DO have TTL (contrast with rag:doc:* which don't)
cnt = r.incr(f"rate_limit:user_{user_id}:minute")
if cnt == 1:
    r.expire(f"rate_limit:user_{user_id}:minute", 60)
if cnt > 10:
    raise HTTPException(429, "Rate limit exceeded")
```

## Feature: Combined Rate Limit + Session Runaway

### 8.1 Section Difference
Compares isolated rate limiting with combined burst traffic and session memory growth with previous section

<div class="comparison-table">

| | Previous Section | This Section (Burst Test) |
|---|---|---|
| `session_id` | `"default"` | `"burst:test"` |
| Purpose | Show 429 rate limit only | Show 429 + session memory growing simultaneously |
| BetterDB signals | `rate:limit:*` keys | `rate:limit:*` AND `langchain:memory:session:burst:test` both change |
| What you watch | Rate limit keys appear | Rate limit burst + session runaway in same test |

</div>

#### In one burst you see 3 features at once:
- **Feature 2** — `semantic:cache:*` grows (new unique queries cached, no TTL)
- **Feature 3** — `langchain:memory:session:burst:test` memory grows (same key, bigger HASH)
- **Feature 4** — queries 11–20 return `HTTP 429`

### 8.2 Burst Traffic Test

Paste entire block into terminal — all 20 fire simultaneously:

```bash
queries=(
  "Summarize the candidate’s educational background."
  "What research areas has the candidate worked on?"
  "What machine learning projects are mentioned in the CV?"
  "What are the candidate’s strongest programming languages?"
  "Describe the candidate’s experience with NLP and LLMs."
  "What internships has the candidate completed?"
  "Which frameworks and libraries does the candidate use?"
  "What backend development experience does the candidate have?"
  "Has the candidate worked on retrieval-augmented generation systems?"
  "What achievements or awards are listed in the CV?"
  "What competitive programming platforms has the candidate used?"
  "What is the candidate’s experience with deep learning?"
  "Which cloud or deployment technologies are mentioned in the CV?"
  "Describe the candidate’s most impactful project."
  "What databases and storage systems has the candidate worked with?"
  "What experience does the candidate have in AI research labs?"
  "Has the candidate published or written technical blogs?"
  "What technologies has the candidate used for full-stack development?"
  "What problem-solving or algorithmic experience does the candidate have?"
  "Summarize the candidate’s overall profile for an AI engineering role."
)

for query in "${queries[@]}"; do
  curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -H "X-User-ID: demo" \
    -d "{\"query\":\"$query\",\"session_id\":\"burst:test\"}" &
done

wait
echo "> All 20 done"
```

### 8.3 Observe in BetterDB

<div class="comparison-table">

| Pattern | Change | Feature |
|---|---|---|
| `rate:limit:user:demo:minute` | Appears with **w/TTL=1**, Avg Idle ~10s | Feature 4 — burst detected |
| `rate:limit:user:demo:hour` | Counter increments | Feature 4 — cumulative |
| `langchain:memory:session:burst:test` | **Memory grows** (same key, bigger HASH) | Feature 3 — runaway |
| `semantic:cache:*` | Count increases by unique queries | Feature 2 — unbounded cache |

</div>

#### Key insight — rate limit vs session memory

- `rate:limit:user:demo:minute`   w/TTL=1  ← RESETS every 60s (correct design)
- `langchain:memory:session:*`    w/TTL=0  ← NEVER resets (the bug)

Rate limit = ephemeral by design. Session memory = permanent by mistake. BetterDB shows both in same dashboard row — `w/TTL` column tells the story.

### 8.4 Ask the Agent

```
"Show me the burst:test session — how much memory has it accumulated vs the rate limit keys?"
```

## Feature: RAG Pipeline Latency Attribution

In this setup, Redis HGETALL is **not** the dominant bottleneck because Valkey is running locally with negligible RTT. The primary latency source is remote LLM inference.

### 9.1 Redis Latency Benchmark
Measure sequential `HGETALL` latency directly on `rag:doc:*` keys.

```bash
python3 -c "
import time, statistics
from rag.config import get_redis

r = get_redis()
keys = list(r.scan_iter('rag:doc:*'))
print(f'Total rag:doc:* keys: {len(keys)}')

latencies = []
for key in keys:
    t0 = time.perf_counter()
    r.hgetall(key)
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
print(f'Per-key HGETALL:  avg={statistics.mean(latencies):.1f}ms  p95={latencies[int(len(latencies)*0.95)]:.1f}ms  max={max(latencies):.1f}ms')

t0 = time.perf_counter()
for key in keys:
    r.hgetall(key)
print(f'Full scan ({len(keys)} keys): {(time.perf_counter()-t0)*1000:.0f}ms total')
"
```

#### Observed Response:
```
Total rag:doc:* keys: 14
Per-key HGETALL:  avg=0.7ms  p95=1.1ms  max=1.1ms
Full scan (14 keys): 9ms total
```

This confirms that:
- Redis latency is extremely low
- Sequential HGETALL retrieval is healthy
- Redis contributes negligible overhead to the pipeline

### 9.2 API Response Latency
Compare Redis retrieval time with end-to-end RAG query latency.

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -H "X-User-ID: student1" \
     -d '{"query": "What AWS skills this candidate has?", "session_id": "demo"}' \
     | python3 -m json.tool
```

#### Observed Response:
```bash
{
    "response": "The candidate’s AWS experience includes working with S3, EKS, IAM, and VPC (provisioning infrastructure via Terraform on AWS), along with CI/CD pipelines on the platform.",
    "cache_hit": false,
    "session_id": "demo",
    "docs_used": 3,
    "latency_ms": 9968.9
}
```

The total query latency is: `9969.9ms ≈ 10 second`

> The high inference latency is primarily due to using a free-tier OpenRouter-hosted model endpoint, which introduces queueing, cold-start, and remote inference overhead.

### 9.3 RAG Latency Waterfall
Break down latency across embedding, retrieval, Redis, and LLM generation stages.

```
Embedding call:     ~300ms   (Embedding model/API)
Vector search:      ~20ms    (Retriever / similarity search)
Redis scan+HGETALL: ~9ms     (14 keys total)
Prompt assembly:    ~20ms
LLM generation:     ~10000ms ← BOTTLENECK
──────────────────────────────────────────
Total:              ~10400ms
```

#### Latency Attribution:

Redis contribution: `9ms / 10400ms ≈ 0.09%`
Therefore:
- Redis is NOT the bottleneck *( partly because of small `rag:doc:*` )*
- HGETALL retrieval is effectively negligible
- The system is inference-bound, not retrieval-bound

### 9.4 Ask the Agent

```
"Show per-stage latency breakdown for the RAG pipeline and identify the dominant source of latency."
```

#### Notes on Optimization: 

Even though Redis is already fast, batched retrieval is still better practice. However, in this setup, pipelining will not significantly reduce total latency because Redis already contributes <0.1% of request time.

Pipeline batching reduces repeated client-server round trips by sending all `HGETALL` commands in a single network request.


```python
# FIX 1: Pipeline — batch all HGETALLs
pipe = r.pipeline(transaction=False)
for key in keys:
    pipe.hgetall(key)
results = pipe.execute()

# FIX 2: Store only the embedding hash, retrieve content separately
```

> If Redis were hosted remotely (cloud/managed Redis), sequential HGETALL calls could become a major bottleneck due to network RTT. Local Redis remains extremely fast.

## Quick Reference: curl Commands

Common API commands for ingestion, querying, stress testing, and observability experiments.

```bash
# Health check
curl http://localhost:8000/health

# Ingest PDF
curl -F "file=@Arpon-Kapuria-CV.pdf" -H "X-User-ID: demo" http://localhost:8000/ingest

# Query (first time — cache miss)
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" -H "X-User-ID: demo" \
     -d '{"query": "Where did the candidate graduate from?", "session_id": "default"}'

# Query (same — cache hit)
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" -H "X-User-ID: demo" \
     -d '{"query": "Where did the candidate graduate from?", "session_id": "default"}'

# Stats
curl http://localhost:8000/stats

# Trigger rate limit (run this 11+ times rapidly)
for i in $(seq 1 15); do
  curl -s -o /dev/null -w "HTTP %{http_code}\n" \
       -X POST http://localhost:8000/query \
       -H "Content-Type: application/json" -H "X-User-ID: abuser" \
       -d '{"query": "test '$i'", "session_id": "test"}'
done
```

## All MCP Questions

Ask these directly in `Agent` terminal after connecting `BetterDB MCP`:

**Feature 1 — Incident overview**
```
"What are the slowest commands in the last 24h?"
"Show me memory breakdown by namespace"
"Who are the top clients by command count?"
"Show me any anomalies detected"
"Show the complete incident timeline"
```

**Feature 2 — TTL bug**
```
"Why does my semantic cache have no TTL?"
"What happens to hit quality as underlying data changes?"
"Show me key analytics — which namespaces have zero TTL coverage?"
"How much memory will semantic_cache:* consume in 30 days at current growth?"
```

**Feature 3 — Agent memory**
```
"Show agent memory observability — which session is the runaway?"
"How does BetterDB identify which session key caused the latency spike?"
"What is the memory growth pattern for langchain:memory:session:* keys?"
```

**Feature 4 — Rate limiter**
```
"Show per-client INCR/EXPIRE command breakdown for rate_limit:* keys"
"Any burst patterns from a specific IP or user ID?"
"What is the 30-second burst window for the abuser user?"
```

**Feature 5 — RAG latency**
```
"Show BetterDB per-command latency breakdown for RAG pipeline"
"Identify HGETALL commands on rag:doc:* as the latency source"
"Show before/after TTL fix reducing HGETALL from the slowlog"
"What is the p95 latency for HGETALL on rag:doc keys?"
```

## Key Patterns Reference

<div class="comparison-table">

| Redis Key | Written by | TTL | BetterDB Feature |
|---|---|---|---|
| `rag:doc:{sha256}` | `POST /ingest` | **None (bug)** | Feature 2, 5 |
| `semantic_cache:{md5}` | `POST /query` (miss) | **None (bug)** | Feature 2 |
| `rate_limit:user_{id}:minute` | `POST /query` | 60s ✓ | Feature 4 |
| `rate_limit:user_{id}:hour` | `POST /query` | 3600s ✓ | Feature 4 |
| `langchain:memory:session:{id}` | `POST /query` | **None (bug)** | Feature 3 |

</div>