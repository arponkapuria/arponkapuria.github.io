---
title: DocRAG
page_title: DocRAG: Cost-Aware Multi-Tenant Agentic RAG Pipeline
description: Project walkthrough of DocRAG, a multi-tenant agentic RAG pipeline for document Q&A.
tagline: A cost-aware, multi-tenant Agentic RAG pipeline for document Q&A
author: Arpon Kapuria
author_url: https://arponkapuria.github.io
stack: Python, FastAPI, LangGraph, Docling, Qdrant, Redis Stack, PostgreSQL, MinIO, Langfuse, RAGAS, Docker
cover: David - The Death of Socrates.jpg
cover_position: center 40%
---

## System diagram {#system}

```diagram
id: d-system
size: 920 672
alt: System architecture: browser, FastAPI process with session, guardrail, cache, rewriter, LangGraph agent, LLM client and ingestion, plus Postgres, Redis Stack, Qdrant, MinIO and external services
caption: Everything the API process owns, the four stores it talks to, and the outside services it calls.
---
{
  groups: [
    [215, 20, 490, 505, 'FastAPI process (services/api)'],
    [225, 222, 470, 195, 'LangGraph agent', 50],
    [735, 20, 170, 505, 'Stores'],
    [215, 545, 690, 95, 'External services', 150]
  ],
  nodes: [
    ['br', 'Browser\nstatic chat UI', 25, 270, 150, 80, 'white'],
    ['sess', 'Session cookie +\nrate limits', 235, 55, 210, 60, 'blue'],
    ['guard', 'Input guardrail\n(regex, 4000 chars)', 495, 55, 190, 60, 'pink'],
    ['rw', 'Query rewriter\n(+ optional HyDE)', 235, 145, 210, 60, 'purple'],
    ['cache', 'Response cache\nL1 exact, L2 semantic', 495, 145, 190, 60, 'yellow'],
    ['plan', 'Planner', 245, 300, 100, 55, 'orange'],
    ['retr', 'Retriever\nhybrid + rerank', 380, 262, 135, 50, 'green'],
    ['tool', 'Tool node\nweb or sandbox', 380, 348, 135, 50, 'green'],
    ['resp', 'Responder', 560, 300, 115, 55, 'orange'],
    ['llm', 'LLM client\nGroq, OpenRouter\nmodel tiers', 235, 440, 210, 70, 'blue'],
    ['ing', 'Ingestion pipeline\nDocling, embed, index', 475, 440, 210, 70, 'green'],
    ['pg', 'Postgres\nchat, docs,\nfeedback', 750, 45, 140, 85, 'grey', 'db'],
    ['rd', 'Redis Stack\nsessions, cache,\nrate limits', 750, 150, 140, 95, 'grey', 'db'],
    ['qd', 'Qdrant\ndense + sparse\nvectors', 750, 270, 140, 95, 'grey', 'db'],
    ['s3', 'MinIO / S3\nuploads, then\nwebhook', 750, 395, 140, 95, 'grey', 'db'],
    ['gq', 'Groq /\nOpenRouter', 235, 575, 180, 50, 'white'],
    ['ms', 'Mistral vision\n(figure captions)', 430, 575, 180, 50, 'white'],
    ['tv', 'Tavily\nsearch', 625, 575, 120, 50, 'white'],
    ['sb', 'Code sandbox\ncontainer', 760, 575, 130, 50, 'white']
  ],
  edges: [
    ['br', 'sess', 'requests', 'r', 'l', { lx: 205, ly: 200 }],
    ['sess', 'guard', '', 'r', 'l'],
    ['guard', 'cache', 'clean', 'b', 't', { lx: 600, ly: 130 }],
    ['cache', 'rw', 'miss', 'l', 'r'],
    ['rw', 'plan', '', 'b.1', 't.1'],
    ['plan', 'retr', '', 'r.15', 'l'],
    ['plan', 'resp', 'direct', 'r', 'l', { lx: 460, ly: 327 }],
    ['plan', 'tool', '', 'r.9', 'l'],
    ['retr', 'resp', '', 'r', 't.3'],
    ['tool', 'resp', '', 'r', 'b.3'],
    ['resp', 'llm', '', 'b.8', 't', { via: [[652, 406], [340, 406]] }],
    ['cache', 'rd', '', 'r', 'l'],
    ['guard', 'pg', 'messages', 'r', 'l', { lx: 718, ly: 70 }],
    ['retr', 'qd', '', 't', 'l.2', { via: [[447, 230], [722, 230], [722, 289]] }],
    ['s3', 'ing', '', 'l.8', 'r.45', {}],
    ['ing', 'qd', '', 'r.12', 'l.9', { via: [[712, 448], [712, 355]] }],
    ['llm', 'gq', '', 'b', 't'],
    ['ing', 'ms', '', 'b', 't']
  ],
  notes: [
    [545, 662, 'the tool node calls Tavily and the sandbox', 14]
  ]
}
```

## Overview {#overview}

DocRAG lets someone open a web page, upload a few documents, and ask questions about them. The answers come back grounded in those documents, with the source filename cited. There is no login. Instead, the first request mints a random `corpus_id` and stores it in an httpOnly cookie, and every store in the system (Qdrant, Postgres, Redis, object storage) is filtered or tagged by that id. A background loop deletes everything for a session once it has been inactive for an hour.

When a file is uploaded, the browser PUTs it straight to MinIO using a presigned URL, and MinIO calls a webhook on the API. That kicks off an in-process ingestion run: Docling parses the document and splits it into token-bounded chunks, a vision model captions any figures, FastEmbed produces a dense and a sparse (BM25) vector per chunk, and everything is upserted into one Qdrant collection tagged with the session's id. A Redis counter called `corpus_version` is bumped so the cache knows the corpus changed.

A chat request first passes the rate limiter and a regex input guardrail, then the two-layer cache (exact match, then semantic match). On a miss, an optional rewriter resolves pronouns from the conversation history, and a LangGraph state machine takes over: a planner LLM decides whether to retrieve, call a tool (the Python sandbox, or web search after the user agrees to it), or just answer. Retrieval runs dense and sparse search in Qdrant, fuses them with Reciprocal Rank Fusion, and reranks the top ten with a cross-encoder down to five. The responder writes the answer with citations, an output guardrail checks it, and the result is streamed back as NDJSON status events followed by the final answer. LLM calls go through a Groq-first client with per-model rate tracking, a circuit breaker, retries with backoff, and an optional OpenRouter failover. I also built an evaluation harness around the real pipeline: 25 hand-verified questions over five papers, scored with deterministic metrics and RAGAS judge metrics.

## File structure by component {#files}

The repo is split by what each piece does at runtime. Each tree below lists only the files that belong to that component. The deep dives further down walk through them one at a time.

#### Sessions and edge protection

```
services/api/app/session/
  store.py          sessions:active sorted set in Redis, sliding TTL
  dependency.py     get_corpus_id(): cookie -> validated corpus_id (401 otherwise)
  rate_limit.py     fixed-window limits per IP and per session
  cleanup.py        background purge + cascade delete across all stores
services/api/app/routes/session.py     POST /session/init
libs/guardrails/
  patterns.py       compiled regex lists (input and output)
  filters.py        check_input, check_output, guard_output
  plan_schema.py    validate_plan(): sanitizes the planner's JSON
libs/utils/refusal.py                  refusal / truncation detection, fixed messages
```

#### Chat request and agent

```
services/api/main.py                   app wiring, lifespan startup/shutdown
services/api/app/routes/chat.py        POST /chat/stream (cache, enhance, agent, persist)
services/api/app/routes/feedback.py    thumbs up/down
services/api/app/routes/health.py      liveness / readiness
services/api/app/agents/
  graph.py          StateGraph: planner -> retriever | tool | responder
  state.py          AgentState TypedDict
  nodes/planner.py  decides the action, flags existence-check questions
  nodes/retriever.py  hybrid search + rerank
  nodes/tool.py     runs web_search / sandbox
  nodes/responder.py  grounded answer, refusal normalization, sandbox echo
services/api/app/enhancers/
  query_rewriter.py   resolves "it / that / he" using history
  hyde.py             optional hypothetical-document query
libs/utils/model_router.py             simple vs complex model tier
```

#### Retrieval and embeddings

```
services/api/app/clients/
  qdrant.py             collection setup, hybrid / dense / sparse search, delete by corpus
  embedding.py          async wrapper around FastEmbed (+ unused OpenRouter template)
  fastembed_client.py   dense (bge-large) + sparse (Qdrant/bm25) models, ONNX/CPU
  fastembed_reranker.py bge-reranker-base cross-encoder
  reranker.py           async reranker, optional OpenRouter failover
services/api/app/tools/vector_search.py   retrieval exposed as a tool
```

#### LLM layer

```
services/api/app/clients/llm/
  base.py               LLMClient interface
  openai_compatible.py  the engine: model list, rate limiter, breaker, retry
  groq_client.py        Groq limits + round-robin dispatch
  openrouter_client.py  OpenRouter shared account limits
  ollama_client.py, vllm_client.py   manual-select local backends
  mistral_client.py     vision captioning only
  factory.py            builds the active client, FailoverLLMClient
libs/utils/
  backoff.py            exponential backoff + jitter decorator
  circuit_breaker.py    closed / open / half-open
  rate_limiter.py       client-side RPM/RPD/TPM/TPD tracking
```

#### Cache

```
services/api/app/cache/
  redis.py          singleton connection for sessions and health
  redis_cache.py    L1 exact + L2 semantic (RediSearch HNSW), corpus_version logic
```

#### Ingestion

```
services/api/app/routes/
  upload.py         presigned PUT URL + pending Document row
  webhooks.py       MinIO event -> BackgroundTask
  ingest_status.py  per-file stage polling
  corpus.py         list the caller's documents
pipelines/ingestion/
  pipeline.py       run_ingestion(): fetch, parse, embed, index, bump version
  loaders/docling_loader.py   Docling convert + HybridChunker + figure captions
  chunking/metadata.py        content hash, timestamp, length
  indexing/qdrant.py          column-batch upsert into Qdrant
  debug_dump.py               optional on-disk dump of chunks
  config.yaml                 reference copy of the ingestion settings
libs/utils/s3_client.py       boto3 client (MinIO or AWS) + delete by prefix
```

#### Sandbox

```
services/sandbox/
  runner.py          Flask /execute: child process, timeout, stdout capture
  Dockerfile         slim image, non-root user
  limits.yaml        timeouts and limits (reference)
  network-policy.yaml  Kubernetes: deny all egress
services/api/app/tools/sandbox.py   HTTP client for the sandbox
```

#### Evaluation

```
eval/
  cli.py                 one entry point for every stage
  config.py              fixed corpus id and the 5 pinned arXiv papers
  dataset/questions.json 25 questions with reference answers and gold spans
  ingest_corpus.py       runs the real ingestion path on the papers
  verify_gold.py         checks every gold span exists in an indexed chunk
  run_retrieval.py       dense / BM25 / hybrid / hybrid+rerank ablation
  run_generation.py      shipped agent vs closed-book baseline
  run_judge.py           RAGAS metrics + refusal judge
  metrics/deterministic.py   Hit@k, MRR, Precision@k, refusal regex, citation validity
  judges/, clients/      adapters for Gemma and Cohere judges, FastEmbed embeddings
  report.py              aggregates everything into results/report.md
  storage.py             one JSON file per question per stage, atomic writes
```

#### Data, frontend and tooling

```
services/api/app/memory/models.py     ChatHistory, Feedback, Document tables
services/api/app/memory/postgres.py   PostgresMemory, DocumentStore
services/api/app/config.py            one pydantic Settings object for every tunable
services/api/app/logging.py           console + rotating JSON file logs
services/api/static/index.html        chat UI (vanilla JS)
docker-compose.yaml                   Postgres, Redis Stack, Qdrant, MinIO, sandbox (profiles)
Makefile, pyproject.toml, uv.lock     commands and dependency groups
scripts/                              load_test.py (Locust), bulk_upload_s3.py
tests/                                guardrails, LLM client, rate limit, session, truncation
models/                               reference YAMLs for the models (not loaded at runtime)
```

## Tech stack and design choices {#stack}

The guiding idea is simplicity: a small system that is easy to reason about, with heavier pieces added only when a real need appears. Each row starts with the **need** the tool had to meet and how it fits. Every alternative is then judged against that same need.

:::table stack 17% 38%
| Choice | Why I picked it | Why not the alternatives |
| --- | --- | --- |
| **FastAPI + uvicorn** | **Need:** handle many chats at once, stream answers, and reuse session and rate-limit checks.<br>**Fit:** the app spends almost all its time waiting on LLM, Qdrant, Redis and Postgres, so async lets one process serve many chats. Dependency injection lets me write the session check and rate limiter once and attach them to any route. `StreamingResponse` carries the chat stream. | **Flask**: sync-first, so the async I/O and streaming this path needs would be bolted on. I still use it for the sandbox, where one blocking endpoint is enough.<br>**Django**: its ORM, admin and templates solve problems this API does not have.<br>**Node / Express**: Docling, FastEmbed, LangGraph and RAGAS are all Python, so a Node API would split the codebase into two languages. |
| **LangGraph** | **Need:** an agent with a few fixed steps that I can test and observe.<br>**Fit:** planner, retriever, tool and responder are separate nodes over one shared state. Each node can be tested alone, the UI gets one status event per node, and a future re-plan loop is one extra edge. | **Plain async functions**: enough for four nodes, but I would hand-write the state passing, the per-node events and any future loop. That is exactly what LangGraph provides.<br>**ReAct / LangChain agent executor**: the LLM decides every step, so one question costs several LLM calls. This agent needs one planner call, and free-tier quotas can't absorb more. |
| **Qdrant (one collection, named vectors)** | **Need:** hybrid search (dense + BM25) with per-user isolation, in one call.<br>**Fit:** one query does dense search, sparse search, RRF fusion and a `corpus_id` filter, all server-side. Multi-tenancy is a payload filter, not a collection per user. | **pgvector**: dense search only. BM25 and fusion would be mine to build, and Postgres full-text ranking is not BM25.<br>**Elasticsearch / OpenSearch**: does hybrid search, but a JVM cluster is too heavy for a free-tier, single-box demo.<br>**FAISS / Chroma**: neither gives sparse + dense + filter + fusion in one call.<br>**A collection per tenant**: many small collections cost more memory and overhead than one filtered collection. |
| **FastEmbed: `bge-large-en-v1.5` (dense), `Qdrant/bm25` (sparse)** | **Need:** embeddings with no GPU, no per-call cost, and identical handling of documents and queries.<br>**Fit:** ONNX runs on CPU. The same in-process client embeds both documents and queries, so the two paths cannot drift apart. It is the only embedder on purpose, because the collection is fixed at 1024 dimensions. | **Hosted embedding APIs (OpenAI, Cohere)**: a paid, rate-limited call for every chunk at ingest. A fallback model with a different dimension would silently corrupt the index.<br>**sentence-transformers / torch**: heavier install and built around GPUs.<br>**bge-m3**: my original choice, but FastEmbed does not ship it. |
| **`bge-reranker-base` cross-encoder** | **Need:** better precision in the final top 5 without much added latency.<br>**Fit:** a cross-encoder reads the query and each passage together, so it is more accurate than comparing two independent vectors. It only reorders the 10 fused candidates, which keeps it fast. It lifted Hit@5 from 0.619 to 0.667 in the eval. | **LLM-based reranking**: spends LLM quota on every query. It exists only as an optional failover.<br>**Hosted rerank API (for example Cohere)**: another paid, external call on the hot path.<br>**No reranker**: cheapest, but the ablation shows the Hit@5 gain is worth keeping. |
| **Docling + `HybridChunker`** | **Need:** chunks that respect document structure, from mixed file types.<br>**Fit:** one path for PDF, DOCX, HTML and Markdown that keeps table structure, finds figures, and returns a heading tree the chunker uses. Chunks are bounded by the embedder's own tokenizer, and tables become compact Markdown. | **PyMuPDF / pdfplumber**: fast, but they flatten tables and give no heading hierarchy.<br>**LangChain text splitters**: split by characters, so they ignore headings and tables.<br>**Unstructured**: the closest comparison. Docling already covered the need, so I did not run a second parser. |
| **Groq primary, OpenRouter backup, one OpenAI-compatible client** | **Need:** zero cost, and staying up when a free tier rate-limits.<br>**Fit:** Groq's three models have separate quotas, and every backend speaks `/chat/completions`, so one engine class serves all of them. Failover triggers only when a backend's whole model list is exhausted, with a circuit breaker on top. | **OpenAI / Anthropic APIs**: paid, which breaks the zero-cost goal.<br>**A single provider**: one free-tier 429 would take the demo down.<br>**Ollama / vLLM**: implemented behind the same interface but not validated live, and cold starts make them poor failover targets, so they are manual-select only. |
| **Mistral `ministral-14b` for figure captions** | **Need:** a vision model that does not compete with chat for quota and cannot bias the eval.<br>**Fit:** it sees images, has its own rate limit, and is a different vendor from the eval judge. | **The chat models**: `gpt-oss` is text-only, and captioning would also eat the chat quota.<br>**Docling's built-in captioning**: bypasses the shared rate limiter and circuit breaker. |
| **Postgres 15 (SQLAlchemy async)** | **Need:** durable, relational records that I update and delete per tenant.<br>**Fit:** chat history, feedback and per-document ingestion status are records I update and query. Deleting a whole tenant is one transaction across three tables. | **SQLite**: one writer on a file is awkward with async access from several containers.<br>**MongoDB**: nothing here needs a flexible schema, and the data is relational.<br>**Redis for history**: ephemeral by design. An eviction or restart would lose conversations, and there is no relational querying. |
| **Redis Stack** | **Need:** fast shared state: sessions, rate limits, and two cache layers.<br>**Fit:** one container gives sessions (sorted set), rate-limit counters, the L1 exact cache, and the L2 semantic cache through RediSearch HNSW. | **Plain Redis + a separate vector store**: one more service just for the semantic cache.<br>**In-process dictionaries**: not shared across replicas and lost on restart.<br>**Memcached**: no sorted sets and no vector search. |
| **MinIO + boto3 + presigned URLs** | **Need:** uploads that do not tie up the API, with the same code locally and in production.<br>**Fit:** S3-compatible, so the same boto3 code runs against MinIO locally and real S3 in production (only `S3_ENDPOINT_URL` changes). Presigned PUTs mean file bytes never pass through the API, and MinIO's webhook starts ingestion. | **Uploading through FastAPI**: the API would buffer files and tie up workers.<br>**Local disk**: not shared between containers, and no upload event to trigger ingestion.<br>**AWS S3 only**: needs credentials and cost even for local development. |
| **Flask sandbox in its own container** | **Need:** run model-generated code without putting the API at risk.<br>**Fit:** the code never runs in the API process. It runs in a child process with a hard timeout, as a non-root user, and a Kubernetes NetworkPolicy denies all egress. The image needs only Flask and PyYAML. | **`exec` inside the API**: a bad snippet could read or crash the API itself.<br>**RestrictedPython**: Python-level sandboxes have a history of escapes, so I trust a container boundary more.<br>**Firecracker / gVisor**: stronger isolation but more setup. gVisor is on the hardening list. |
| **Tavily web search** | **Need:** web search that is free, returns clean text, and is used rarely.<br>**Fit:** free tier, clean text output, and it is only called after the user agrees to an explicit offer. | **Scraping search pages**: brittle and against most terms of service.<br>**Bing / Google paid APIs**: cost and key management for a rarely used path.<br>**Brave / Serper**: comparable. Tavily's free tier was enough. |
| **Regex guardrails** | **Need:** protect an open demo with no login, at no per-request cost.<br>**Fit:** input and output pattern lists plus rate limits. They need no model call, so every decision is instant, free and unit-testable. | **LLM moderation call**: adds latency, spends quota, and can itself be prompt-injected.<br>**Llama Guard / NeMo Guardrails**: another model or service to host.<br>**Nothing**: not acceptable for an open endpoint. |
| **RAGAS + Gemma / Cohere judges** | **Need:** established RAG metrics, scored by a judge that is not biased toward the generator.<br>**Fit:** real RAGAS metrics, kept in their own `eval` dependency group because they pull in about 30 packages that do not belong in the deployed image. The judge is a different vendor from the generator. | **Hand-written metric prompts**: unvalidated and easy to drift.<br>**DeepEval / TruLens**: similar tools. RAGAS already covered the metrics I needed.<br>**Same vendor as judge**: a model tends to favour its own style. |
| **uv + Docker Compose profiles** | **Need:** lean installs, and starting only the services a task needs.<br>**Fit:** dependency groups (`api`, `dev`, `ingestion`, `eval`) keep RAGAS out of the API image. Every container sits behind a profile, so you start only what you need. | **pip + requirements files**: no clean way to keep the eval stack out of the API install.<br>**Poetry**: also has groups, but uv is faster and gives a reproducible lockfile.<br>**One big Compose stack**: starts everything every time, which is heavy on a laptop. |
:::

## Sessions and tenant isolation {#c1}

This is the part that lets the app run without a login while still keeping every visitor's documents private. Everything else in the system leans on it, so it comes first.

```diagram
id: d-c1
size: 920 575
alt: Session flow: session init, cookie validation, per-request validation, and the background cleanup loop with cascade delete
caption: Session creation, per-request validation, and the cleanup loop that deletes expired tenants.
---
{
  notes: [
    [20, 20, 'First request, or a page reload', 17, 'start', '#777'],
    [20, 160, 'Every route that touches data', 17, 'start', '#777'],
    [20, 372, 'Background cleanup', 17, 'start', '#777']
  ],
  nodes: [
    ['a', 'Browser\nPOST /session/init', 20, 50, 150, 60, 'white'],
    ['b', 'IP rate limit\n30 per hour', 200, 50, 130, 60, 'pink'],
    ['c', 'valid cookie\nin Redis?', 360, 35, 140, 90, 'yellow', 'diamond'],
    ['d', 'create_session\nuuid4 + ZADD', 545, 25, 150, 50, 'blue'],
    ['e', 'touch(existing)\nrefresh cookie', 545, 90, 150, 50, 'blue'],
    ['f', 'Set-Cookie\nhttpOnly, Lax,\nSecure in prod', 730, 45, 170, 70, 'green'],
    ['g', 'Request\n+ cookie', 20, 190, 130, 60, 'white'],
    ['h', 'get_corpus_id()', 180, 190, 150, 60, 'blue'],
    ['i', 'ZSCORE:\nage < 60 min?', 360, 175, 150, 90, 'yellow', 'diamond'],
    ['j', 'touch()\nZADD XX', 550, 190, 130, 60, 'blue'],
    ['k', 'corpus_id\nto the route', 710, 190, 150, 60, 'green'],
    ['l', 'HTTP 401\nNo active session', 365, 295, 160, 50, 'pink'],
    ['m', 'Loop wakes every\nmax(60s, TTL/2)', 20, 395, 190, 55, 'white'],
    ['n', 'purge_expired\nZRANGEBYSCORE + ZREM', 240, 395, 190, 55, 'blue'],
    ['o', 'cascade delete\nper corpus_id', 470, 395, 150, 55, 'orange'],
    ['p1', 'Qdrant\nfilter delete', 20, 500, 190, 52, 'grey'],
    ['p2', 'Postgres\nhistory, feedback, docs', 250, 500, 190, 52, 'grey'],
    ['p3', 'Redis cache\nSCAN + DEL', 480, 500, 190, 52, 'grey'],
    ['p4', 'S3 / MinIO\nuploads/{corpus_id}/', 710, 500, 190, 52, 'grey']
  ],
  edges: [
    ['a', 'b', '', 'r', 'l'], ['b', 'c', '', 'r', 'l'],
    ['c', 'd', 'no', 'r', 'l', { lx: 522, ly: 50 }],
    ['c', 'e', 'yes', 'r', 'l', { lx: 522, ly: 115 }],
    ['d', 'f', '', 'r', 'l'], ['e', 'f', '', 'r', 'l'],
    ['g', 'h', '', 'r', 'l'], ['h', 'i', '', 'r', 'l'],
    ['i', 'j', 'yes', 'r', 'l'], ['j', 'k', '', 'r', 'l'],
    ['i', 'l', 'no', 'b', 't'],
    ['m', 'n', '', 'r', 'l'], ['n', 'o', '', 'r', 'l'],
    ['o', 'p1', '', 'b.1', 't'], ['o', 'p2', '', 'b.35', 't'],
    ['o', 'p3', '', 'b.65', 't'], ['o', 'p4', '', 'b.9', 't']
  ]
}
```

##### Flow

1. When the page loads, the frontend calls `POST /api/v1/session/init`. This route is rate-limited by IP only (30 per hour).
2. If the request already carries a cookie and Redis says that session is still alive, the session is touched and the cookie is re-sent. Otherwise a new `corpus_id` (a uuid4) is created and added to the `sessions:active` sorted set with the current time as its score.
3. The cookie is set `httpOnly`, `SameSite=Lax`, `Secure` when `ENV=prod`, with a max age equal to the session TTL (60 minutes).
4. Every route that touches data declares `Depends(get_corpus_id)`. It reads the cookie, checks the Redis score against the TTL, refreshes the score, and returns the id. A missing or expired cookie gets a 401.
5. A background loop wakes up periodically, removes sessions that have been inactive past the TTL, and cascade-deletes everything tagged with each expired id.

##### Explanation

The session store is a Redis sorted set where each member is a `corpus_id` and the score is the last-active Unix time. I chose a sorted set over plain keys with a TTL because the cleanup job needs to know *which* ids expired so it can delete their data in four other systems. A key that expires on its own just disappears, and I would have no list to work from. With a sorted set, finding the expired ones is one range query by score:

```python services/api/app/session/store.py
async def touch(self, corpus_id: str) -> None:
    client = redis_client.get_client()
    await client.zadd(ACTIVE_SESSIONS_KEY, {corpus_id: time.time()}, xx=True)

async def purge_expired(self, ttl_minutes: int) -> List[str]:
    cutoff = time.time() - (ttl_minutes * 60)
    expired = await client.zrangebyscore(ACTIVE_SESSIONS_KEY, min=0, max=cutoff)
    if expired:
        await client.zrem(ACTIVE_SESSIONS_KEY, *expired)
    return list(expired)
```

`touch` uses `xx=True`, which means "only update if the member already exists". That stops a request with a stale or forged id from re-creating a session. There is a unit test for exactly that case.

The important rule is that `corpus_id` only ever comes from the cookie. No route accepts it in the body or query string, so a client cannot reach another client's data by guessing an id. The tenant key then shows up everywhere:

- **Qdrant:** every search carries a `corpus_id` payload filter, and every indexed point is tagged with it.
- **Postgres:** chat history, feedback and document rows all have a `corpus_id` column. The ingestion-status route returns 404 if the document belongs to a different corpus, so ownership is checked, not just existence.
- **Redis:** cache keys start with `cache:{corpus_id}:`, and rate-limit keys include it.
- **Object storage:** uploads live under `uploads/{corpus_id}/`, and the ingestion pipeline reads the corpus id *out of that key* rather than trusting anything the client sends.

The cleanup loop lives in `session/cleanup.py`. Its poll interval is `max(60, TTL × 60 / 2)` seconds, so with the default 60-minute TTL it checks every 30 minutes. For each expired id it deletes from Qdrant (filter delete), Postgres (three tables in one transaction), the Redis cache (a `SCAN` with a prefix pattern, so it never blocks a live instance), and S3 (list and delete by prefix). Each store is wrapped in its own try/except, so one failing store does not stop the others from being cleaned.

:::qa Why no login at all?
A login would add a user table and passwords, and it would not remove the real risks anyway. An unguessable session id, a one-hour TTL, tenant filters on every store and rate limits cover those. The goal was a simple system: open the page, upload, ask. The trade-off is that data is temporary by design.
:::

:::qa How do you stop one visitor from reading another's documents?
Four layers: the id comes only from an httpOnly cookie, it is a random uuid4 so it cannot be guessed, every query is filtered by it at the database level, and the ingestion pipeline derives it from the S3 key instead of any client field. The weak spot: the cookie value *is* the corpus id, so anyone who obtains it (for example from a log line) can act as that session until it expires.
:::

:::qa What happens when Redis is down?
Sessions live in Redis, so every data route stops working. The rate limiter deliberately fails open in that case, since the session check would fail anyway.
:::

:::qa How would this scale to several API replicas?
Session state is already in Redis, so validation works across replicas. The cleanup loop runs in every process, though. Two replicas could pick up the same expired ids because the range query and the removal are two separate commands. The deletes are idempotent, so the worst case is duplicated work, but a lock or a single scheduler would be cleaner.
:::

:::gaps
##### Trade-offs and gaps

- The browser cookie's lifetime is fixed when `/session/init` is called, while Redis is refreshed on every request. The frontend only calls init on page load, so someone who stays active for longer than the TTL without reloading could lose the cookie client-side.
- The cookie is not signed. Its security rests on the uuid4 being unguessable and the id staying private.
- Log lines include corpus ids, which makes the logs sensitive.
:::

## Edge protection: rate limits and guardrails {#c2}

With no login, the API protects itself in three ways before any LLM is involved: rate limits, a regex input filter, and a regex output filter. A fourth, smaller check validates the planner's JSON.

```diagram
id: d-c2
size: 920 505
alt: Edge protection flow: IP and session rate limits, input guardrail, plan validation, and output guardrail
caption: What happens to a chat request before the agent, and to the agent's output before the user.
---
{
  notes: [
    [20, 22, 'Request side', 17, 'start', '#777'],
    [20, 232, 'Planner output', 17, 'start', '#777'],
    [20, 352, 'Answer side', 17, 'start', '#777']
  ],
  nodes: [
    ['req', 'POST /chat/stream', 20, 50, 140, 60, 'white'],
    ['ip', 'limit_by_ip\nINCR + EXPIRE', 190, 50, 140, 60, 'pink'],
    ['ses', 'limit_by_session\n(cookie must be valid)', 360, 50, 170, 60, 'pink'],
    ['dc', 'over\nlimit?', 560, 35, 100, 90, 'yellow', 'diamond'],
    ['ig', 'check_input\nregex + 4000 char cap', 690, 50, 210, 60, 'pink'],
    ['r429', 'HTTP 429\nRetry-After: N', 520, 150, 140, 50, 'pink'],
    ['blk', 'blocked: persist both\nmessages, fixed reply,\nno LLM, not cached', 690, 140, 210, 75, 'pink'],
    ['pj', 'Planner JSON', 20, 262, 140, 55, 'orange'],
    ['vp', 'validate_plan\naction in 3, tool in 2', 195, 262, 190, 55, 'blue'],
    ['safe', 'safe plan\n(bad value becomes\nretrieve / no tool)', 420, 255, 190, 70, 'green'],
    ['ans', 'Responder answer', 20, 382, 150, 55, 'orange'],
    ['co', 'check_output\n4 leak patterns', 205, 382, 170, 55, 'pink'],
    ['dl', 'leak\nmatched?', 405, 365, 110, 90, 'yellow', 'diamond'],
    ['rep', 'replace with fixed text,\nclear backend and model,\nnever cached', 545, 380, 220, 60, 'pink'],
    ['okk', 'answer\nunchanged', 790, 385, 110, 50, 'green']
  ],
  edges: [
    ['req', 'ip', '', 'r', 'l'], ['ip', 'ses', '', 'r', 'l'], ['ses', 'dc', '', 'r', 'l'],
    ['dc', 'r429', 'yes', 'b', 't', { lx: 600, ly: 135 }],
    ['dc', 'ig', 'no', 'r', 'l'],
    ['ig', 'blk', 'match', 'b', 't', { lx: 770, ly: 128 }],
    ['pj', 'vp', '', 'r', 'l'], ['vp', 'safe', '', 'r', 'l'],
    ['ans', 'co', '', 'r', 'l'], ['co', 'dl', '', 'r', 'l'],
    ['dl', 'rep', 'yes', 'r', 'l'],
    ['dl', 'okk', 'no', 'b', 'b', { via: [[460, 482], [845, 482]] }]
  ]
}
```

##### Flow

1. The chat route declares two rate-limit dependencies: one per client IP, one per session. Each does an atomic `INCR` + `EXPIRE` on a window-scoped Redis key and raises 429 with a `Retry-After` header when the count passes the limit.
2. The route then runs `check_input`: a 4000-character cap plus seven regex patterns for instruction overrides, prompt extraction and persona jailbreaks. A match short-circuits everything: both messages are stored, a fixed refusal is returned, and no cache or LLM is touched.
3. Later, the planner's JSON goes through `validate_plan`, which swaps any out-of-range `action` or `tool_choice` for a safe default.
4. Every answer-producing branch of the responder passes through `guard_output`, which checks for system-prompt leakage and replaces the answer with a fixed message if it finds one.

##### Explanation

**Rate limiting.** Limits are per route and per key type: chat 10 per minute, uploads 5 per hour, session creation 30 per hour, all configurable. Chat and upload are limited by both IP and session. I used both because sessions cost nothing to mint, so a per-session limit alone is trivially bypassed, while an IP-only limit punishes people behind a shared network. Session init is IP-only, because no session exists yet and this is exactly what stops someone from minting sessions to dodge the per-session cap.

```python services/api/app/session/rate_limit.py
key = f"rl:{scope}:{subject}:{int(now // window_seconds)}"
async with redis_client.get_client().pipeline(transaction=True) as pipe:
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    count, _ = await pipe.execute()
```

Putting the window index into the key means each window starts with a fresh counter, so `EXPIRE` can be set unconditionally in the same transaction as `INCR`. There is never a gap where a counter exists without an expiry. This is a fixed-window limiter, which is simple and cheap but allows a burst of up to twice the limit across a window boundary. If Redis is unreachable the limiter logs a warning and lets the request through (it fails open), since sessions need Redis anyway.

The session limiter depends on `get_corpus_id`, so a forged cookie gets a 401 instead of creating a junk counter key. Client IP comes from the socket unless `TRUSTED_PROXY=true`, in which case it takes the *rightmost* `X-Forwarded-For` entry, because everything to its left is client-supplied and spoofable.

**Input guardrail.** The patterns are in `libs/guardrails/patterns.py` and each carries a label so a block can be logged with a reason. They match multi-word phrases specific to an override attempt ("ignore all previous instructions", "reveal your system prompt", "you are now in DAN mode"), never single common words, so ordinary questions are not blocked by accident. A message over 4000 characters is rejected as `input_too_long`. The tests cover both directions: known attacks are blocked and legitimate questions pass.

**Output guardrail.** Four patterns look for the model reciting its own instructions (for example "you are a helpful assistant", or "I was instructed to"). `guard_output` is applied once, at the end of `generate_node`, so every code path is covered. When it fires it replaces the content with `"I can't share that response."`, clears `backend_used` and `model_used`, and because that string is in `REFUSAL_PREFIXES` it is never cached.

**Plan validation.** `validate_plan` never raises. A non-dict becomes an empty dict, an unknown action becomes `retrieve`, and an unknown tool choice becomes `None`. It does not invent a refined query.

:::qa Why regex instead of an LLM-based guardrail?
There is no login to lean on, so I wanted protection that is cheap and predictable. Pattern lists need no model call, so every decision is instant, costs nothing against the LLM quota, and can be read and tested as plain data. The cost is coverage, which I address below. An LLM-based guard would also add latency, spend free-tier quota, and could itself be prompt-injected.
:::

:::qa How would you evade this guardrail, and what would you do about it?
Easily: paraphrase, another language, character substitution, or splitting the attack across messages. The regex layer is a speed bump against copy-pasted jailbreaks, not a security boundary. The real protection is that the system holds nothing sensitive: each tenant sees only its own documents, and the code sandbox is isolated. A next step would be an LLM or classifier check on borderline inputs, and a groundedness check on outputs.
:::

:::qa Why fixed-window and not a token bucket?
A fixed window is one atomic Redis transaction with no Lua script and no read-modify-write race. For an abuse limit on a demo that is enough, and simplicity won. If the limiter ever protected a paid resource, a token bucket would be the next step.
:::

:::gaps
##### Trade-offs and gaps

- The guardrails are pattern-based, so they are easy to bypass. There is no PII filtering and no groundedness check on the output; the evaluation measured faithfulness at 0.705.
- If the planner returns something that is not a dict, `validate_plan` returns `{}`, then the planner node hits a `KeyError` on `reasoning` and lands in its exception handler, which defaults to retrieval. It works, but by accident rather than design.
:::

## The chat request, end to end {#c3}

`POST /api/v1/chat/stream` in `routes/chat.py` is the orchestrator. It does not answer questions itself. It decides whether the answer already exists in the cache, prepares the query, hands it to the agent, and then persists and caches the result.

```diagram
id: d-c3
size: 920 470
alt: Chat request lifecycle: input check, L1 and L2 cache, history, rewrite, HyDE, agent stream, persist, answer, cache write
caption: The full lifecycle of one chat turn, including every early exit.
---
{
  notes: [[20, 20, 'Before the agent', 17, 'start', '#777'], [20, 194, 'Preparing the query and running the agent', 17, 'start', '#777'], [20, 338, 'After the agent', 17, 'start', '#777']],
  nodes: [
    ['n1', 'POST /chat/stream\nrate limits + cookie', 20, 40, 145, 60, 'white'],
    ['n2', 'check_input', 210, 40, 130, 60, 'pink'],
    ['n3', 'L1 exact\nget_exact', 390, 40, 130, 60, 'yellow'],
    ['n4', 'L2 semantic\nget_semantic', 570, 40, 140, 60, 'yellow'],
    ['n5', 'stale entry\nget_all_versions', 760, 40, 140, 60, 'yellow'],
    ['b1', 'blocked:\nfixed reply', 205, 120, 140, 45, 'pink'],
    ['b2', 'hit: answer with\ncache_hit=exact', 390, 120, 150, 45, 'green'],
    ['b3', 'hit: answer with\ncache_hit=semantic', 575, 120, 150, 45, 'green'],
    ['r1', 'last 6 messages\nfrom Postgres', 20, 215, 150, 60, 'blue'],
    ['r2', 'rewrite_query\n(if history exists)', 205, 215, 150, 60, 'purple'],
    ['r3', 'HyDE\n(only if requested)', 390, 215, 150, 60, 'purple'],
    ['r4', 'AgentState\nbuilt', 575, 215, 150, 60, 'white'],
    ['r5', 'agent_app\n.astream()', 760, 215, 140, 60, 'orange'],
    ['s1', 'one StatusEvent\nper finished node', 20, 365, 150, 60, 'white'],
    ['s2', 'responder done:\nsave user + assistant', 205, 365, 150, 60, 'blue'],
    ['s3', 'answer line\n(ChatResponse)', 390, 365, 150, 60, 'green'],
    ['s4', 'set_exact if\ncacheable', 575, 365, 150, 60, 'yellow'],
    ['s5', 'error line:\n"internal error"', 760, 365, 140, 60, 'pink']
  ],
  edges: [
    ['n1', 'n2', '', 'r', 'l'], ['n2', 'n3', 'clean', 'r', 'l'], ['n3', 'n4', 'miss', 'r', 'l'], ['n4', 'n5', 'miss', 'r', 'l'],
    ['n2', 'b1', '', 'b', 't'], ['n3', 'b2', '', 'b', 't'], ['n4', 'b3', '', 'b', 't'],
    ['n5', 'r1', '', 'b', 't', { via: [[830, 172], [95, 172]] }],
    ['r1', 'r2', '', 'r', 'l'], ['r2', 'r3', '', 'r', 'l'], ['r3', 'r4', '', 'r', 'l'], ['r4', 'r5', '', 'r', 'l'],
    ['r5', 's1', '', 'b.2', 't', { via: [[788, 315], [95, 315]] }],
    ['s1', 's2', '', 'r', 'l'], ['s2', 's3', '', 'r', 'l'], ['s3', 's4', '', 'r', 'l'],
    ['r5', 's5', 'exception', 'b.8', 't', { lx: 880, ly: 330 }]
  ]
}
```

##### Flow

1. Rate limits and the cookie check run as dependencies (see the previous section).
2. `check_input` runs. A match stores both messages and returns a fixed refusal.
3. **L1**: look up the exact cache entry for this corpus and this normalized message at the current `corpus_version`. A hit is stored in history and returned with `cache_hit="exact"`.
4. **L2**: if L1 missed, try the semantic cache. A hit returns with `cache_hit="semantic"`.
5. On a full miss, fetch any older cached entry for the same question. It becomes `previous_answer` so the UI can show how the answer changed after new uploads.
6. Load the last 6 chat messages. If there is history and the rewriter is on (the default), rewrite the question into a standalone query. If `use_hyde` is set, replace the query with a hypothetical document.
7. Append the raw user message to the history list, build `AgentState`, and iterate `agent_app.astream()`.
8. Each finished node produces a `status` line. When the responder finishes, the route saves the user and assistant messages and emits the `answer` line.
9. After the stream, the answer is cached if it is eligible. Any exception yields a single `error` line.

##### Explanation

The response is NDJSON: one JSON object per line, with `type` set to `status`, `answer` or `error`. The status lines drive the "Planning… Retrieving… Writing…" label in the UI. It is worth being precise in an interview here: this streams *progress between graph nodes*, not tokens. The answer arrives in one piece when the responder finishes.

The answer line is a `ChatResponse` with the text, `message_id` (the Postgres row id, used later to attach feedback), `cache_hit` (`none`, `exact` or `semantic`), `tool_used`, `backend_used`, `model_used`, `sources`, and an optional `previous_answer` with its corpus version and timestamp.

The order of work is deliberate. Cheap checks come first (regex, then an exact Redis lookup), then the semantic lookup that needs an embedding, and only then the LLM calls. A cached answer costs no LLM tokens at all. This is the "cost-aware" part of the project.

Query preparation has a subtle ordering. The rewriter gets the history *without* the current message, because it is rewriting that message. The raw message is appended to the history list afterwards, so when the planner looks at the last two messages it sees the previous assistant turn followed by the current user turn. That pair is what lets it recognise a bare "yes" as a reply to a web-search offer.

```python services/api/app/routes/chat.py
if final_answer and not _is_refusal(final_answer) and not _is_truncated(final_answer) \
        and final_tool_used not in NO_CACHE_TOOLS:
    await cache.set_exact(corpus_id, req.message, final_answer, final_sources,
                          final_tool_used, final_backend_used, final_model_used)
```

Refusals, answers cut off at the token limit, and web-search answers are never cached. Web results go stale, and a refusal cached today would still be served after the user uploads the document that answers it.

The rest of the app wiring is in `main.py`. Its lifespan function creates the tables, marks any document left in a non-terminal state by a previous crash as failed, connects Redis and both LLM clients, and starts the cleanup task. The Qdrant collection is created lazily on first use, so the API still boots when the vector profile is not running. The feedback route stores a score of 1 or -1 with an optional category and comment. Health has liveness and a readiness check that pings Redis.

:::qa Is the answer actually streamed token by token?
No. The stream carries status events as each graph node finishes, then one final answer line. Token streaming would mean changing the responder to stream from the LLM client and relaying chunks. It is the next step on streaming, but the refusal normalization and the output guardrail need the full answer, so streaming would mean buffering or checking after the fact. The per-node status events were enough to remove the frozen feeling during a response of about six seconds.
:::

:::qa Why check the cache before rewriting the query?
To avoid paying for any LLM call on a repeated question. The trade-off is that the cache key is the raw message, which matters for follow-ups (see below).
:::

:::qa What does the user see after uploading a new document and asking the same question again?
Ingestion bumps `corpus_version`, so the old cache entry no longer matches. The agent runs again, and the earlier answer is attached as `previous_answer`. The UI shows it as "Earlier answer (vN)" under the new one.
:::

:::gaps
##### Trade-offs and gaps

- **Follow-up questions and the cache.** The cache key is the raw message, not the rewritten query. A short follow-up such as "what about its results?" can therefore collide with an earlier identical message about a different topic in the same session.
- **The no-cache list is a string, not a tuple.** `NO_CACHE_TOOLS = ("web_search")` is just the string `"web_search"`, so `not in` is a substring test. Web search is still excluded as intended. Side effects: answers with an empty tool name (greetings) are never cached, and sandbox answers are cached even though the comment says they are not. Writing `("web_search",)` fixes it.
- Persistence happens only when the responder finishes, so a crash mid-run leaves no record of that turn.
- Chat history is stored per session, not per conversation thread.
:::

## The agent: planner, retriever, tool, responder {#c4}

The agent is a LangGraph `StateGraph` with four nodes that share one state object. It is a router-style agent: one planning step picks a path, one path runs, and the responder finishes. It does not loop back to re-plan.

```diagram
id: d-c4a
size: 920 360
alt: Agent routing: initial state to planner, validation, action decision, then retriever, tool or direct answer to the responder
caption: Routing: how a query reaches the retriever, the tool node, or the responder directly.
---
{
  nodes: [
    ['s', 'initial state\nrewritten query,\nhistory, corpus_id', 20, 135, 150, 80, 'white'],
    ['p', 'planner_node\nLLM call, JSON mode,\ntemperature 0', 200, 125, 170, 100, 'orange'],
    ['v', 'validate_plan', 400, 148, 120, 55, 'blue'],
    ['d', 'action', 550, 125, 110, 100, 'yellow', 'diamond'],
    ['retr', 'retriever_node', 720, 30, 150, 55, 'green'],
    ['tool', 'tool_node\nweb_search | sandbox', 720, 148, 150, 55, 'green'],
    ['resp', 'responder\n(generate_node)', 720, 270, 150, 60, 'orange']
  ],
  edges: [
    ['s', 'p', '', 'r', 'l'], ['p', 'v', '', 'r', 'l'], ['v', 'd', '', 'r', 'l'],
    ['d', 'retr', 'retrieve', 't', 'l', { lx: 585, ly: 62 }],
    ['d', 'tool', 'tool_use', 'r', 'l', { lx: 690, ly: 165 }],
    ['d', 'resp', 'direct_answer', 'b', 'l', { lx: 560, ly: 300 }],
    ['retr', 'resp', '', 'r', 'r', { via: [[895, 57], [895, 300]] }],
    ['tool', 'resp', '', 'r', 'r', { via: [[895, 175], [895, 300]] }]
  ],
  notes: [
    [285, 270, 'on any planner error: default to retrieve', 15],
    [285, 292, 'history sent to planner: last 2 messages', 15]
  ]
}
```

```diagram
id: d-c4b
size: 920 370
alt: Responder logic: branch by action, zero-hit handling, RAG answer with refusal normalization, sandbox echo, conversational answer, output guardrail
caption: Inside the responder: three branches, then the output guardrail.
---
{
  nodes: [
    ['g', 'generate_node', 20, 150, 130, 60, 'orange'],
    ['dA', 'action', 175, 130, 110, 100, 'yellow', 'diamond'],
    ['dB', 'documents\nretrieved?', 325, 25, 135, 100, 'yellow', 'diamond'],
    ['zero', 'none: existence check?\nyes: "No, not mentioned"\nno: offer web search', 495, 10, 210, 70, 'pink'],
    ['llm1', 'RAG prompt, temp 0.3\nrouted model, max 2048', 495, 100, 210, 55, 'blue'],
    ['chk', 'is_refusal?\nyes: offer or "No"\nno: answer + cut-off flag', 735, 90, 170, 75, 'yellow'],
    ['sb', 'sandbox: echo stdout\nin a code block, no LLM', 325, 175, 210, 55, 'green'],
    ['conv', 'conversational LLM\nlast 4 messages, temp 0.5', 325, 270, 210, 60, 'blue'],
    ['gd', 'guard_output\nleak patterns, fixed text', 735, 250, 170, 65, 'pink']
  ],
  edges: [
    ['g', 'dA', '', 'r', 'l'],
    ['dA', 'dB', 'retrieve', 't', 'l', { via: [[230, 75]], lx: 255, ly: 62 }],
    ['dA', 'sb', 'sandbox', 'r', 'l', { lx: 300, ly: 167 }],
    ['dA', 'conv', 'otherwise', 'b', 'l', { via: [[230, 300]], lx: 262, ly: 318 }],
    ['dB', 'zero', 'no', 'r', 'l'],
    ['dB', 'llm1', 'yes', 'b', 'l', { via: [[392, 127]], lx: 420, ly: 118 }],
    ['llm1', 'chk', '', 'r', 'l'],
    ['chk', 'gd', '', 'b', 't'],
    ['zero', 'gd', '', 'r', 'l.2', { via: [[720, 45], [720, 263]] }],
    ['sb', 'gd', '', 'r', 'l.5'],
    ['conv', 'gd', '', 'r', 'l.8']
  ]
}
```

##### Flow

1. The graph's entry point is the planner. It receives the (possibly rewritten) query and the last two messages and returns JSON with `action`, `refined_query`, `reasoning`, `tool_choice`, `tool_input` and `is_existence_check`.
2. `validate_plan` sanitizes it. A conditional edge routes on `action`: `retrieve` goes to the retriever, `tool_use` to the tool node, `direct_answer` straight to the responder.
3. The retriever and the tool node both feed the responder, which feeds `END`.
4. The responder branches by action. A retrieval turn builds a grounded RAG prompt, or returns a deterministic non-answer if nothing was found. A sandbox turn echoes the program's output with no LLM call. Anything else gets a plain conversational answer.
5. The answer passes through `guard_output` and goes back to the chat route.

##### Explanation

```python services/api/app/agents/graph.py
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_from_planner,
    {"retriever": "retriever", "responder": "responder", "tool": "tool"})
workflow.add_edge("tool", "responder")
workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
agent_app = workflow.compile()
```

**State.** `AgentState` is a TypedDict carrying the messages (with an `operator.add` reducer so nodes append rather than overwrite), the retrieved documents, the current query, the planner's decision, the tenant's `corpus_id`, and bookkeeping that ends up in the response: `tool_used`, `backend_used`, `model_used`, `sources`, plus `tool_result` and `is_existence_check`.

**Planner.** Its prompt makes three decisions that came from watching real behaviour. First, any question seeking a fact, even one the corpus will not cover, like the weather, must go to `retrieve`. The prompt tells the planner not to reason about whether the corpus probably has the answer. Retrieval finding nothing is what correctly triggers the web-search offer. If the planner answered directly instead, the model would improvise something like "try a weather app", which is not the intended flow. Second, web search is never chosen on the planner's own initiative: only when the most recent assistant message offered to search *and* the user's reply is a clear yes, in which case `tool_input` is the topic read out of the offer. Third, it flags yes/no questions about document content ("did the paper use ResNet?") as `is_existence_check`, which changes how an empty result is worded.

The planner runs at temperature 0 in JSON mode with the model chosen by the tier router. Only tool turns adopt its `refined_query`; retrieval and direct turns keep the coreference-resolved query, which scored better against the index. If the call or the parsing fails for any reason, the node logs a warning and defaults to retrieval.

**Retriever and tool node.** The retriever is covered in its own section. The tool node dispatches on `tool_choice` to web search or the sandbox (see the tools section), and returns the result both as a message and as `tool_result`.

**Responder, retrieval branch.** Retrieved chunks are joined into one context string and sent with a system prompt that demands citations in the form `[Source: Filename]` and an exact refusal sentence when the answer is not in the context. If retrieval returned nothing, no LLM is called at all: an existence check gets the fixed answer "No — that isn't mentioned in the documents.", and anything else gets the standard offer to search the web. If retrieval returned something irrelevant and the model refuses anyway, `is_refusal` catches it and normalizes it to the same two outputs, so the planner only ever has to recognise one offer format. Existence-check negatives are real answers and may be cached, while the web-search offer is a refusal and is not. If the model hit its token cap, a visible note is appended.

**Model tiers.** `route_model` picks `MODEL_TIER_SIMPLE` (default `openai/gpt-oss-20b`) or `MODEL_TIER_COMPLEX` (`openai/gpt-oss-120b`). It picks the complex tier when query plus context passes 1500 characters or the query contains words like "compare", "why", "analyze" or "trade-off". The rule is fixed in code and the models are environment variables, so swapping models needs no code change. The planner, the rewriter and the responder all use it.

**Query enhancers.** The rewriter prompts the LLM with the history and the new question and asks for a short standalone search query (temperature 0, 64 tokens). If the result is over 200 characters, which usually means the model answered instead of rewriting, or if the call fails, the original question is used. HyDE (hypothetical document embeddings) asks the model to write a plausible answer paragraph to embed instead of the question. It is off by default and opt-in per request.

:::qa Is this really "agentic"?
It is a routing agent: an LLM decides between retrieving, using a tool, or answering directly, and the graph executes that choice. It does not loop, so it cannot retrieve, notice the results are poor, and try again. That is a deliberate simplicity choice: one planning step keeps the flow easy to reason about and cheap on free-tier quotas. If retrieval quality ever needs it, a grader after retrieval, capped at two passes, is the natural addition.
:::

:::qa Why does the planner send "what's the weather?" to retrieval?
Because a no-hit retrieval is the trigger for the web-search offer. Letting the planner skip retrieval would let the model answer from its own head or improvise. The cost is one wasted retrieval on questions the corpus will never answer.
:::

:::qa Why is web search behind a confirmation?
It sends the user's query to an external service and returns content that is not from their documents. Making it an explicit, user-approved step keeps that visible, and the next turn's planner only treats a clear yes to the offer as permission.
:::

:::qa Why does the sandbox output skip the LLM?
So a model cannot mis-transcribe stdout or accidentally emit its own tool-call format while narrating tool output. The result is echoed deterministically in a code block.
:::

:::gaps
##### Trade-offs and gaps

- No relevance threshold sits before generation. A weak but non-empty retrieval still reaches the LLM and is caught afterwards by the refusal check.
- With HyDE on, the hypothetical paragraph replaces `current_query`, so the planner and the responder prompt also see that paragraph as "the question". It helps the embedding match but muddies the prompts. This is why it is off by default.
- The tool node has a `vector_search` branch, but `validate_plan` only allows `web_search` and `sandbox`, so the planner can never select it. Retrieval already has its own node.
:::

## Hybrid retrieval and reranking {#c5}

Retrieval combines meaning-based search and keyword search, fuses the two rankings, and then reorders the best candidates with a model that reads the query and each passage together.

```diagram
id: d-c5
size: 920 350
alt: Hybrid retrieval: dense and sparse embedding in parallel, two Qdrant prefetches, RRF fusion with tenant filter, cross-encoder rerank, top five
caption: From a query string to the five chunks the responder sees.
---
{
  nodes: [
    ['q', 'query\n(after rewrite)', 20, 140, 120, 60, 'white'],
    ['de', 'dense embedding\nbge-large-en-v1.5\n1024 dimensions', 175, 40, 185, 70, 'green'],
    ['se', 'sparse embedding\nQdrant/bm25\nindices + values', 175, 230, 185, 70, 'green'],
    ['pd', 'Prefetch: dense\ntop 10', 400, 50, 140, 55, 'grey'],
    ['ps', 'Prefetch: sparse\ntop 10', 400, 238, 140, 55, 'grey'],
    ['rrf', 'RRF fusion\nfilter corpus_id\nlimit 10', 575, 130, 150, 80, 'yellow'],
    ['none', 'zero hits:\ndocuments = []', 575, 255, 150, 55, 'pink'],
    ['rr', 'cross-encoder\nbge-reranker-base\nscores all 10', 755, 40, 145, 75, 'purple'],
    ['top', 'keep top 5\n"text [Source: file]"', 755, 215, 145, 70, 'green']
  ],
  edges: [
    ['q', 'de', 'in parallel', 'r', 'l', { lx: 148, ly: 62 }],
    ['q', 'se', '', 'r', 'l'],
    ['de', 'pd', '', 'r', 'l'], ['se', 'ps', '', 'r', 'l'],
    ['pd', 'rrf', '', 'r', 't.3'], ['ps', 'rrf', '', 'r', 'b.3'],
    ['rrf', 'rr', '', 'r', 'l', { lx: 740, ly: 150 }],
    ['rrf', 'none', '', 'b', 't'],
    ['rr', 'top', '', 'b', 't']
  ]
}
```

##### Flow

1. The query is embedded twice at the same time: a dense vector with bge-large-en-v1.5 and a sparse BM25 vector with `Qdrant/bm25`, both on CPU in worker threads via `asyncio.gather`.
2. One Qdrant query runs two prefetches (top 10 dense, top 10 sparse) and fuses them with Reciprocal Rank Fusion on the server, restricted to the caller's `corpus_id`, returning 10 candidates.
3. No hits means an empty document list and the responder takes its zero-hit path.
4. Otherwise the cross-encoder scores each of the 10 passages against the query, they are sorted by score, and the top 5 (`RERANK_TOP_N`) are kept.
5. Each kept chunk is formatted as `"text [Source: filename]"`, and the filenames are parsed back out of that suffix into the `sources` list.

##### Explanation

The collection holds two named vectors per point: `dense` (1024 dimensions, cosine) and `sparse`. The same schema is declared in two places, the async API client and the synchronous ingestion indexer, and each carries a comment saying they must match exactly, because a mismatch would silently break search instead of raising an error.

```python services/api/app/clients/qdrant.py
response = await self.client.query_points(
    collection_name=settings.QDRANT_COLLECTION,
    prefetch=[
        models.Prefetch(query=dense_vector, using="dense", limit=limit),
        models.Prefetch(query=models.SparseVector(indices=..., values=...),
                        using="sparse", limit=limit),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    query_filter=models.Filter(must=[models.FieldCondition(
        key="corpus_id", match=models.MatchValue(value=corpus_id))]),
    limit=limit, with_payload=True)
```

**Why both kinds of search.** Dense vectors catch paraphrases but can blur exact tokens like model names, numbers and acronyms. BM25 is the opposite. The evaluation backs this up on research papers: BM25 alone had a better MRR than dense alone (0.516 vs 0.421), and fusing them beat both on Hit@5 (0.619 vs 0.571). The cross-encoder then lifted Hit@5 to 0.667 and MRR to 0.563. A cross-encoder reads the query and the passage together, which is more accurate than comparing two independent vectors, but too slow to run over the whole corpus, so it only reorders the ten fused candidates. Keep the sample size in mind: with 21 answerable questions, one question moves a rate by about 5 points, so these are directional.

**Fusion setting.** I tested the fusion constant and ended up using Qdrant's default, which is `k = 2` in Qdrant's formula (the value 60 from the original RRF paper corresponds to `k = 61` there). The ablation shows hybrid fusion with the default already beats both single retrievers, so I did not tune it further. The `RRF_K` setting (60) is kept in the config for future cases, such as a larger corpus or a different question mix, but it is not passed to Qdrant today.

The embedding work is shared. The same `FastEmbedClient` embeds queries at chat time and documents during ingestion, so the two paths cannot drift apart. The models load lazily on first use. FastEmbed's calls are synchronous, so every caller wraps them in `asyncio.to_thread` to keep the event loop free. The reranker has an optional failover to a prompted-LLM reranker through OpenRouter, switched on by the same `ENABLE_OPENROUTER_FALLBACK` flag.

The embedder is intentionally the only one, with no failover. The collection is fixed at one vector size, and a fallback with a different dimension would write incompatible vectors. There is an OpenRouter embedder in the file, clearly marked as an unused template for a future same-dimension provider.

:::qa What is Reciprocal Rank Fusion?
Each ranked list gives every item a score of 1 / (k + rank), and the scores are summed across lists. Items that rank well in both lists rise to the top, and it needs no score normalisation between BM25 and cosine similarity, which live on different scales.
:::

:::qa Why retrieve 10 and keep 5?
Ten gives the reranker room to promote a relevant passage that fusion placed lower, and five keeps the prompt small. The eval notes that `RERANK_TOP_N=5` was not compared with a higher value, so it is unverified whether misses sit just outside the top five. Five chunks of up to 512 tokens is roughly 2,500 tokens of context, which is cheap and fits small models. It is a starting value, not a tuned one.
:::

:::qa Where does retrieval fail?
By category (Hit@5): single-fact 0.857, multi-passage 0.8, table lookup 0.5, figure caption 0.5, cross-paper 0.333. Two false refusals in the spot check were table facts that were in the corpus. The weak spots are tables and questions that need several papers at once.
:::

:::gaps
##### Trade-offs and gaps

- The `corpus_id` filter is set on the outer query only. Putting the same filter inside each `Prefetch` would scope the candidate lists to the tenant before fusion.
- The reranker sees the passage text including the appended `[Source: …]` suffix, which is added before reranking.
:::

## The LLM layer {#c6}

Every LLM call in the app, whether from the planner, rewriter, responder, HyDE, figure captioning or the eval judges, goes through one engine class. It is built to survive free-tier limits: it tracks quotas locally, skips models that are out of budget, retries transient errors, and stops hammering a backend that keeps failing.

```diagram
id: d-c6
size: 920 520
alt: LLM client flow: circuit breaker, model ordering, rate-limit check, POST with retry, success or next model, exhaustion and failover
caption: What `chat_completion()` does with one request.
---
{
  nodes: [
    ['call', 'chat_completion\nmessages, model?,\njson_mode', 20, 40, 165, 75, 'white'],
    ['cb', 'breaker\nopen?', 215, 30, 110, 95, 'yellow', 'diamond'],
    ['ord', 'build model order:\npin first, round robin,\nor priority list', 355, 40, 195, 75, 'blue'],
    ['tr', 'rate tracker\nallows?', 580, 25, 140, 105, 'yellow', 'diamond'],
    ['post', 'POST /chat/completions\nretry 2x with\nbackoff + jitter', 750, 40, 150, 75, 'green'],
    ['next', 'try next model\nin the list', 580, 215, 150, 55, 'grey'],
    ['ok', 'status ok and\nnon-empty text?', 750, 180, 150, 100, 'yellow', 'diamond'],
    ['succ', 'record headers, reset\nbreaker, keep model +\nfinish_reason, return', 745, 340, 160, 75, 'green'],
    ['exh', 'list exhausted:\nrecord_failure(),\nModelExhaustedError', 330, 340, 230, 75, 'pink'],
    ['fo', 'FailoverLLMClient (if enabled)\ncalls the backup backend', 330, 450, 230, 55, 'blue']
  ],
  edges: [
    ['call', 'cb', '', 'r', 'l'], ['cb', 'ord', 'closed', 'r', 'l', { lx: 340, ly: 55 }],
    ['ord', 'tr', '', 'r', 'l'], ['tr', 'post', 'yes', 'r', 'l', { lx: 735, ly: 60 }],
    ['tr', 'next', 'no', 'b', 't'],
    ['post', 'ok', '', 'b', 't'],
    ['ok', 'next', 'no', 'l', 'r'],
    ['ok', 'succ', 'yes', 'b', 't'],
    ['next', 'ord', 'loop', 'l', 'b', { via: [[455, 242]], lx: 440, ly: 180 }],
    ['next', 'exh', 'none left', 'b', 'r', { lx: 660, ly: 330 }],
    ['cb', 'exh', 'open', 'b', 'l', { via: [[270, 378]], lx: 258, ly: 250 }],
    ['exh', 'fo', 'caught', 'b', 't']
  ]
}
```

##### Flow

1. Check the circuit breaker. If it is open, raise `ModelExhaustedError` immediately.
2. Decide the order to try models in: a pinned model goes first with the rest behind it, otherwise Groq rotates round-robin and the others use the configured priority order.
3. For each model, ask its rate-limit tracker whether the request fits. If not, skip it without calling the API.
4. POST the request. The HTTP call retries twice with exponential backoff and jitter.
5. An HTTP error, a transport error, a malformed body or an empty answer moves on to the next model. A good answer records the provider's rate-limit headers, resets the breaker, and remembers which model answered and why it stopped.
6. If every model fails or is skipped, record a breaker failure and raise `ModelExhaustedError`. A failover wrapper, when enabled, catches it and calls the backup backend.

##### Explanation

The design is one abstract `LLMClient` (`start`, `close`, `chat_completion`) and one engine, `OpenAICompatibleClient`, that implements the OpenAI `/chat/completions` schema. Groq, OpenRouter, Ollama, vLLM and Mistral are thin subclasses that only supply a base URL, a model list and a key. Callers depend on the interface and never on a vendor. `factory.py` builds the active client from `LLM_BACKEND`: `api` for Groq or OpenRouter, or `ollama`, `vllm_local` and `vllm_modal`, which are manual-select only because cold starts make them poor failover targets.

**Model lists.** Each backend has a priority-ordered list from the environment (comma-separated, parsed by a validator). Groq's three models each have their own quota, so Groq uses round-robin dispatch to spread load. OpenRouter has one shared account-level quota, so it uses plain priority order.

**Rate limiting before the call.** `rate_limiter.py` tracks requests per minute and per day and tokens per minute and per day for each model, estimating tokens as characters divided by four. It checks *before* sending, instead of waiting for a 429. When a provider returns its own rate-limit headers (Groq on every response, OpenRouter on errors), those live values override the local estimate until the window resets, since they are more accurate. A model that is out of budget is skipped rather than waited on. The eval notes spell out the consequence: that works with a fallback model but would null every remaining question with a single-model judge, so judge calls sleep a fixed interval first.

```python libs/utils/circuit_breaker.py
def record_failure(self):
    self._consecutive_failures += 1
    if self._consecutive_failures >= self.failure_threshold:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
```

**Circuit breaker.** One breaker per backend, with a threshold of 3 and a 30-second cooldown. A failure is only recorded when the *whole* model list is exhausted, not when one model fails. While open, calls fail instantly. After the cooldown it goes half-open and lets one trial call through.

**Backoff.** `@exponential_backoff(max_retries=2)` wraps the HTTP post. Delays are 1 s then 2 s (capped at 10 s) plus up to half a second of random jitter, so concurrent requests do not retry in lockstep. It retries any `httpx.HTTPError`, which includes the status errors the client raises itself for 4xx and 5xx responses.

**Failover.** With `ENABLE_OPENROUTER_FALLBACK` on, `FailoverLLMClient` wraps a primary and a backup and switches only when the primary raises `ModelExhaustedError`. The backup call does not carry the model pin, because model names differ between vendors. It is off by default, so with it off an exhausted Groq raises up to the chat route, which returns a generic error line.

**Truncation.** The client records the `finish_reason`. When it is `length`, the responder appends a visible "answer cut off" note and the chat route refuses to cache that answer.

The base interface asks callers to put stable content (the system prompt) first and variable content (context, question) last, so provider-side prompt caching can work.

:::qa Why a circuit breaker when you already fail over between models?
Model failover handles one bad model. The breaker handles a bad *backend*: after three fully exhausted calls it stops spending time and quota walking the entire list on every request, and probes again after 30 seconds.
:::

:::qa How do you handle the free-tier limits?
Three layers: local per-model counters checked before each call, live header values overriding them, and skip-to-next-model on any failure. Groq's three models have separate quotas, so together they give more headroom than any one.
:::

:::qa How does the model tier routing work?
Simple queries go to `gpt-oss-20b` and long or comparative ones to `gpt-oss-120b`. It is a plain rule on query length and keywords, because most questions are simple lookups and the large model has tighter quotas. I kept it a heuristic on purpose, and it can be tuned when cost or quota shows a need. The eval can compare routing on and off.
:::

:::gaps
##### Trade-offs and gaps

- Model routing always supplies a model, so Groq's round-robin only applies to calls that pass none (such as HyDE and the closed-book eval run). In the main flow, the routed model is tried first and the rest follow in order.
- `last_model_used` and `last_finish_reason` are attributes on a shared client object, so under concurrent requests one request can read another's values. The answer text is unaffected, but the model name and the truncation flag shown to the user could be wrong.
- Ollama and vLLM backends are implemented behind the same interface but have not been run end to end.
:::

## Two-layer response cache {#c7}

The cache saves LLM cost on repeated and near-repeated questions without ever serving an answer that was written for a different set of documents.

```diagram
id: d-c7
size: 920 480
alt: Cache: L1 exact lookup, L2 semantic lookup with three gates, miss, cache write rules, and corpus_version invalidation
caption: L1, L2 and the write path, with the version check that keeps entries fresh.
---
{
  notes: [[20, 20, 'L1: exact', 17, 'start', '#777'], [20, 160, 'L2: semantic (only if L1 missed)', 17, 'start', '#777'], [540, 378, 'Invalidation after a new upload', 17, 'start', '#777']],
  nodes: [
    ['q', 'query', 20, 45, 100, 50, 'white'],
    ['nz', 'normalize:\nlowercase, spaces', 150, 45, 150, 50, 'blue'],
    ['key', 'key = cache:corpus:\nschema:sha256', 330, 45, 175, 50, 'blue'],
    ['hg', 'HGET entries', 535, 45, 120, 50, 'blue'],
    ['d1', 'entry at\ncurrent version?', 685, 22, 130, 95, 'yellow', 'diamond'],
    ['h1', 'L1 HIT', 835, 47, 70, 45, 'green'],
    ['e1', 'embed query,\nKNN 1 in same\ncorpus (HNSW)', 20, 195, 150, 70, 'blue'],
    ['d2', 'cosine\nsim ≥ 0.85', 195, 185, 120, 90, 'yellow', 'diamond'],
    ['d3', 'Jaccard\n≥ 0.5', 340, 185, 110, 90, 'yellow', 'diamond'],
    ['d4', 'length ratio\n≥ 0.8', 475, 185, 125, 90, 'yellow', 'diamond'],
    ['d5', 'version\nmatches', 625, 185, 115, 90, 'yellow', 'diamond'],
    ['h2', 'L2 HIT', 770, 207, 90, 45, 'green'],
    ['miss', 'MISS: run the agent', 195, 310, 545, 38, 'pink'],
    ['wa', 'answer', 20, 395, 100, 50, 'orange'],
    ['wb', 'cacheable?', 145, 380, 140, 80, 'yellow', 'diamond'],
    ['wc', 'set_exact: newest first,\nkeep 2 versions,\nEXPIRE 24 h', 310, 390, 190, 65, 'yellow'],
    ['ia', 'ingestion done:\nINCR corpus_version', 540, 390, 170, 65, 'green'],
    ['ib', 'old entries stop\nmatching, kept as\nprevious_answer', 740, 385, 160, 75, 'grey']
  ],
  edges: [
    ['q', 'nz', '', 'r', 'l'], ['nz', 'key', '', 'r', 'l'], ['key', 'hg', '', 'r', 'l'], ['hg', 'd1', '', 'r', 'l'],
    ['d1', 'h1', 'yes', 'r', 'l'],
    ['d1', 'e1', 'no', 'b', 't', { via: [[750, 145], [95, 145]], lx: 760, ly: 135 }],
    ['e1', 'd2', '', 'r', 'l'], ['d2', 'd3', 'yes', 'r', 'l'], ['d3', 'd4', 'yes', 'r', 'l'], ['d4', 'd5', 'yes', 'r', 'l'],
    ['d5', 'h2', 'yes', 'r', 'l'],
    ['d2', 'miss', '', 'b', 't.05'], ['d3', 'miss', '', 'b', 't.35'],
    ['d4', 'miss', 'any "no"', 'b', 't.65', { lx: 600, ly: 292 }], ['d5', 'miss', '', 'b', 't.95'],
    ['miss', 'wa', '', 'b.05', 't', { via: [[222, 362], [70, 362]] }],
    ['wa', 'wb', '', 'r', 'l'], ['wb', 'wc', 'yes', 'r', 'l'],
    ['ia', 'ib', '', 'r', 'l']
  ]
}
```

##### Flow

1. **L1.** The query is lowercased and whitespace-collapsed, hashed with SHA-256, and combined with the corpus id and a schema version into a Redis key. The key holds a list of entries, each stamped with the `corpus_version` it was written at. A hit needs an entry at the *current* version.
2. **L2.** On an L1 miss, embed the query and run a vector KNN (top 1) in RediSearch restricted to the same corpus. The candidate must clear three gates: cosine similarity at least 0.85, token overlap (Jaccard) at least 0.5, and a shorter-to-longer token-count ratio of at least 0.8. Then its version must match the current one.
3. **Miss.** Run the agent.
4. **Write.** If the answer is not a refusal, not truncated and not from web search, store a new entry. Same-version entries are replaced, the newest goes first, and only 2 versions are kept. The key expires after 24 hours.
5. **Invalidation.** When ingestion finishes, it increments `corpus_version:{corpus_id}`. Old entries stay but stop matching, which is what powers the "earlier answer" comparison.

##### Explanation

Each L1 key is a Redis hash with an `entries` field (JSON list of up to two versions: answer, sources, tool, backend, model, version, timestamp), plus `corpus_id` and the normalized `query` text, plus, for retrieval answers only, a packed float32 `vector`. That last field is why L2 only works for answers produced by vector search: only those are written with a vector.

```python services/api/app/cache/redis_cache.py
VectorField("vector", "HNSW",
            {"TYPE": "FLOAT32", "DIM": VECTOR_DIM, "DISTANCE_METRIC": "COSINE"})
...
search_query = (Query(f"(@corpus_id:{{{escaped}}})=>[KNN 1 @vector $vec AS score]")
                .sort_by("score").return_fields("score").dialect(2))
similarity = 1.0 - float(top.score)   # RediSearch returns cosine distance
```

The vector index is created lazily and idempotently on first use and covers every key with the `cache:` prefix. The tenant is a TAG field, so a semantic lookup can never match another session's answer. The semantic cache uses its own Redis connection with `decode_responses=False`, because raw binary vectors cannot go through a text-decoding client.

Why three gates? Embedding similarity alone is risky: two questions can be close in vector space and still ask different things. The lexical-overlap gate is a second opinion, and the length-ratio gate catches the case where one question is a compound and the other is just one of its sub-clauses, a mistake the other two gates miss. The thresholds live in `.env` so they can be tuned without code changes.

Two more details. Entries that exist from before refusals were excluded from caching are treated as misses when read. And cascade delete at session expiry removes cache keys with a non-blocking `SCAN`.

:::qa How do you avoid serving a stale answer after the user uploads another document?
The version stamp. Ingestion increments `corpus_version`, and both cache layers only accept an entry whose version equals the current one. A global schema version, `CACHE_SCHEMA_VERSION`, is part of every key as a manual way to invalidate everything after a logic change, and the 24-hour TTL is only a backstop.
:::

:::qa What are the risks of semantic caching?
False hits: a near-duplicate question with a different intent gets the wrong cached answer. The three gates and the version check reduce that, and L2 is limited to retrieval answers. A false hit is worse than a miss, because a miss costs one LLM call and a false hit returns a wrong answer, so I made all three gates strict and kept the logic simple. The thresholds live in the environment, so they can be tuned later from sampled logs if the hit rate shows a need.
:::

:::gaps
##### Trade-offs and gaps

- The cache key ignores conversation context (see the chat section), so short follow-ups can collide.
- Every L1 miss pays for one embedding to attempt L2, before any LLM call.
- `CACHE_SCHEMA_VERSION` is bumped by hand, not tied to deploys.
:::

## Upload and ingestion pipeline {#c8}

Getting a document from the browser into a searchable index. The file never passes through the API server, and ingestion runs in the same process as a background task.

```diagram
id: d-c8
size: 920 470
alt: Upload and ingestion: presigned URL, direct PUT to MinIO, webhook, background run through fetch, parse, embed, index, version bump and completion
caption: From the file picker to a complete document, with the status the UI sees at each stage.
---
{
  notes: [
    [20, 22, 'Upload', 17, 'start', '#777'],
    [20, 160, 'Ingestion (BackgroundTask in the API process)', 17, 'start', '#777'],
    [460, 440, 'status: pending, fetching, parsing, embedding, indexing, complete or failed', 15]
  ],
  nodes: [
    ['u1', 'Browser picks\na file', 20, 45, 130, 60, 'white'],
    ['u2', 'POST generate-\npresigned-url\nsize check, 20 MB', 175, 40, 175, 70, 'blue'],
    ['u3', 'key uploads/corpus/\nfile_id.ext\nDocument: pending', 375, 40, 175, 70, 'green'],
    ['u4', 'PUT straight to\nMinIO (signed\nmetadata headers)', 575, 40, 160, 70, 'white'],
    ['u5', 'bucket event\nto webhook', 760, 45, 140, 60, 'grey'],
    ['w1', 'POST /webhooks/minio\nBackgroundTask per\nrecord', 20, 200, 170, 70, 'blue'],
    ['r1', 'fetching\nhead_object size\ncheck, then read', 215, 200, 150, 70, 'green'],
    ['r2', 'parsing\nDocling convert,\nchunk, captions', 390, 200, 150, 70, 'green'],
    ['r3', 'embedding\ndense + sparse', 565, 205, 150, 60, 'green'],
    ['r4', 'indexing\nQdrant upsert', 740, 205, 160, 60, 'green'],
    ['c1', 'bump corpus_version\n(Redis INCR)', 560, 330, 170, 55, 'yellow'],
    ['c2', 'set_complete\nchunk_count', 755, 330, 145, 55, 'green'],
    ['f', 'any exception:\nstatus = failed + error', 215, 335, 210, 55, 'pink']
  ],
  edges: [
    ['u1', 'u2', '', 'r', 'l'], ['u2', 'u3', '', 'r', 'l'], ['u3', 'u4', '', 'r', 'l'], ['u4', 'u5', '', 'r', 'l'],
    ['u5', 'w1', '', 'b', 't', { via: [[830, 140], [105, 140]] }],
    ['w1', 'r1', '', 'r', 'l'], ['r1', 'r2', '', 'r', 'l'], ['r2', 'r3', '', 'r', 'l'], ['r3', 'r4', '', 'r', 'l'],
    ['r4', 'c1', '', 'b', 't', { via: [[820, 300], [645, 300]] }],
    ['c1', 'c2', '', 'r', 'l'],
    ['r2', 'f', 'on error', 'b', 't', { dashed: true, lx: 400, ly: 305 }]
  ]
}
```

##### Flow

1. The browser asks `POST /api/v1/upload/generate-presigned-url` with filename, content type and size. The API rejects a declared size over 20 MB, builds the key `uploads/{corpus_id}/{file_id}.{ext}`, creates a `pending` row in the documents table, and returns a presigned PUT URL.
2. The browser PUTs the file directly to MinIO with the signed headers.
3. MinIO sends a bucket-notification webhook to `POST /api/v1/webhooks/minio`, which queues `run_ingestion(bucket, key)` as a background task for each record.
4. **fetching:** check the stored object's real size against the cap (deleting it if too large) and read the bytes.
5. **parsing:** Docling converts and chunks the file, figures are captioned, and metadata is added.
6. **embedding:** compute dense and sparse vectors for all chunks.
7. **indexing:** upsert the points into Qdrant, tagged with the corpus id.
8. Increment `corpus_version`, mark the document `complete` with its chunk count. Any exception marks it `failed` with the error text.
9. Meanwhile the browser polls `GET /api/v1/ingest/status/{file_id}` every 1.5 seconds and refreshes the document list.

##### Explanation

**Presigned upload.** The API hands out a short-lived (one hour) PUT URL so file bytes go straight to object storage. The URL is signed with the content type and two metadata fields, `original_filename` and `corpus_id`, so the browser must send matching `x-amz-meta-*` headers or MinIO rejects the PUT. Because the session cookie is httpOnly and invisible to JavaScript, the response includes the `corpus_id` for that header. The pending row is created at issuance, not on webhook receipt, so the UI can show the file immediately. boto3 is synchronous, but presigning is quick, so it runs in `asyncio.to_thread` instead of pulling in an async S3 SDK.

**Two size checks.** The declared size is client-supplied, so it is only a fast failure. The authoritative check is in `_fetch_object`, which reads the stored object's real `ContentLength` and deletes the object if it is over the cap.

**Where the tenant id comes from.** `run_ingestion` derives `corpus_id` from the S3 key layout and `file_id` from the key's file stem. Nothing the client sends in the webhook is trusted. It then looks up the real filename in the documents table, so the citation shown to the user is the original filename and not a UUID.

```python pipelines/ingestion/pipeline.py
await _set_status("fetching");  content = await _fetch_object(bucket, file_key)
await _set_status("parsing");   chunks  = await _parse_and_chunk(content, filename, corpus_id)
await _set_status("embedding"); dense, sparse = await _embed(texts)
await _set_status("indexing");  await asyncio.to_thread(_index_qdrant, chunks, dense, sparse, corpus_id)
await asyncio.to_thread(bump_corpus_version, corpus_id)
await document_store.set_complete(file_id, chunk_count=len(chunks))
```

Status writes are best-effort: if the database hiccups, ingestion keeps going and the status just lags until the next stage. All CPU-bound work (Docling inference, embedding, sparse encoding) runs through `asyncio.to_thread`, because ingestion shares an event loop with live chat requests.

**Indexing.** `QdrantIndexer` uses a synchronous Qdrant client, created once per process. It takes a column-oriented batch (a dict of lists) and builds one point per chunk with a random UUID, both named vectors, and a payload of `text`, `filename`, `page`, `section` and `corpus_id`. The comment on the payload says it plainly: every query-time search filters on `corpus_id`, so a wrong value here would be a cross-tenant leak.

**Crash recovery.** Because ingestion is an in-process background task, a restart kills any run in progress. On startup the app marks every document still in a non-terminal state as `failed` with the message "Orphaned by server restart", so nothing hangs forever.

The evaluation harness imports `run_ingestion` directly, so the numbers in the eval section were produced by the real ingestion code, just without the webhook in front of it.

:::qa Why a background task instead of a queue like Celery?
It keeps the deployment to one process and nothing to run besides the four stores, which fits a free-tier demo. Ingestion is short, bounded by the 20 MB cap and scoped to one user, so a single process with no broker is enough. The price is durability: a restart loses the in-flight job, and the startup sweep marks it failed instead of retrying. `run_ingestion(bucket, key)` is already a standalone function, so a queue can be added when load requires it.
:::

:::qa Why presigned URLs?
File bytes never touch the API process, so a large upload cannot tie up the server, and the app does not need to buffer or stream the body itself.
:::

:::qa What if the same file is uploaded twice?
It is indexed twice. Point ids are random UUIDs and nothing deduplicates. A content hash is computed per chunk in `enrich_metadata` but it is not stored in the Qdrant payload or used for dedupe.
:::

:::gaps
##### Trade-offs and gaps

- The webhook endpoint has no signature or secret check, so anyone who can reach it and knows a valid key layout could trigger ingestion.
- The presign step does not restrict file type, so an unsupported file fails later inside Docling and ends up as `failed`.
- Each file is read fully into memory and embedded in one batch.
- A document is only searchable once it is fully indexed, so there is no partial availability.
:::

## Parsing and chunking with Docling {#c9}

This is the step that decides what the index can ever find. One parser handles PDF, DOCX, HTML and Markdown, keeps tables intact, and cuts the document into chunks that fit the embedder.

```diagram
id: d-c9
size: 920 440
alt: Docling parsing: converter, document, hybrid chunker, picture classification and captioning, merge into chunk dicts with metadata
caption: Parsing, chunking and figure captioning, merged into the final chunk records.
---
{
  nodes: [
    ['b', 'file bytes\n+ filename', 20, 40, 140, 60, 'white'],
    ['c', 'DocumentConverter\nPDF: layout (ONNX),\nTableFormer (torch),\npicture classifier', 190, 22, 215, 95, 'blue'],
    ['doc', 'DoclingDocument', 435, 42, 150, 55, 'white'],
    ['ch', 'HybridChunker\n512 tokens (bge\ntokenizer), Markdown\ntables', 615, 25, 175, 90, 'green'],
    ['raw', 'raw chunks', 815, 42, 90, 55, 'white'],
    ['pics', 'pictures found\nin the document', 435, 175, 150, 60, 'white'],
    ['cls', 'skip class?', 250, 160, 150, 90, 'yellow', 'diamond'],
    ['cap', 'crop PNG, wait 2 s\nbetween calls,\nMistral caption', 20, 170, 190, 70, 'purple'],
    ['sk', 'skipped', 275, 280, 100, 40, 'grey'],
    ['mg', 'per chunk: contextualize (headings on top),\nsection path, pages, has_table,\nappend [Figure caption: ...]', 300, 345, 320, 75, 'yellow'],
    ['out', 'chunk dicts, then\nenrich_metadata:\nmd5, ingested_at, length', 665, 350, 240, 65, 'green']
  ],
  edges: [
    ['b', 'c', '', 'r', 'l'], ['c', 'doc', '', 'r', 'l'], ['doc', 'ch', '', 'r', 'l'], ['ch', 'raw', '', 'r', 'l'],
    ['doc', 'pics', '', 'b', 't'],
    ['pics', 'cls', '', 'l', 'r'],
    ['cls', 'cap', 'no', 'l', 'r', { lx: 228, ly: 190 }],
    ['cls', 'sk', 'yes', 'b', 't'],
    ['raw', 'mg', '', 'b', 't.8', { via: [[860, 315], [556, 315]] }],
    ['cap', 'mg', '', 'b', 'l', { via: [[115, 382]] }],
    ['mg', 'out', '', 'r', 'l']
  ]
}
```

##### Flow

1. The file bytes are wrapped in a `DocumentStream` and converted by a `DocumentConverter` that is built once and reused. All of this blocking work runs in a worker thread.
2. For PDFs the converter runs layout detection and picture classification through ONNX Runtime, and table structure (TableFormer, accurate mode) through torch. OCR is off by default. DOCX, HTML and Markdown use Docling's own default pipelines.
3. `HybridChunker` splits the document into chunks bounded at 512 tokens using the embedder's own tokenizer, merging small neighbours and repeating the table header in every fragment of a split table.
4. Separately, every figure is classified, optionally skipped by class, cropped to a PNG, and captioned by the Mistral vision model one at a time, with a two-second gap between calls.
5. Each chunk becomes a record: its text is contextualized with its heading path, any captions for figures inside it are appended as `[Figure caption: …]`, and metadata (section path, page numbers, table flag) is attached. `enrich_metadata` then adds a content hash, timestamp and length.

##### Explanation

Several choices in `docling_loader.py` are documented right in its header comments, so they are easy to defend:

- **ONNX where possible.** Layout detection (`layout_heron_default`) and the figure classifier run on ONNX Runtime with no torch. TableFormer has no first-party ONNX path in mainline Docling, so it stays on torch. It is kept on anyway because table accuracy matters for a literature-survey corpus. If the ONNX options cannot be imported, the code falls back to Docling's defaults instead of failing ingestion.
- **Table options V1.** It is pinned to `TableStructureOptions` V1 because V2 has an open upstream bug that duplicates rows on multi-page tables.
- **Own captioning, not Docling's.** Docling's built-in picture description would call the vision API from inside Docling's own HTTP client, bypassing the shared rate limiter and circuit breaker that every other LLM call goes through. So Docling only classifies and crops, and captioning is a separate step through `mistral_client`.
- **Paced captioning.** Calls are sequential and spaced by `CAPTION_MIN_INTERVAL_SECONDS`, because the provider's per-second cap is invisible to a per-minute limiter, and back-to-back calls would still get 429s. A failed caption drops that figure and never fails the document.
- **Tokenizer equals embedder.** The chunker loads the tokenizer for `FASTEMBED_MODEL`, so the 512-token ceiling is the one that actually applies at embedding time.
- **Markdown tables.** A custom serializer provider renders tables as compact Markdown pipe tables. Docling's default (triplet) serializer produces a wall of low-signal text for real academic tables.

```python pipelines/ingestion/loaders/docling_loader.py
hf_tokenizer = AutoTokenizer.from_pretrained(settings.FASTEMBED_MODEL)
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=settings.CHUNK_MAX_TOKENS)
return HybridChunker(tokenizer=tokenizer, merge_peers=True,
                     serializer_provider=_MarkdownTableSerializerProvider(),
                     repeat_table_header=True)
```

The stored chunk text is `chunker.contextualize(chunk)`, which prepends the heading hierarchy, so a paragraph under "Experiments > Results" carries that context into its embedding. Metadata includes the section path (`"A > B"`, or `"Document"` if there are no headings), the first page number, and whether the chunk contains a table. For debugging, `debug_dump.py` can write the chunks and the per-figure classification to disk, so quality can be inspected instead of guessed from log counts. The first run downloads about 1 GB of model weights.

:::qa Why Docling over a simpler PDF text extractor?
A plain extractor flattens tables and loses figures. Docling gives layout-aware structure, table cells, and a document tree whose headings feed the chunker. It covered what this corpus needs (papers with tables and figures), so I did not add a second parser to compare.
:::

:::qa How did you choose 512 tokens?
It matches the embedder's own token ceiling, so chunks are as large as the model can embed without truncation, and it is the simplest setting to defend. Bigger chunks dilute the match and smaller ones lose context. If the eval shows a need, the harness makes a sweep cheap: re-ingest at 256, 512 and 1024 tokens and compare Hit@5.
:::

:::qa Where does it struggle?
Tables. Table-lookup questions had Hit@5 of 0.5 and the lowest correctness (0.592), and the two false refusals in the human spot check were table facts. The evaluation report's own note says the content was likely missed by chunking or retrieval.
:::

:::gaps
##### Trade-offs and gaps

- Qdrant only stores `text`, `filename`, `page`, `section` and `corpus_id`. The table flag, the hash and the page list are computed but not indexed, and citations show a filename only, not a page.
- `INGEST_DEBUG_DUMP` defaults to on and writes raw document content to local disk. The config says to turn it off before any public deploy.
:::

## Tools and the code sandbox {#c10}

Besides retrieval, the agent has two tools: a Python sandbox and web search. Both are deliberately constrained.

```diagram
id: d-c10
size: 920 410
alt: Tools: sandbox lane with HTTP call, Flask runner, child process, timeout handling and deterministic echo; web search lane with user confirmation, Tavily and conversational answer
caption: Two lanes: running code in an isolated container, and confirmed web search.
---
{
  groups: [[385, 12, 520, 235, 'sandbox container: non-root user, egress denied (Kubernetes)']],
  notes: [[20, 28, 'Code lane', 17, 'start', '#777'], [20, 288, 'Web search lane', 17, 'start', '#777']],
  nodes: [
    ['p', 'planner: tool_choice\n= sandbox, tool_input\n= the code', 20, 50, 160, 75, 'orange'],
    ['c', 'run_python_code()\nPOST {code, timeout}\nclient timeout 12 s', 205, 50, 160, 75, 'blue'],
    ['f', 'Flask\nPOST /execute', 410, 55, 125, 60, 'green'],
    ['m', 'child process:\nexec(code), stdout\ncaptured', 560, 50, 150, 70, 'green'],
    ['ok', 'finished: queue ->\nJSON {status, output}', 425, 165, 175, 60, 'green'],
    ['k', 'after 8 s: terminate,\nthen kill, HTTP 408', 660, 165, 225, 60, 'pink'],
    ['res', 'tool_result:\n"Output:..." or\n"Execution Error:..."', 205, 170, 160, 75, 'blue'],
    ['echo', 'responder: echo\nin a code block,\nno LLM', 20, 170, 150, 75, 'orange'],
    ['w1', 'user said yes to\nthe web-search offer', 20, 315, 190, 65, 'orange'],
    ['w2', 'tool_input =\ntopic from the offer', 240, 315, 160, 65, 'blue'],
    ['w3', 'Tavily POST: basic,\n3 results, 10 s', 430, 315, 170, 65, 'green'],
    ['w4', 'tool message\nadded to history', 630, 315, 120, 65, 'blue'],
    ['w5', 'conversational\nresponder', 780, 315, 120, 65, 'orange']
  ],
  edges: [
    ['p', 'c', '', 'r', 'l'], ['c', 'f', '', 'r', 'l'], ['f', 'm', '', 'r', 'l'],
    ['m', 'ok', '', 'b.3', 't.8'], ['m', 'k', 'timeout', 'b.8', 't', { lx: 740, ly: 143 }],
    ['ok', 'res', '', 'l', 'r'], ['res', 'echo', '', 'l', 'r'],
    ['k', 'res', 'status != 200', 'b', 'b', { via: [[772, 270], [285, 270]], dashed: true }],
    ['w1', 'w2', '', 'r', 'l'], ['w2', 'w3', '', 'r', 'l'], ['w3', 'w4', '', 'r', 'l'], ['w4', 'w5', '', 'r', 'l']
  ]
}
```

##### Flow

1. **Sandbox:** the planner outputs `tool_use` with `tool_choice="sandbox"` and the Python code as `tool_input`.
2. `run_python_code` POSTs `{code, timeout}` to the sandbox service with a 12-second client timeout, which is longer than the server's worst case so a timeout is clean.
3. The Flask runner starts a child process that `exec`s the code with stdout redirected into a buffer, and waits for the soft timeout.
4. If it finishes, the result goes back through a queue as JSON. If not, the process is terminated, then killed, and the server returns 408.
5. The API turns the result into `"Output:\n…"` or an error string, and the responder echoes it with no LLM call.
6. **Web search:** only after a zero-hit answer offered it and the user replied yes. The planner extracts the topic from the offer, the tool node calls Tavily, and the results go to the conversational responder.

##### Explanation

Generated code never runs inside the API process. It goes to a separate container built from a minimal Python image, running as a non-root user. The container also has a health endpoint, and the Dockerfile's health check uses Python's standard library because the slim image has no `curl`. In Kubernetes, `network-policy.yaml` selects the sandbox pods and declares an empty egress list, which means the sandbox can receive requests but can never call out.

```python services/sandbox/runner.py
p = multiprocessing.Process(target=execute_code_safe, args=(code, queue))
p.start()
p.join(SOFT_TIMEOUT)            # 8 s, from limits.yaml
if p.is_alive():
    p.terminate(); p.join(2)
    if p.is_alive(): p.kill()
    return jsonify({"output": "Error: Execution timed out."}), 408
```

A child process, rather than a thread, is what makes a hard timeout possible: a runaway loop can be killed, which a thread cannot be. The tool returns every failure mode, whether a connection failure, a non-200 status or an execution error, as a readable string, so the rest of the graph never has to handle exceptions from it.

Web search is a plain call to Tavily's API (basic depth, three results, ten-second timeout). If no `TAVILY_API_KEY` is set it returns a "disabled" message instead of failing. Results are formatted as title, content and URL and then summarised by the conversational responder. Because answers from web search are never cached, stale web content cannot be replayed.

:::qa Is the sandbox secure?
Its safety comes from the container boundary, not from the interpreter. Code runs in a disposable process, as a non-root user, with a hard timeout, inside a container with no network egress in Kubernetes. Inside that process, `exec` gets the full set of builtins, so code can import modules and touch the container's filesystem. For a demo with no sensitive data in the sandbox, that was enough. If it ever ran untrusted code at scale, the hardening steps are container memory, CPU and process limits, a read-only filesystem, a seccomp profile with dropped capabilities, an enforced import allowlist, then gVisor, plus the egress block in Compose and not only in Kubernetes.
:::

:::qa Why does the planner write the code?
The planner's JSON has a `tool_input` field defined as "the exact input to pass to the tool", so for the sandbox the LLM supplies the program text directly.
:::

:::gaps
##### Trade-offs and gaps

- `limits.yaml` lists an allowed-imports whitelist, a memory limit and a CPU limit, but the runner only reads the soft timeout from it. The others are documentation, not enforcement. The Compose service has a 256 MB memory cap.
- The API sends `timeout: 5`, but the runner ignores that field and uses 8 seconds from `limits.yaml`.
- The image installs only Flask and PyYAML, so the pandas and numpy imports named in the YAML would not work.
- The Docker Compose sandbox has no egress restriction. Only the Kubernetes policy does, and the repo has the policy file but no Kubernetes deployment yet (deployment is still on the roadmap).
- The Tavily key is read with `os.getenv` instead of through the settings object, so it must be a real environment variable.
:::

## Evaluation harness {#c11}

I wanted numbers instead of impressions, so `eval/` runs the shipped pipeline end to end against a fixed corpus and a hand-verified question set. Nothing is reimplemented for evaluation: it calls the same ingestion, retrieval and agent code as the app.

```diagram
id: d-c11
size: 920 400
alt: Evaluation stages: ingest, verify gold, retrieval ablation, generation shipped and closed-book, judge check, judging, human spot check, report
caption: Each stage is its own command and writes one JSON file per question.
---
{
  notes: [[20, 22, 'Collect', 17, 'start', '#777'], [20, 222, 'Score', 17, 'start', '#777']],
  nodes: [
    ['s1', 'ingest\n5 pinned arXiv PDFs,\nreal pipeline', 20, 45, 160, 75, 'green'],
    ['s2', 'verify-gold\ngold span in a\nchunk?', 205, 45, 145, 75, 'yellow'],
    ['s3', 'retrieve\ndense, BM25, hybrid,\nhybrid + rerank', 375, 40, 170, 85, 'blue'],
    ['s4', 'generate\nshipped agent', 575, 30, 140, 55, 'orange'],
    ['s5', 'generate\nclosed-book', 575, 100, 140, 55, 'grey'],
    ['db', 'results/\nretrieval, generation,\njudge: one JSON per\nquestion', 745, 30, 155, 120, 'grey', 'db'],
    ['s6', 'judge-check\n1 question,\n6 metrics', 20, 255, 150, 70, 'yellow'],
    ['s7', 'judge --retrieval\ncontext precision,\nrecall', 200, 235, 165, 65, 'purple'],
    ['s8', 'judge --generation\nfaithfulness, relevancy,\ncorrectness, refusals', 200, 315, 165, 75, 'purple'],
    ['s9', 'human spot-check\n6 answers', 400, 265, 150, 60, 'pink'],
    ['s10', 'report.py\nresults/report.md', 585, 265, 160, 60, 'green']
  ],
  edges: [
    ['s1', 's2', '', 'r', 'l'], ['s2', 's3', '', 'r', 'l'],
    ['s3', 's4', '', 'r', 'l'], ['s3', 's5', '', 'r.9', 'l'],
    ['s4', 'db', '', 'r', 'l'],
    ['s5', 's6', '', 'b', 't', { via: [[645, 185], [95, 185]] }],
    ['s6', 's7', '', 'r', 'l'], ['s6', 's8', '', 'r', 'l'],
    ['s7', 's9', '', 'r', 'l'], ['s8', 's9', '', 'r', 'l'],
    ['s9', 's10', '', 'r', 'l']
  ]
}
```

##### Flow

1. `ingest` downloads five pinned arXiv papers (Transformer, ResNet, BERT, LoRA, RAG) and pushes them through the production ingestion path under a fixed corpus id.
2. `verify-gold` scrolls every indexed chunk out of Qdrant and checks that each question's gold span is a substring of at least one. Nothing proceeds until it passes.
3. `retrieve` runs dense-only, BM25-only, hybrid and hybrid+rerank for each of the 25 questions. No LLM calls.
4. `generate` runs the real LangGraph agent for each question, uncached and single-turn. `generate --closed-book` asks the same model with no retrieval.
5. `judge-check` scores one question on all six metrics as a pre-flight check. Then `judge --retrieval` and `judge --generation` run the full sweep.
6. A human spot-check compares the judge's scores against my own on six answers. `report` aggregates everything into `report.md`.

##### Explanation

**The question set.** 25 questions with human-verified reference answers and gold spans: 7 single-fact, 5 multi-passage, 4 table lookup, 2 figure/caption, 3 cross-paper, and 4 unanswerable ones that have no gold span on purpose, so abstention can be measured in both directions.

**Metrics.** Deterministic code scores retrieval (Hit@5, MRR, Precision@5), false-refusal rate (by regex), citation validity (does every `[Source: X]` match a retrieved source) and latency. An LLM judge, through the real RAGAS implementations, scores context precision, context recall, faithfulness, answer relevancy and answer correctness. A separate binary judge decides whether a refusal on an unanswerable question was a genuine decline. Telling a real decline from a confident fabrication needs judgment, while telling "refused" from "answered" on an answerable question is pattern matching, so each uses the right tool.

**Independence.** The judge is a different vendor from the generator and the captioner (Gemma or Cohere for judging, Groq for generation, Mistral for captions), so none of them grades its own work. The reported numbers come from the Cohere judge.

**Resumable.** Every (stage, question) is one JSON file, written atomically through a temp file and rename, and a call is skipped if its key is already filled. A crash or a daily rate limit loses at most one call, and rerunning the same command continues where it stopped.

**Dependencies.** RAGAS lives in its own `eval` group because it drags in about 30 packages that should never be in the deployed image. RAGAS 0.4.3 also has an import bug (it imports a VertexAI module that current langchain-community removed), so `_ragas_compat.py` stubs that import and every eval module loads it first.

#### Results (answerable questions, n = 21)

| Retrieval variant | Hit@5 | MRR | Precision@5 |
| --- | --- | --- | --- |
| dense only | 0.571 | 0.421 | 0.238 |
| BM25 only | 0.571 | 0.516 | 0.286 |
| hybrid (RRF) | 0.619 | 0.532 | 0.276 |
| hybrid + rerank (shipped) | 0.667 | 0.563 | 0.305 |
| rewritten query + rerank (tested, then removed) | 0.619 | 0.508 | 0.295 |

I also tested rewriting the query before retrieval. It did not beat plain hybrid + rerank (Hit@5 0.619 against 0.667), so I removed it from the retrieval path and deleted the script that produced it. That is why the table has a row for it but the repo has no code for it. No claim on this page depends on that variant.

| Measure | Shipped RAG | Closed-book |
| --- | --- | --- |
| Answer correctness (mean, n = 21) | 0.77 | 0.708 |
| Correct refusals (unanswerable, n = 4) | 1.0 (4/4) | 0.75 (3/4) |
| False refusals (answerable, n = 21) | 0.143 (3/21) | 0.0 (0/21) |
| Overall correctness (all 25) | 0.807 | 0.715 |
| Latency p50 / p95 | 6.0 s / 9.9 s | 1.1 s (p50) |

Judge-scored retrieval and generation (shipped): context precision 0.865, context recall 0.952, faithfulness 0.705, answer relevancy 0.757. Citation validity was 1.0. By category, correctness was highest for single-fact (0.833) and lowest for table lookup (0.592); Hit@5 was lowest for cross-paper questions (0.333).

**Human spot-check.** I hand-scored six answers (the two highest, two lowest and two median by judge score) before seeing the judge's number. The judge agreed within 0.15 on five of six (83%). It was too lenient on refusals: it gave a flat "I don't have information" answer 0.43. So for refusals I trust the deterministic false-refusal rate more than the judged correctness score.

:::qa What did the evaluation teach you?
Hybrid plus rerank was the best retrieval setup, BM25 was stronger than dense on exact-term questions, and tables and cross-paper questions are the weak spots. The shipped pipeline refused all four unanswerable questions correctly but also wrongly refused three answerable ones, which points at the missing relevance threshold. If those categories need work, table-aware chunking and multi-query retrieval are the options. The judge was lenient on refusals, so for those I trust the deterministic false-refusal rate more.
:::

:::qa Why is faithfulness only 0.705?
The single-fact category scored 0.56 while multi-passage scored 0.971, so the average hides a lot of spread, and single-fact has only 7 questions. I treat 0.705 as a baseline. If it needs fixing, the next step is to read the lowest-scoring single-fact answers and sort them into judge strictness versus the model adding facts that are not in the context.
:::

:::qa How much should I trust these numbers?
As a baseline, not a benchmark. There are only 25 questions (21 answerable), so one flipped question moves a rate by about 5 points. There is one judge per run, and the closed-book baseline is contaminated because these famous papers are likely memorised by the model, so the retrieval ablation is the cleaner evidence that RAG helps.
:::

## Data model, frontend, dev setup and tests {#c12}

```diagram
id: d-c12
size: 920 470
alt: Data model across stores: Postgres tables, Redis keys, Qdrant point, and S3 key layout, all keyed by corpus_id
caption: What each store holds. Every one of them is keyed or filtered by the same `corpus_id`.
---
{
  notes: [[460, 232, 'one tenant key (corpus_id) ties every store together', 17]],
  nodes: [
    ['ch', 'chat_history (Postgres)\nid, corpus_id (indexed)\nrole, content\nmetadata_, created_at', 20, 30, 270, 130, 'blue'],
    ['dc', 'documents (Postgres)\nfile_id (unique), corpus_id\nfilename, s3_key, status\nerror, chunk_count\ncreated_at, completed_at', 320, 30, 280, 160, 'blue'],
    ['fb', 'feedback (Postgres)\nid, corpus_id, message_id\nscore (1 / -1), category\ncomment, created_at', 630, 30, 270, 130, 'blue'],
    ['rd', 'Redis keys\nsessions:active (sorted set)\nrl:{scope}:{subject}:{window}\ncache:{corpus}:{schema}:{hash}\ncorpus_version:{corpus}', 20, 280, 290, 150, 'yellow'],
    ['qd', 'Qdrant point\nid: random uuid\nvectors: dense (1024), sparse\npayload: text, filename,\npage, section, corpus_id', 335, 280, 270, 150, 'green'],
    ['s3', 'Object storage key\nuploads/{corpus_id}/\n{file_id}.{ext}\nmetadata: original_filename,\ncorpus_id', 630, 280, 270, 150, 'grey']
  ]
}
```

##### Flow

1. The browser loads `index.html`, calls `/session/init`, and lists the documents already in the session.
2. An upload runs the presign, PUT, and status-polling flow described earlier.
3. A chat message is POSTed and the NDJSON response is read line by line: status lines update a single live label, the answer line renders the message with feedback buttons, badges and sources.
4. Thumbs up or down POSTs to `/feedback/` with the message id.

##### Explanation

**Postgres tables.** `ChatHistory` holds the conversation, `Document` tracks each upload's ingestion stage (`pending → fetching → parsing → embedding → indexing → complete | failed`, with an error string and a chunk count), and `Feedback` stores the score and optional category. Tables are created at startup with `create_all`; there is no migration tool, so schema changes need a manual `ALTER TABLE`. Cascade delete removes all three tables' rows for a session in one transaction.

**Configuration.** One pydantic `Settings` object reads every tunable from environment variables or `.env`: backends, model lists, retrieval, cache thresholds, ingestion, rate limits, logging and eval keys. Only the database URL, Redis URL and bucket name are required. Optional services degrade gracefully: without OpenRouter you lose failover, without a Mistral key ingestion still completes but figures go uncaptioned, and without Tavily web search reports itself as disabled. Logging is a readable console handler plus a rotating JSON file handler.

**Frontend.** A single vanilla-JS page with no build step, served as static files by FastAPI. It maps node names to friendly labels, shows a small badge for cache hit, tool and model, renders sources and the earlier answer, and turns a 429 into a "retry in N seconds" message using the `Retry-After` header.

**Dev workflow.** The API runs on the host (`make dev`, uvicorn with reload) while the stores run in Docker. Every Compose service sits behind a profile (`core` Postgres, `cache` Redis Stack, `vector` Qdrant, `storage` MinIO, `sandbox`), and `make up PROFILE=…` requires you to name the ones you want, so nothing runs by accident. `make stop` keeps volumes, `make down-all` removes containers, `make nuke` deletes volumes after a typed confirmation. A one-shot `minio-init` container creates the bucket and registers the webhook, retrying until the API is up. Dependencies are managed with uv in groups (`api`, `dev`, `ingestion`, `eval`), on Python 3.13 or later.

**Tests.** Pytest covers the guardrails (attack strings blocked, normal questions allowed, end-to-end through the responder), plan validation, LLM client failover and model ordering, backend selection, the Mistral limiter settings, truncation handling, the Redis rate limiter against a fake Redis (limit, 429 with `Retry-After`, window rollover, fail-open, trusted-proxy handling, oversize uploads), and the session store and dependency. A conftest force-sets the required environment variables so tests never read a real `.env`. `scripts/load_test.py` is a Locust test where each simulated user creates a real session first. `scripts/bulk_upload_s3.py` is a parallel multipart uploader for seeding data.

## Trade-offs and gaps, in one place {#limits}

These are the trade-offs of keeping the system simple, grouped by area. Each one gets addressed when a real need shows up.

- **Caching:** the cache key ignores conversation context; `NO_CACHE_TOOLS` is a string, so sandbox answers are cached and empty-tool answers are not; the semantic thresholds were not measured.
- **Retrieval:** no relevance threshold before generation; the tenant filter should also be placed inside each prefetch; tables and cross-paper questions are the weakest categories.
- **Agent:** single-pass routing, no re-plan loop; HyDE changes the question the planner and responder see; the planner's `vector_search` tool path is unreachable.
- **Ingestion:** in-process background task (lost on restart, marked failed on boot); no deduplication; webhook has no authentication; no file-type check at upload.
- **Sandbox:** full builtins inside the process, limits in YAML that are not enforced, and no egress block outside Kubernetes.
- **Sessions:** unsigned cookie that is also the tenant id; cookie lifetime not refreshed while Redis is; cleanup runs in every replica.
- **Guardrails:** regex only, no groundedness check on answers (faithfulness 0.705).
- **LLM layer:** shared per-client attributes for model and finish reason can race; Ollama and vLLM untested live.
- **Ops:** no tracing or metrics yet, no migrations, and debug dumping of raw document content is on by default.

## What's next? {#next}

- Fix the known bugs
- Add per-node tracing
- Diagnose the retrieval misses
- Cut false refusals
- Explain the faithfulness score
- Grow the evaluation
- Harden the sandbox
- Upgrade the agent: re-plan loop and token streaming
- Make ingestion durable

---

:::note
Source code: [github.com/arponkapuria/scalable-agentic-rag-pipeline](https://github.com/arponkapuria/scalable-agentic-rag-pipeline)
:::

:::note
About the painting: ["The Death of Socrates (1787)" by Jacques-Louis David (public domain), from Wikimedia Commons](https://commons.wikimedia.org/wiki/File:David_-_The_Death_of_Socrates.jpg#/media/File:David_-_The_Death_of_Socrates.jpg)
:::
