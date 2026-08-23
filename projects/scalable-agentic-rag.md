---
Project: Scalable Agentic RAG Pipeline
Github: https://github.com/arponkapuria/scalable-agentic-rag-pipeline
---

<h3>Project Description</h3>

This project is a production-oriented Agentic RAG system designed to answer complex enterprise questions accurately while remaining scalable and cost-efficient. The overall pipeline consists of four major stages: query planning and retrieval, context preparation, LLM inference, and production monitoring & evaluation.

When a user submits a query through the FastAPI endpoint, the request is first passed to a LangGraph planner, which acts as the orchestrator of the workflow. Unlike a traditional RAG pipeline where every query follows the same sequence of steps, the planner decides how the query should be processed based on its complexity. For example, a simple factual question may only require semantic retrieval, whereas a multi-hop or ambiguous query may trigger additional steps such as query rewriting or HyDE before retrieval.

Once the retrieval strategy is selected, the query is converted into embeddings using BGE-M3. The system then performs hybrid retrieval by querying both Qdrant, which provides semantic similarity search over document embeddings, and Neo4j, which retrieves relationship-based information through graph traversal. This combination allows the system to capture both semantic similarity and explicit entity relationships, improving performance on multi-hop reasoning tasks.

The retrieved candidates are then passed through a cross-encoder reranker. While vector search efficiently retrieves candidate documents, the reranker jointly evaluates the query and each document to produce a more accurate relevance score. The highest-ranked documents are selected and assembled into the final prompt.

The prompt is then sent to an LLM served using vLLM. To maximize GPU utilization and reduce latency, the inference server uses techniques such as continuous batching, AWQ quantization, and tensor parallelism where appropriate. Model serving is handled by Ray Serve, which manages replicas, request routing, and autoscaling to support concurrent users efficiently.

The entire infrastructure is provisioned using Terraform on AWS, with EKS orchestrating containerized services, Aurora Serverless managing relational data, ElastiCache providing caching, and S3 storing documents. Karpenter dynamically provisions CPU and GPU nodes, including GPU scale-to-zero, to minimize infrastructure cost during idle periods.

Finally, the system is instrumented using OpenTelemetry, Prometheus, and Grafana for distributed tracing, metrics collection, and monitoring. To evaluate quality, retrieval and generation are assessed offline using RAGAS and an LLM-as-a-Judge pipeline against a curated golden dataset, ensuring both retrieval relevance and answer faithfulness before deployment.