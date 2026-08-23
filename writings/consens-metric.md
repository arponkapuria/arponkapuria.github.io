---
title: ConSens: Assessing Context Grounding in Open‑Book Question Answering
description: A concise summary and analysis of ConSens, a score used in assessing context grounding in open‑book question answering

date: June 19, 2025
modified: June 19, 2025

author: Arpon Kapuria
category: Research Journal
tags: LLM Reliability, Paper Implementation
---

## Background

Large Language Models (LLMs) often rely on both internal knowledge and external context when answering open-book questions. Evaluating how much of an answer is actually grounded in the provided context is critical for assessing factuality and trustworthiness. Existing metrics either require human annotations, are resource-intensive, or cannot isolate context usage.

To address this, the authors propose **ConSens**, a simple and model-logit-based score that quantifies how dependent an answer is on external context by measuring how much more confidently the model answers with context than without it. If a model produces a much lower perplexity (i.e., more confidence) when context is available versus when it is not, the answer is more likely to be grounded in that context.


## Methodology

### 2.1 Key Assumption

If a model produces a much lower perplexity (i.e., more confidence) when context is available versus when it is not, the answer is more likely to be grounded in that context.

### 2.2 Step-by-Step Computation

- #### *Compute Perplexity with Context (PC)*
   - Pass the model the input: `context + question`.
   - Compute perplexity over the given answer text.
   - This represents how easily the model predicts the answer when context is available.

- #### *Compute Perplexity without Context (PE):*
   - Pass only the question to the model, removing any external grounding.
   - Again, compute perplexity over the same answer.
   - This tests how well the model would answer without the given context.

### 2.3 Contrastive Signal Calculation

Calculate the log-ratio of perplexities:

$$r = \log\left(\frac{PE}{PC}\right)$$

A higher value of $r$ means that the model finds the answer much harder to generate without context, suggesting stronger dependence on the context.

### 2.4 Sigmoid Transformation

To make the score interpretable and normalized to $[−1,+1]$, apply:

$$ConSens = 2(1 + e^{-r})^{-1} - 1$$

- Values close to $+1$: Strongly grounded in context
- Values close to $0$: Neutral or ambiguous
- Values close to $–1$: Answer likely comes from internal memory or is hallucinated

### 2.5 Token Filtering (Optional Preprocessing)

To avoid skewing results with uninformative tokens:

- Exclude “closed-class” tokens (e.g., determiners, conjunctions).
- Exclude tokens already present in the question.

This makes the contrast sharper and the score more reflective of actual grounding.

### 2.6 Efficiency and Scope

- *Model-Agnostic*: Can be used with any model as long as you have access to token-level logits.
- *No Annotations Required*: Purely automated and cheap to compute.
- *Scales Across Model Sizes*: Tested on models from 1B to 70B parameters.

<hr>

## References

- **Paper:** Vankov, I., Ivanov, M., Correia, A., & Botev, V. (2025). *ConSens: Assessing Context Grounding in Open-Book Question Answering*. arXiv:2505.00065. [Paper](https://arxiv.org/abs/2505.00065)
- **Implementation:** ConSens Metric Implementation — [GitHub Notebook](https://github.com/arponkapuria/notebooks/blob/main/conSens.ipynb)