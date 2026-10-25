# Claude Developer & Context Guidance (claude.md)

This file provides system context, architectural conventions, and code generation guidelines for working on the **TokenLens Agent Infrastructure** codebase.

## 1. System Vision & Paradigm

When writing code or assisting with architecture in this repository, remember:

* We **do not** dump entire raw HTML pages or standard DOM trees into the main LLM context window.
* We treat the web page as a **dynamic multi-vector knowledge graph** evaluated via Late Interaction (ColBERT / ColPali style MaxSim operators).
* We enforce strict **asynchronous isolation** between the background encoding daemon and the primary agent reasoning loop.

## 2. Core Concepts & Nomenclature

| Term | Definition |
| ----- | ----- |
| **Late Interaction** | Token-level dot-product scoring deferred until query time: $\sum_{i \in Q} \max_{j \in D} (q_i \cdot d_j^\top)$. |
| **Patch Vector** | Multi-vector embedding representing a $32 \times 32$ visual sub-region of a web viewport. |
| **Intent Vector ($Q$)** | Embeddings representing the agent's current sub-goal, used to prune irrelevant page graph nodes. |
| **Yield-and-Inspect** | Asynchronous control flow enabling agents to act on partial page index states. |

## 3. Current status
v1 is the offline Mind2Web evaluation implemented in `eval.py`.
The multi-vector index, quantization, CUDA and yield-and-inspect work is deferred per PLAN.md D5/D8.
The runner, ranker limits and picker decisions live in PLAN.md.
Use the cached Qwen sub-goals; do not resume the Qwen3-4B picker (D10).
The Haiku backend and current evaluation evidence are documented in BRIEF_5_REPORT.md.
