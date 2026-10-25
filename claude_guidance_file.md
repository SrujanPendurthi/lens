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

## 3. Codebase Standards & Technical Constraints

### Threading & Async Isolation
* Use `asyncio` for all networking, browser management, and protocol streams.
* The **Background Graph Daemon** (CDP streaming + PyTorch GPU encoder) MUST run in an isolated worker thread or process. Never block the main asyncio event loop with GPU tensor passes.
* Use `asyncio.to_thread()` or CUDA stream queues for encoder passes.

> **v1 status (2026-10-23):** v1 is the offline Mind2Web eval (`eval.py`, plan D1-D7). The index, quantization and CUDA rules below are **later phase (post v1 gate)**. In v1, MaxSim is exact torch over per-node vectors, the dev device is MPS, and the deploy device is CUDA via `LENS_DEVICE`.

### Vector Databases & Search Engines
* Do **NOT** use standard single-vector ANN engines (Pinecone, Chroma, standard HNSW) for token-level Late Interaction.
* All local multi-vector indexes MUST use specialized late-interaction backends (e.g., PLAID C++ bindings, custom PyTorch/Numba MaxSim kernels, or Vespa).

### Quantization Rules
* Intermediate token arrays must use FP16 or INT8 formats.
* Low-relevance peripheral tokens must be quantized to 1-bit binary vectors evaluated via Hamming distance XOR operations.

## 4. Protocol Control-Flow Pattern

Always structure agent interaction loops using the async yield-and-inspect primitive:

```python
# Protocol Pattern Example
async for graph_snapshot in lens_stream.subscribe(intent_query=agent_goal):
    # Check if top nodes satisfy confidence threshold
    if graph_snapshot.max_confidence >= 0.85:
        top_patches = graph_snapshot.get_top_patches(k=3)
        action = await agent.decide_action(top_patches)
        await browser_session.execute(action)
        break
    else:
        # Pause reasoning loop to let background GPU encoder expand deeper nodes
        await lens_stream.pause_for_index(timeout_ms=100, confidence_threshold=0.85)
```

## 5. Primary Contribution Areas

1. **Quantization Kernels:** Writing fast C++/CUDA/Numba implementations of PLAID centroid residual packing and 1-bit binary MaxSim scoring.
2. **Dynamic Graph Pruners:** Implementing intent-steered graph node eviction algorithms.
3. **Protocol Bindings:** Maintaining WebSocket/gRPC primitive handlers for `YIELD_AND_INSPECT` and `UPDATE_INTENT_VECTOR`.