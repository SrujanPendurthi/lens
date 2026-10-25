# Product Requirements Document (PRD): Real-Time Agentic Web Lens System

**Product Name:** Project TokenLens (Web Agent Perception & Execution Engine)

**Target Audience:** Autonomous AI Agent Frameworks, Web Automation Engineers, Deep Research Agents

**Version:** 1.2.0-Draft

## 1. Purpose & Goals

### 1.1 Objective

To construct an open-spec, high-throughput web perception engine that streams web pages, converts them into token-level multi-vector knowledge graphs, and allows AI agents to traverse and interact with web applications with sub-2-second loop latencies.

### 1.2 Key Metrics & OKRs

* **Loop Latency:** Reduce full agent perception-action cycle to $< 2.0\text{s}$ (P90).

* **Token Efficiency:** Reduce LLM input context payload per browser step by $\ge 80\%$.

* **Accuracy:** Achieve $\ge 95\%$ accuracy on fine-grained UI element selection (tables, forms, dynamic buttons).

* **Memory Footprint:** Compress token multi-vector memory usage by $\ge 10\times$ using PLAID residual quantization and binary Hamming operations.

## 2. System Architecture & Components

```
   [ External Web Page / Browser Session (Playwright / Browserbase) ]
                                 │
                                 ▼ (Raw HTML / CDP Viewport Stream)
┌────────────────────────────────────────────────────────────────┐
│  Background Daemon Thread                                      │
│                                                                │
│  1. Tokenizer / ColPali Vision Patch Encoder (GPU Forward)     │
│  2. Static Quantizer (1-bit / Residual Centroid Offsets)       │
│  3. Intent-Aware Graph Pruner ($Q \cdot D^\top$)                │
└───────────────────────────────┬────────────────────────────────┘
                                │ (Streams Compressed Multi-Vectors)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Volatile Local Graph Buffer (In-Memory C++ / PLAID Index)     │
│  - High-Relevance Subgraphs (Lossless FP16)                    │
│  - Low-Relevance Subgraphs (Binary / Quantized)                │
└───────────────────────────────┬────────────────────────────────┘
                                │ (MaxSim Late Interaction Sub-Queries)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│  Agent Reasoning Engine (LLM)                                  │
│  - Yield-and-Inspect Protocol Controller                       │
└────────────────────────────────────────────────────────────────┘

```

## 3. Key Functional Requirements

### FR-1: Multi-Vector Encoder (Text & Vision)

* **FR-1.1:** System MUST accept raw HTML streams or $1024 \times 1024$ viewport screenshots via Chrome DevTools Protocol (CDP).

* **FR-1.2:** System MUST generate 128d multi-vectors via ColBERTv2 (text) or ColPali (vision patch tokens) within $< 200\text{ms}$ on GPU.

### FR-2: Two-Tier Compression Engine

* **FR-2.1 (Static):** System MUST automatically strip or 1-bit quantize structural stop-words and DOM wrapper tags prior to index insertion.

* **FR-2.2 (Intent-Driven):** System MUST accept an active Intent Vector ($Q$

  $$
  \text{Score}(Q, D) = \sum_{i \in Q} \max_{j \in D} \left( q_i \cdot d_j^\top \right)
  $$

* **FR-2.3:** Tokens scoring below threshold $\tau_{\text{low}}$

### 

* 

* 

* 

* 

### 

* 

* 

* 