import os
from functools import cache

import torch
from rank_bm25 import BM25Okapi

DEVICE = os.environ.get("LENS_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
RANKERS = ("bm25", "single", "maxsim", "cross")
ALLOWED_RANKERS = RANKERS


def maxsim(q, docs):
    """q: [Lq, d]; docs: list of [Ld, d]. Score = sum_i max_j q_i . d_j per doc."""
    return torch.stack([(q @ d.T).max(dim=1).values.sum() for d in docs])


# Fixed max lengths bound the set of MPS graph shapes; uncapped, driver memory grows per new shape until OOM.
@cache
def _model(name):
    if DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(min(1.0, 20 * 2**30 / torch.mps.recommended_max_memory()))
    if name == "single":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)
        m.max_seq_length = 128
        return m
    if name == "maxsim":
        from pylate import models
        return models.ColBERT("answerdotai/answerai-colbert-small-v1", device=DEVICE, query_length=32,
                             document_length=64)
    if name == "cross":
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE,
                            max_length=128)


def rank(name, query, texts):
    if name == "bm25":
        return torch.tensor(BM25Okapi([t.lower().split() or [""] for t in texts]).get_scores(query.lower().split()))
    m = _model(name)
    if name == "single":
        e = m.encode([query] + texts, convert_to_tensor=True, normalize_embeddings=True)
        return (e[1:] @ e[0]).cpu()
    if name == "maxsim":
        q = torch.as_tensor(m.encode([query], is_query=True)[0])
        docs = [torch.as_tensor(d) for d in m.encode(texts, is_query=False)]
        return maxsim(q, docs)
    if name == "cross":
        return torch.tensor(m.predict([(query, t) for t in texts], batch_size=64))
