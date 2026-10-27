import os
from functools import cache

import torch
from rank_bm25 import BM25Okapi

DEVICE = os.environ.get("LENS_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
RANKERS = ("bm25", "single", "maxsim", "cross")
ALLOWED_RANKERS = RANKERS + ("pagemaxsim",)


def maxsim(q, docs):
    """q: [Lq, d]; docs: list of [Ld, d]. Score = sum_i max_j q_i . d_j per doc."""
    return torch.stack([(q @ d.T).max(dim=1).values.sum() for d in docs])


# Fixed max lengths bound the set of MPS graph shapes; uncapped, driver memory grows per new shape until OOM.
@cache
def _model(name, node_text="own"):
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
                             document_length=128 if node_text == "context" else 64)
    if name == "cross":
        from sentence_transformers import CrossEncoder
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE,
                            max_length=192 if node_text == "context" else 128)


def rank(name, query, texts, node_text="own"):
    if name == "bm25":
        return torch.tensor(BM25Okapi([t.lower().split() or [""] for t in texts]).get_scores(query.lower().split()))
    m = _model(name, node_text)
    if name == "single":
        e = m.encode([query] + texts, convert_to_tensor=True, normalize_embeddings=True)
        return (e[1:] @ e[0]).cpu()
    if name == "maxsim":
        q = torch.as_tensor(m.encode([query], is_query=True)[0])
        docs = [torch.as_tensor(d) for d in m.encode(texts, is_query=False)]
        return maxsim(q, docs)
    if name == "cross":
        return torch.tensor(m.predict([(query, t) for t in texts], batch_size=64))


def encode_doc_tokens(texts_as_word_lists: list[list[str]]) -> list[tuple[torch.Tensor, list[int | None]]]:
    """Encode fixed-length documents before punctuation filtering, retaining word IDs."""
    if not texts_as_word_lists:
        return []
    model = _model("maxsim", "context")
    model.eval()
    result = []
    for start in range(0, len(texts_as_word_lists), 32):
        batch = texts_as_word_lists[start:start + 32]
        use_prefix = model.document_prefix_id is not None
        encoded = model.tokenizer(batch, is_split_into_words=True, padding="max_length",
                                  truncation=True, max_length=128 - int(use_prefix), return_tensors="pt")
        indices = [encoded.word_ids(i) for i in range(len(batch))]
        features = dict(encoded)
        if use_prefix:
            for key, fill in (("input_ids", model.document_prefix_id), ("attention_mask", 1), ("token_type_ids", 0)):
                if key in features:
                    features[key] = model.insert_prefix_token(features[key], fill)
            indices = [ids[:1] + [None] + ids[1:] for ids in indices]
        features = {key: value.to(model.device) for key, value in features.items()}
        with torch.inference_mode():
            embeddings = model(features)["token_embeddings"]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1).cpu()
        result.extend(zip(embeddings, indices))
    return result


def rank_page(query: str, words: list[tuple[str, str | None]], refs: list[str],
              window: int = 64, stride: int = 32) -> torch.Tensor:
    """MaxSim per candidate over its contextual tokens across all page windows."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    scores = torch.full((len(refs),), float("-inf"))
    if not words or not refs:
        return scores
    model = _model("maxsim", "context")
    q = torch.as_tensor(model.encode([query], is_query=True)[0]).cpu()
    candidates = set(refs)
    best = {}
    # Bound peak memory while keeping fixed 128-token document shapes.
    starts = list(range(0, len(words), stride))
    for offset in range(0, len(starts), 32):
        chunk = starts[offset:offset + 32]
        encoded = encode_doc_tokens([[w for w, _ in words[start:start + window]] for start in chunk])
        for start, (embedding, indices) in zip(chunk, encoded):
            similarities = q @ embedding.T
            owners = {}
            for token, word_index in enumerate(indices):
                if word_index is not None:
                    ref = words[start + word_index][1]
                    if ref in candidates:
                        owners.setdefault(ref, []).append(token)
            for ref, tokens in owners.items():
                maxima = similarities[:, tokens].max(dim=1).values
                best[ref] = torch.maximum(best[ref], maxima) if ref in best else maxima
    for i, ref in enumerate(refs):
        if ref in best:
            scores[i] = best[ref].sum()
    return scores
