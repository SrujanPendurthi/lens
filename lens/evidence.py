"""Budgeted evidence from raw pages, chunk retrieval, sentence windows or reranking."""
from functools import cache
import re
import time

from lens.rank import rank


@cache
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)


def count(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False, truncation=False))


def chunks(text: str, size=256, overlap=32) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('require size > overlap >= 0')
    offsets = _tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True,
                           truncation=False)['offset_mapping']
    parts, start = [], 0
    while start < len(offsets):
        end = min(start + size, len(offsets))
        while end > start and count(text[offsets[start][0]:offsets[end-1][1]]) > size:
            end -= 1
        if end == start:
            raise ValueError('one source token exceeds chunk size after slicing')
        parts.append(text[offsets[start][0]:offsets[end-1][1]])
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)
    return parts


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def _render(units):
    return '\n\n'.join(f'[{title}] {text}' for title, text, _ in units)


def pack(units: list[tuple[str, str, float]], budget: int) -> str:
    selected = []
    for unit in sorted(units, key=lambda u: u[2], reverse=True):
        if count(_render(selected + [unit])) <= budget:
            selected.append(unit)
    return _render(selected)


def _merged(selected):
    # Position keys retain page identity and adjacency without merging repeated sentences.
    groups = []
    for page, index, text, score in sorted(selected, key=lambda u: (u[0], u[1])):
        if groups and groups[-1][0] == page and groups[-1][1] + 1 == index:
            p, _, combined, best = groups[-1]
            groups[-1] = (p, index, combined + ' ' + text, max(best, score))
        else:
            groups.append((page, index, text, score))
    return sorted(groups, key=lambda u: u[3], reverse=True)


def _lens_render(selected, titles):
    return '\n\n'.join(f'[{titles[p]}] {text}' for p, _, text, _ in _merged(selected))


def _raw(pages, budget, style=False):
    render = (lambda title, text: f'[{title}]\n{text}') if style else (lambda title, text: f'[{title}] {text}')
    blocks = [render(title, text) for title, text in pages.items()]
    full = '\n\n'.join(blocks)
    if count(full) <= budget:
        return full, False, len(blocks)
    selected = []
    for block in blocks:
        if count('\n\n'.join(selected + [block])) > budget:
            break
        selected.append(block)
    # Prefer whole pages. Only truncate within a page when no whole page fits.
    if not selected and blocks:
        offsets = _tokenizer()(blocks[0], add_special_tokens=False, return_offsets_mapping=True,
                               truncation=False)['offset_mapping']
        end = min(max(0, budget), len(offsets))
        snippet = blocks[0][:offsets[end-1][1]] if end else ''
        while snippet and count(snippet) > budget:
            end -= 1
            snippet = blocks[0][:offsets[end-1][1]] if end else ''
        selected = [snippet] if snippet else []
    return '\n\n'.join(selected), True, len(selected)


def evidence(arm: str, question: str, pages: dict[str, str], budget: int = 2000) -> tuple[str, dict]:
    started = time.perf_counter()
    truncated = False
    if arm == 'raw':
        # The raw arm has a separate registered default cap.
        text, truncated, units = _raw(pages, 150000 if budget == 2000 else budget)
    elif arm == 'lens':
        titles = list(pages)
        candidates, windows = [], []
        for page, content in enumerate(pages.values()):
            split = sentences(content)
            for index, sentence in enumerate(split):
                candidates.append((page, index, sentence))
                windows.append(' '.join(split[max(0, index-1):index+2]))
        scores = rank('maxsim', question, windows, length=128).tolist() if windows else []
        selected = []
        for index in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True):
            candidate = (*candidates[index], float(scores[index]))
            if count(_lens_render(selected + [candidate], titles)) <= budget:
                selected.append(candidate)
        text = _lens_render(selected, titles)
        units = len(selected)
    elif arm in ('rag', 'rerank'):
        candidates = [(title, chunk) for title, content in pages.items() for chunk in chunks(content)]
        scores = rank('single', question, [c[1] for c in candidates], length=256).tolist() if candidates else []
        if arm == 'rerank' and candidates:
            top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:50]
            candidates = [candidates[i] for i in top]
            scores = rank('cross', question, [c[1] for c in candidates], length=512).tolist()
        ranked = sorted([(title, chunk, float(score)) for (title, chunk), score in zip(candidates, scores)],
                        key=lambda u: u[2], reverse=True)
        text = pack(ranked, budget)
        units = len(text.split('\n\n')) if text else 0
    else:
        raise ValueError(f'unknown evidence arm: {arm}')
    tokens = count(text)
    return text, {'ms': (time.perf_counter()-started)*1000, 'tokens': tokens,
                  'units': units, 'truncated': truncated}
