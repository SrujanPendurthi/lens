"""ColPali retrieval over screenshot patches assigned to candidate boxes."""
import math
import os
import time
from functools import cache
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

TILE = 1280
GRID = 32
PATCH = TILE // GRID
SHOTS = Path("data/shots")
CACHE_DIR = Path("data/colpali")
MODEL = "vidore/colpali-v1.3-hf"
DEVICE = os.environ.get("LENS_VISION_DEVICE", "mps")
_stats = {}


def tile_tops(height: int) -> list[int]:
    return list(range(0, height, TILE))


def _rect(box):
    try:
        x, y, w, h = map(float, box)
        if all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0:
            return x, y, w, h
    except (ValueError, TypeError):
        pass
    return None


def box_patches(box: list[float], top: int) -> list[int]:
    rect = _rect(box)
    if rect is None:
        return []
    x, y, w, h = rect
    left, right = max(0., x), min(float(TILE), x + w)
    upper, lower = max(float(top), y), min(float(top + TILE), y + h)
    if right <= left or lower <= upper:
        return []
    return [row * GRID + col
            for row in range(math.floor((upper - top) / PATCH), math.ceil((lower - top) / PATCH))
            for col in range(math.floor(left / PATCH), math.ceil(right / PATCH))]


def load_tiles(uid: str) -> list[Image.Image] | None:
    path = SHOTS / f"{uid}.png"
    if not path.is_file():
        return None
    with Image.open(path) as source:
        image = source.convert("RGB")
    tiles = []
    for top in tile_tops(image.height):
        tile = Image.new("RGB", (TILE, TILE), "white")
        tile.paste(image.crop((0, top, TILE, min(top + TILE, image.height))), (0, 0))
        tiles.append(tile)
    return tiles


@cache
def _load():
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    if DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(min(1., 20 * 2**30 / torch.mps.recommended_max_memory()))
    processor = ColPaliProcessor.from_pretrained(MODEL)
    model = ColPaliForRetrieval.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
    return model, processor


def _inputs(batch, model):
    return {key: value.to(model.device) for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "pixel_values", "token_type_ids"}}


def _sync():
    if DEVICE == "mps":
        torch.mps.synchronize()


def image_embeddings(embeddings, input_ids, image_token_id):
    """Select actual image-token positions, excluding prompt/special tokens."""
    return embeddings[0, input_ids[0] == image_token_id]


def encoding_stats(uid):
    return dict(_stats[uid])


def encode_tiles(uid: str) -> list[torch.Tensor] | None:
    path = SHOTS / f"{uid}.png"
    if not path.is_file():
        _stats[uid] = {"no_screenshot": True, "n_tiles": 0, "encode_s": 0.,
                       "tile_encode_s": [], "cache_hit": False, "device": DEVICE}
        return None
    cached = CACHE_DIR / f"{uid}.pt"
    if cached.is_file():
        payload = torch.load(cached, map_location="cpu", weights_only=True)
        _stats[uid] = payload["stats"] | {"cache_hit": True, "encode_s": 0.}
        return payload["tiles"]
    tiles = load_tiles(uid)
    model, processor = _load()
    embeddings, timings = [], []
    for tile in tiles:
        _sync()
        start = time.perf_counter()
        batch = _inputs(processor.process_images([tile], return_tensors="pt"), model)
        with torch.inference_mode():
            output = model(**batch, use_cache=False)
            patches = image_embeddings(output.embeddings, batch["input_ids"], processor.image_token_id)
            if patches.shape[0] != GRID * GRID:
                raise ValueError(f"expected {GRID * GRID} image tokens; got {patches.shape[0]}")
            patches = F.normalize(patches.float(), dim=-1).cpu()
        _sync()
        timings.append(time.perf_counter() - start)
        embeddings.append(patches)
        del output, batch
    stats = {"no_screenshot": False, "n_tiles": len(tiles), "encode_s": sum(timings),
             "tile_encode_s": timings, "cache_hit": False, "device": DEVICE}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"tiles": embeddings, "stats": stats}, cached)
    _stats[uid] = stats
    return embeddings


def encode_query(query: str) -> torch.Tensor:
    model, processor = _load()
    batch = _inputs(processor.process_queries([query], return_tensors="pt", padding="max_length",
                                             max_length=50, truncation=True), model)
    with torch.inference_mode():
        embeddings = model(**batch, use_cache=False).embeddings
        q = embeddings[0, batch["attention_mask"][0].bool()].float().cpu()
    return F.normalize(q, dim=-1)


def rank_colpali(query: str, uid: str, refs: list[str], boxes: dict[str, list[float]]) -> torch.Tensor:
    scores = torch.full((len(refs),), float("-inf"))
    tiles = encode_tiles(uid)
    if tiles is None:
        return scores
    with Image.open(SHOTS / f"{uid}.png") as image:
        width, height = min(image.width, TILE), image.height
    q = encode_query(query)
    similarities = [q @ tile.T for tile in tiles]
    for index, ref in enumerate(refs):
        rect = _rect(boxes.get(ref))
        if rect is None:
            continue
        x, y, w, h = rect
        left, right = max(0., x), min(float(width), x + w)
        upper, lower = max(0., y), min(float(height), y + h)
        if left >= right or upper >= lower:
            continue
        box = [left, upper, right - left, lower - upper]
        best = None
        for tile_index, top in enumerate(tile_tops(height)):
            indices = box_patches(box, top)
            if indices:
                maxima = similarities[tile_index][:, indices].max(dim=1).values
                best = maxima if best is None else torch.maximum(best, maxima)
        if best is not None:
            scores[index] = best.sum()
    return scores
