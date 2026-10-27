import hashlib
import json
import re
import subprocess
import time
from functools import cache
from pathlib import Path

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
HAIKU = "claude-haiku-5-5"
CACHE = Path(__file__).parent.parent / "data" / "llm_cache.jsonl"
_memo = {}


@cache
def _load():
    import mlx.core as mx
    from mlx_lm import load
    mx.set_memory_limit(20 * 2**30)  # .gpu/RULES.md cap
    return load(MODEL)


def prompt(task, prev, nodes):
    return (f"Task: {task}\nPrevious actions: {'; '.join(prev) or 'none'}\n"
            "Elements on the page:\n" + "\n".join(n.line() for n in nodes) +
            "\n\nWhich element should be acted on next? Reply with only its number in brackets, e.g. [123], or [none].")


def _cached(key, fn):
    if not _memo and CACHE.exists():
        with CACHE.open() as f:
            _memo.update((r["key"], r) for r in map(json.loads, f))
    if key in _memo:
        return _memo[key]
    row = fn() | {"key": key}
    with CACHE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    _memo[key] = row
    return row


def pick_key(text):
    return hashlib.sha1(f"pick-v2\n{HAIKU}\n{text}".encode()).hexdigest()


def subgoal(task, prev):
    """Generate the next sub-goal; cache its text, total tokens, and generation ms."""
    text = (f"Task: {task}\nPrevious actions: {'; '.join(prev) or 'none'}\n"
            "In one short line, what is the next thing to do on the page? Name the element and the action, "
            'e.g. "click the Search button" or "type Boston into the location box".')
    key = hashlib.sha1(f"subgoal-v1\n{MODEL}\n{text}".encode()).hexdigest()
    generated = False

    def generate():
        nonlocal generated
        generated = True
        from mlx_lm import stream_generate
        model, tok = _load()
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        t0 = time.perf_counter()
        out = ""
        for r in stream_generate(model, tok, ids, max_tokens=24):
            out += r.text
        row = {"key": key, "text": out.strip().strip('"\'').strip(),
               "tokens": r.prompt_tokens + r.generation_tokens, "ms": (time.perf_counter() - t0) * 1000}
        return row

    row = _cached(key, generate)
    return row | {"miss": True} if generated else row

def parse_ref(out):
    match = re.search(r"\[(\d+|none)\]", out, re.IGNORECASE)
    return match.group(1).lower() if match else None


def parse_claude_json(stdout):
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Claude CLI returned invalid JSON") from exc
    if data.get("is_error") or not isinstance(data.get("result"), str):
        raise RuntimeError(f"Claude CLI error: {str(data.get('result', data))[-500:]}")
    usage = data.get("usage", {})
    return {"out": data["result"], "prompt_tokens": sum(usage.get(key, 0) for key in
            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
            "ttft": data["duration_api_ms"] / 1000}


def pick(text):
    """Return a cached Haiku element choice for one prompt."""
    key = pick_key(text)

    def generate():
        response = subprocess.run([
            "claude", "-p", "--model", HAIKU, "--tools", "", "--strict-mcp-config",
            "--setting-sources", "", "--no-session-persistence", "--safe-mode",
            "--disable-slash-commands", "--output-format", "json",
            "--system-prompt", "You pick one web page element. Reply with only its number in brackets, e.g. [123], or [none].",
        ], input=text, capture_output=True, text=True, timeout=120, cwd="/private/tmp")
        if response.returncode:
            raise RuntimeError(response.stderr[-500:])
        row = parse_claude_json(response.stdout)
        return row | {"ref": parse_ref(row["out"])}

    return _cached(key, generate)
