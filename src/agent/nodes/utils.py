import re
import json
import threading
import requests
from collections import Counter

from src.utils.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, CLAUDE_MODEL

_local = threading.local()


def node_log(msg: str):
    """Print immediately (real-time) and buffer for batch file write."""
    print(msg, flush=True)
    if not hasattr(_local, "buffer"):
        _local.buffer = []
    _local.buffer.append(msg)


def flush_node_log(qid, log_path="logs/node_trace.log"):
    """Write buffered logs for this task to file, then clear buffer."""
    lines = getattr(_local, "buffer", [])
    if not lines:
        return
    block = f"=== [{qid}] ===\n" + "\n".join(lines) + "\n"
    with _file_write_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(block + "\n")
    _local.buffer = []


_file_write_lock = threading.Lock()


def extract_keywords(text: str, max_nouns=3, max_verbs=3) -> list[str]:
    if not text:
        return []
    keywords = []
    if any(ord(c) >= 0x4E00 for c in text):
        sentences = re.split(r'[，、。；：！？\s]+', text)
        words = [s for s in sentences if len(s) > 0]
        noun_candidates = [w for w in words if 2 <= len(w) <= 4]
        verb_candidates = [w for w in words if len(w) == 2]
        keywords = noun_candidates[:max_nouns] + verb_candidates[:max_verbs]
    else:
        words = re.findall(r'\b[a-z]+\b', text.lower())
        filtered = [w for w in words if len(w) >= 4 and w not in {'the', 'what', 'how', 'where', 'which', 'this', 'that', 'with', 'from', 'have', 'will', 'does', 'when', 'could', 'would', 'should'}]
        if filtered:
            freq = Counter(filtered)
            keywords = [w for w, _ in freq.most_common(max_nouns + max_verbs)]
    return keywords[:max_nouns + max_verbs]


def call_claude(messages: list[dict], system: str | None = None, max_tokens: int = 2048) -> str:
    if system:
        messages = [{"role": "system", "content": system}] + messages
    response = requests.post(
        f"{ANTHROPIC_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {ANTHROPIC_API_KEY}"},
        json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": messages},
        timeout=120
    )
    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")
    try:
        data = json.loads(response.content.decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise Exception(f"Failed to parse response: {e}. Response text: {response.content[:500]}")


def detect_language(text: str) -> str:
    if not text:
        return "en"
    chinese_chars = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    return "zh" if len(text) > 0 and chinese_chars / len(text) > 0.3 else "en"
