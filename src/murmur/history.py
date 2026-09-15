"""Append-only record of every take, for building a correction map later.

Vocabulary misses ("filler" heard as "fellow") are the failure mode that
actually matters, and you cannot fix them by guessing. You need a corpus of
what the model produced so the substitutions can be seeded from real errors
rather than imagined ones.

Privacy: this is a plaintext record of everything dictated. It is written
0600, and `transcript_log = false` turns it off. Note that the daemon already
logs take text to the journal, so disabling this alone does not stop dictated
text reaching disk. See `log_text` in the config for that.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def default_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "murmur" / "takes.jsonl"


def record(
    raw: str,
    cleaned: str,
    audio_s: float,
    latency_ms: int,
    stopped_by: str,
    path: Path | None = None,
) -> None:
    """Append one take. Never raises: losing a log line must not cost a take."""
    path = path or default_path()
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "audio_s": round(audio_s, 2),
        "latency_ms": latency_ms,
        "stopped_by": stopped_by,
        "raw": raw,
        # Only recorded when post-processing actually changed something, so
        # the diff between the two is easy to grep for.
        **({"cleaned": cleaned} if cleaned != raw else {}),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if not existed:
            os.chmod(path, 0o600)
    except OSError:
        pass
