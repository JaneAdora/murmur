"""User config at ~/.config/murmur/config.toml. Every key is optional.

    model = "nvidia/parakeet-tdt-0.6b-v3"
    device = "cuda"
    source = ""              # PipeWire source name; empty = system default
    silence_timeout = 1.6    # seconds of silence that auto-stops a toggled session
    no_speech_timeout = 10.0 # give up if you toggle on and never speak
    max_seconds = 300        # hard cap so a forgotten session can't run forever
    inject = "wtype"         # "wtype" | "clipboard" | "none"
    strip_fillers = true     # drop standalone "um"/"uh" before injecting
    transcript_log = true    # append every take to ~/.local/share/murmur/takes.jsonl
    log_text = true          # include take text in the journal

Parsed once at daemon start. Deliberately not re-read per call: a config that
changes underneath a running session gives you incoherent behaviour for no
benefit, and this daemon is cheap to restart.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(
    os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
) / "murmur" / "config.toml"


def runtime_dir() -> Path:
    """Where the control socket lives. Falls back to /tmp if XDG is unset."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    return Path(base) if base else Path("/tmp")


SOCKET_PATH = runtime_dir() / "murmur.sock"


@dataclass(frozen=True)
class Config:
    model: str = "nvidia/parakeet-tdt-0.6b-v3"
    device: str = "cuda"
    source: str = ""
    silence_timeout: float = 1.6
    no_speech_timeout: float = 10.0
    max_seconds: int = 300
    inject: str = "wtype"
    strip_fillers: bool = True
    transcript_log: bool = True
    log_text: bool = True

    # Capture is fixed at what the ASR model expects. Exposed as constants
    # rather than config because changing them silently breaks recognition.
    sample_rate: int = 16_000
    channels: int = 1

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()
        try:
            raw = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            # Warn rather than silently defaulting. Dictation quietly going to
            # the wrong device or engine is worse than a noisy startup.
            print(f"murmur: ignoring bad config {path}: {exc}", flush=True)
            return cls()

        known = {f for f in cls.__dataclass_fields__ if f not in ("sample_rate", "channels")}
        unknown = set(raw) - known
        if unknown:
            print(f"murmur: unknown config keys ignored: {sorted(unknown)}", flush=True)
        return cls(**{k: v for k, v in raw.items() if k in known})
