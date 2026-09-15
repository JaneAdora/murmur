"""Put recognised text into whatever window has focus.

Backends, in order of preference:

  wtype      zwp_virtual_keyboard_v1. Types the text as key events. Confirmed
             available on this cosmic-comp build. Does not touch the clipboard.
  clipboard  wl-copy plus a synthesised ctrl+v. Fast for long text, but it
             clobbers the clipboard, which fights Jane's clipboard-picker.
             Only used if explicitly configured.

A third backend, zwp_input_method_v2 `commit_string`, is the correct long-term
answer: it inserts text directly into the focused field with no keycode or
layout translation, and `set_preedit_string` would let text appear live while
speaking. cosmic-comp advertises `zwp_input_method_manager_v2` (confirmed via
wayland-info), but it needs a real Wayland client rather than a subprocess, so
it is deliberately left as the next step rather than a blocker.
"""

from __future__ import annotations

import shutil
import subprocess


class InjectionError(RuntimeError):
    pass


def _run(argv: list[str], stdin: str | None = None) -> None:
    try:
        proc = subprocess.run(
            argv,
            input=stdin.encode() if stdin is not None else None,
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise InjectionError(f"{argv[0]} not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise InjectionError(f"{argv[0]} timed out") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise InjectionError(f"{argv[0]} exited {proc.returncode}: {err}")


def inject_wtype(text: str) -> None:
    # `--` so text beginning with a dash is not parsed as a flag.
    _run(["wtype", "--", text])


def inject_clipboard(text: str) -> None:
    _run(["wl-copy", "--"], stdin=text)
    _run(["wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"])


BACKENDS = {
    "wtype": inject_wtype,
    "clipboard": inject_clipboard,
    "none": lambda text: None,
}


def inject(text: str, backend: str = "wtype", trailing_space: bool = True) -> None:
    """Insert `text` at the cursor. No-op for empty or whitespace-only text.

    One space is appended by default. Dictation comes in bursts, and a take is
    nearly always followed by another, so without it consecutive takes arrive
    welded together ("let it go.And then") and a space has to be typed by hand
    between every pair.

    The blank-text guard runs first, so a take that decodes to nothing still
    injects nothing rather than a stray space. Text that already ends in
    whitespace is left alone.
    """
    if not text.strip():
        return
    fn = BACKENDS.get(backend)
    if fn is None:
        raise InjectionError(f"unknown inject backend {backend!r}")
    if trailing_space and not text[-1].isspace():
        text += " "
    fn(text)


def available(backend: str) -> tuple[bool, str]:
    """Whether a backend's binaries exist. Used by `murmur doctor`."""
    if backend == "none":
        return True, "no-op"
    needed = {"wtype": ["wtype"], "clipboard": ["wl-copy", "wtype"]}.get(backend, [])
    missing = [b for b in needed if shutil.which(b) is None]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "ok"
