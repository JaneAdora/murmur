"""Resident dictation daemon.

The whole point of a daemon is that the model stays warm. Loading Parakeet is
the expensive step, and tools that load per utterance are why local dictation
on Linux feels sluggish regardless of how fast the GPU is.

Design: socket handlers never do work. They push a command onto a queue and
reply immediately, so the keyboard shortcut always returns instantly. A single
controller thread owns the recorder and the engine and processes commands
serially, which removes the need for locking around any of that state. The one
piece of genuinely shared state, the status snapshot, sits behind a lock and is
only ever whole-object replaced.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .asr import ASRError, ParakeetEngine
from .capture import Recorder
from .config import Config, SOCKET_PATH
from .inject import InjectionError, inject
from .history import record
from .postprocess import clean


@dataclass(frozen=True)
class Status:
    state: str = "starting"  # starting | idle | recording | transcribing
    model: str = ""
    model_loaded: bool = False
    last_text: str = ""
    last_latency_ms: int = 0
    last_audio_s: float = 0.0
    last_stopped_by: str = ""
    last_error: str = ""
    takes: int = 0


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} murmur: {msg}", flush=True)


class Daemon:
    def __init__(self, cfg: Config, socket_path: Path = SOCKET_PATH) -> None:
        self.cfg = cfg
        self.socket_path = socket_path
        self.engine = ParakeetEngine(cfg.model, cfg.device)
        self.recorder = Recorder(
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
            device=cfg.source or None,
            silence_timeout=cfg.silence_timeout,
            max_seconds=cfg.max_seconds,
            no_speech_timeout=cfg.no_speech_timeout,
        )
        self._commands: queue.Queue[str] = queue.Queue()
        # Set when the current take was started by `dry`: transcribe and report,
        # but do not inject. Only ever touched by the controller thread.
        self._dry = False
        self._shutdown = threading.Event()
        self._status = Status(model=cfg.model)
        self._status_lock = threading.Lock()

    # -- status ------------------------------------------------------------

    def status(self) -> Status:
        with self._status_lock:
            return self._status

    def _set(self, **fields) -> None:
        with self._status_lock:
            self._status = replace(self._status, **fields)

    # -- controller --------------------------------------------------------

    def _controller(self) -> None:
        """Owns the recorder and engine. Serial by construction."""
        try:
            log(f"loading {self.cfg.model} on {self.cfg.device}")
            took = self.engine.load()
            self.recorder.load_vad()
            warmed = self.engine.warm()
            self._set(state="idle", model_loaded=True)
            log(
                f"ready in {took:.1f}s (+{warmed:.1f}s warmup) "
                "- model resident, waiting for toggle"
            )
        except Exception as exc:
            self._set(state="idle", last_error=f"model load failed: {exc}")
            log(f"FATAL: model load failed: {exc}")
            self._shutdown.set()
            return

        while not self._shutdown.is_set():
            try:
                cmd = self._commands.get(timeout=0.25)
            except queue.Empty:
                # The VAD backstop stops the recorder on its own; notice that
                # and finish the take without waiting for another command.
                if self.status().state == "recording" and not self.recorder.recording:
                    self._finish_take()
                continue

            if cmd == "toggle":
                self._begin_take() if self.status().state == "idle" else self._finish_take()
            elif cmd == "dry" and self.status().state == "idle":
                self._begin_take(dry=True)
            elif cmd == "start" and self.status().state == "idle":
                self._begin_take()
            elif cmd == "stop" and self.status().state == "recording":
                self._finish_take()
            elif cmd == "quit":
                break

        if self.recorder.recording:
            self.recorder.stop()
        log("controller stopped")

    def _begin_take(self, dry: bool = False) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            self._set(last_error=f"capture start failed: {exc}")
            log(f"capture start failed: {exc}")
            return
        self._dry = dry
        self._set(state="recording", last_error="")
        log("recording (dry - will not inject)" if dry else "recording")

    def _finish_take(self) -> None:
        self._set(state="transcribing")
        take = self.recorder.stop()
        if take is None:
            self._set(state="idle", last_error="recorder returned nothing")
            log("recorder returned nothing")
            return
        if take.stopped_by == "error":
            # Surface the real cause. Reporting "no audio captured" here once
            # hid a missing PortAudio library behind a generic message.
            self._set(state="idle", last_stopped_by="error", last_error=take.error)
            log(f"capture error: {take.error}")
            return
        if take.seconds <= 0:
            self._set(state="idle", last_stopped_by=take.stopped_by,
                      last_error="no audio captured")
            log("no audio captured")
            return

        t0 = time.perf_counter()
        try:
            text = self.engine.transcribe(take.audio, self.cfg.sample_rate)
        except ASRError as exc:
            self._set(state="idle", last_error=str(exc))
            log(f"transcribe failed: {exc}")
            return
        raw = text
        text = clean(text, self.cfg.strip_fillers)
        latency = int((time.perf_counter() - t0) * 1000)

        if self.cfg.transcript_log:
            record(raw, text, take.seconds, latency, take.stopped_by)

        if text and not self._dry:
            try:
                inject(text, self.cfg.inject)
            except InjectionError as exc:
                self._set(last_error=f"inject failed: {exc}")
                log(f"inject failed: {exc} - text was: {text!r}")
        self._dry = False

        self._set(
            state="idle",
            last_text=text,
            last_latency_ms=latency,
            last_audio_s=round(take.seconds, 2),
            last_stopped_by=take.stopped_by,
            takes=self.status().takes + 1,
        )
        shown = f" -> {text[:70]!r}" if self.cfg.log_text else ""
        log(
            f"{take.seconds:.1f}s audio -> {latency} ms "
            f"(stopped by {take.stopped_by}){shown}"
        )

    # -- socket ------------------------------------------------------------

    def _serve(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        # Owner-only: this socket injects keystrokes into the session.
        os.chmod(self.socket_path, 0o600)
        srv.listen(8)
        srv.settimeout(0.5)
        log(f"listening on {self.socket_path}")

        while not self._shutdown.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                try:
                    raw = conn.recv(4096).decode().strip()
                    reply = self._handle(raw)
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except Exception as exc:
                    log(f"socket error: {exc}")
        srv.close()
        self.socket_path.unlink(missing_ok=True)

    def _handle(self, raw: str) -> dict:
        cmd = (raw or "").strip().lower()
        if cmd == "status":
            return {"ok": True, **asdict(self.status())}
        if cmd in ("toggle", "dry", "start", "stop", "quit"):
            self._commands.put(cmd)
            if cmd == "quit":
                self._shutdown.set()
            return {"ok": True, "accepted": cmd, "state": self.status().state}
        return {"ok": False, "error": f"unknown command {cmd!r}"}

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> int:
        def handle_signal(signum, _frame):
            log(f"caught {signal.Signals(signum).name} - shutting down")
            self._shutdown.set()
            self._commands.put("quit")

        # Both, not just SIGINT: systemd sends SIGTERM on stop and logout.
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        controller = threading.Thread(target=self._controller, name="murmur-ctl", daemon=True)
        controller.start()
        try:
            self._serve()
        finally:
            self._shutdown.set()
            self._commands.put("quit")
            controller.join(timeout=10)
            self.socket_path.unlink(missing_ok=True)
        return 0


def send(command: str, socket_path: Path = SOCKET_PATH, timeout: float = 5.0) -> dict:
    """Client side: one command, one reply."""
    if not socket_path.exists():
        return {"ok": False, "error": "daemon not running"}
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(socket_path))
            sock.sendall(command.encode())
            return json.loads(sock.recv(8192).decode() or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}
