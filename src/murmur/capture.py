"""Microphone capture with a Silero VAD backstop.

Threading contract, which is the part that is easy to get wrong: the PortAudio
callback runs on a real-time-ish audio thread and does exactly one thing, copy
the buffer onto a queue. All VAD inference, accumulation and stop-condition
logic happens on a normal worker thread draining that queue. Nothing mutable is
shared between the two except the queue itself and a threading.Event.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

# Silero operates on fixed 512-sample windows at 16 kHz (32 ms).
VAD_WINDOW = 512

# How much of the trailing silence to cut when the VAD ends a take. Not all of
# it: models want a little room after the final word, and cutting flush to the
# last speech frame clips plosives.
TRIM_FRACTION = 0.8


def trim_trailing_silence(
    audio: np.ndarray,
    silence_timeout: float,
    sample_rate: int,
    fraction: float = TRIM_FRACTION,
) -> np.ndarray:
    """Drop most of the silence the VAD waited through before stopping.

    Without this the model decodes ~`silence_timeout` seconds of nothing on
    every VAD-terminated take. Returns the input unchanged when there is not
    enough audio to trim safely.
    """
    if fraction <= 0 or silence_timeout <= 0 or audio.size == 0:
        return audio
    cut = int(silence_timeout * sample_rate * fraction)
    if cut <= 0 or audio.size <= cut:
        return audio
    return audio[:-cut]


@dataclass
class Capture:
    """One completed dictation take."""

    audio: np.ndarray  # float32 mono at 16 kHz, range [-1, 1]
    seconds: float
    stopped_by: str  # "user" | "silence" | "no_speech" | "max_seconds" | "error"
    # Populated only when stopped_by == "error". Carried on the result rather
    # than just logged, so the daemon can report the real cause instead of a
    # generic "no audio captured".
    error: str = ""


class Recorder:
    """Records until stopped by the user, by trailing silence, or by the cap.

    `silence_timeout` only arms once speech has actually been detected, so a
    slow start (thinking before talking) does not immediately cancel the take.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        device: str | int | None = None,
        silence_timeout: float = 1.6,
        max_seconds: int = 300,
        speech_threshold: float = 0.5,
        no_speech_timeout: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.silence_timeout = silence_timeout
        self.max_seconds = max_seconds
        self.speech_threshold = speech_threshold
        # `silence_timeout` only arms after speech is heard, so without this a
        # toggle you never speak into would record until `max_seconds`.
        self.no_speech_timeout = no_speech_timeout

        self._vad = None
        self._stop = threading.Event()
        self._result: Capture | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def load_vad(self) -> None:
        """Load Silero once, at daemon start, not per take."""
        if self._vad is not None:
            return
        from silero_vad import load_silero_vad

        self._vad = load_silero_vad()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("already recording")
        self.load_vad()
        self._stop.clear()
        self._result = None
        self._thread = threading.Thread(target=self._run, name="murmur-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> Capture | None:
        """Ask the worker to finish and return the take."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        return self._result

    @property
    def recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        import sounddevice as sd
        import torch

        frames: queue.Queue[np.ndarray] = queue.Queue()

        def on_audio(indata, _frames, _time, status):
            # Audio thread. Copy and hand off; never allocate models, never
            # touch disk, never take a lock that another thread might hold.
            if status:
                pass  # over/underruns are reported on the take, not here
            frames.put(indata[:, 0].copy())

        collected: list[np.ndarray] = []
        pending = np.empty(0, dtype=np.float32)
        speech_seen = False
        last_speech = time.monotonic()
        started = time.monotonic()
        reason = "user"
        error = ""

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=VAD_WINDOW,
                device=self.device,
                callback=on_audio,
            ):
                while not self._stop.is_set():
                    try:
                        block = frames.get(timeout=0.1)
                    except queue.Empty:
                        block = None

                    if block is not None:
                        collected.append(block)
                        pending = np.concatenate([pending, block])

                        # Feed VAD in exact 512-sample windows.
                        while len(pending) >= VAD_WINDOW:
                            window, pending = pending[:VAD_WINDOW], pending[VAD_WINDOW:]
                            prob = float(
                                self._vad(torch.from_numpy(window), self.sample_rate).item()
                            )
                            if prob >= self.speech_threshold:
                                speech_seen = True
                                last_speech = time.monotonic()

                    now = time.monotonic()
                    if speech_seen and (now - last_speech) >= self.silence_timeout:
                        reason = "silence"
                        break
                    if not speech_seen and (now - started) >= self.no_speech_timeout:
                        reason = "no_speech"
                        break
                    if (now - started) >= self.max_seconds:
                        reason = "max_seconds"
                        break
        except Exception as exc:  # capture failures must not kill the daemon
            error = f"{type(exc).__name__}: {exc}"
            print(f"murmur: capture failed: {error}", flush=True)
            reason = "error"

        audio = (
            np.concatenate(collected).astype(np.float32)
            if collected
            else np.empty(0, dtype=np.float32)
        )
        if reason == "silence":
            audio = trim_trailing_silence(audio, self.silence_timeout, self.sample_rate)

        self._result = Capture(
            audio=audio,
            seconds=len(audio) / self.sample_rate,
            stopped_by=reason,
            error=error,
        )
