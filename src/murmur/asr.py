"""Parakeet TDT via NeMo, held resident on the GPU.

Why NeMo and not faster-whisper: this machine has an RTX 5080 (Blackwell,
sm_120). CTranslate2 does not support sm_120 INT8 tensor-core ops, so
faster-whisper and WhisperX crash with CUBLAS_STATUS_NOT_SUPPORTED on their
default compute type. NeMo is plain PyTorch and runs fine here.

Why Parakeet and not Whisper: Whisper pads every input to a fixed 30-second
window, so a four-second dictation costs nearly as much as a thirty-second one.
For short utterances, which is all dictation ever is, that dominates latency.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


class ASRError(RuntimeError):
    pass


class ParakeetEngine:
    """Loads once, transcribes many times. Not thread-safe by design: the
    daemon serialises takes, and two concurrent decodes on one GPU would only
    make both slower."""

    def __init__(self, model_name: str, device: str = "cuda", scratch: Path | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        # Decode scratch space. XDG_RUNTIME_DIR is tmpfs, so the temp wav never
        # touches a real disk.
        self._scratch = scratch or Path(os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir()))

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> float:
        """Pull weights onto the GPU. Returns seconds taken."""
        if self._model is not None:
            return 0.0
        import torch
        import nemo.collections.asr as nemo_asr

        t0 = time.perf_counter()
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        model.eval()
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise ASRError("device=cuda but torch reports no CUDA device")
            model = model.cuda()
        self._model = model
        return time.perf_counter() - t0

    def transcribe(self, audio, sample_rate: int = 16_000) -> str:
        """Decode a float32 mono numpy array to text."""
        if self._model is None:
            raise ASRError("engine used before load()")
        if audio is None or len(audio) == 0:
            return ""

        import numpy as np
        import soundfile as sf
        import torch

        audio = np.asarray(audio, dtype=np.float32)

        # NeMo's transcribe() signature has moved around across releases. Try
        # the in-memory path, fall back to a tmpfs wav rather than guessing.
        fd, path = tempfile.mkstemp(suffix=".wav", dir=str(self._scratch))
        os.close(fd)
        try:
            sf.write(path, audio, sample_rate)
            with torch.inference_mode():
                out = self._model.transcribe([path], batch_size=1, verbose=False)
        except Exception as exc:
            raise ASRError(f"decode failed: {exc}") from exc
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

        if not out:
            return ""
        first = out[0]
        text = getattr(first, "text", None)
        if text is None:
            text = first if isinstance(first, str) else str(first)
        return text.strip()

    def warm(self) -> float:
        """Decode a fraction of a second of silence to force lazy CUDA kernel
        init. Without this the first real take after every daemon start pays
        ~800 ms instead of the ~45 ms every later take costs. Returns seconds.

        Raises ASRError if that decode fails. This used to be swallowed as
        best-effort, which meant a daemon whose CUDA context was poisoned at
        startup still logged "ready" and sat there looking healthy until the
        first real take came back empty. A warmup that cannot decode silence
        will not decode speech either, so it is worth failing loudly: the
        controller treats it as a fatal start and systemd restarts the unit.
        """
        import numpy as np

        t0 = time.perf_counter()
        self.transcribe(np.zeros(8000, dtype=np.float32), 16_000)
        return time.perf_counter() - t0

    def release(self) -> None:
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
