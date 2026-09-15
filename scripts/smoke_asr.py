"""Prove Parakeet loads and runs a forward pass on this GPU.

The point of this script is narrow: CTranslate2-based stacks (faster-whisper,
WhisperX) crash on Blackwell sm_120, so before building anything on top of an
ASR backend we confirm the backend actually executes here. Synthetic audio is
fine for that; it exercises the same CUDA kernels real speech would. Accuracy
has to be judged on a real voice sample.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nvidia/parakeet-tdt-0.6b-v3"
SR = 16_000


def synth(seconds: float = 3.0) -> np.ndarray:
    """Formant-ish noise. Not speech, just something with spectral structure."""
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sig = sum(np.sin(2 * np.pi * f * t) * a for f, a in [(120, 0.4), (700, 0.2), (1220, 0.1)])
    sig *= (1 + 0.5 * np.sin(2 * np.pi * 4 * t))  # syllable-rate envelope
    sig += np.random.default_rng(0).normal(0, 0.01, len(t))
    return (sig / np.abs(sig).max() * 0.6).astype(np.float32)


def main() -> int:
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"device: {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")

    import nemo.collections.asr as nemo_asr

    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s")

    import soundfile as sf
    import tempfile
    import os

    audio = synth(3.0)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, SR)

    try:
        # First call includes lazy CUDA kernel init; second is the real number.
        timings = []
        for i in range(2):
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.transcribe([path], batch_size=1, verbose=False)
            timings.append(time.perf_counter() - t0)
            print(f"  pass {i + 1}: {timings[-1] * 1000:.0f} ms")

        text = out[0].text if hasattr(out[0], "text") else str(out[0])
        print(f"\ntranscript of synthetic audio (garbage expected): {text!r}")
        rtf = timings[-1] / 3.0
        print(f"warm latency {timings[-1] * 1000:.0f} ms for 3.0 s audio  (RTF {rtf:.3f})")
        if torch.cuda.is_available():
            print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print("\nRESULT: PASS - model executes on this GPU")
        return 0
    except Exception as exc:
        print(f"\nRESULT: FAIL - {type(exc).__name__}: {exc}")
        return 1
    finally:
        os.unlink(path)


if __name__ == "__main__":
    raise SystemExit(main())
