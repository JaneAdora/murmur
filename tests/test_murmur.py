"""Tests for the pure logic. The audio thread and the GPU decode are covered by
`murmur doctor` and the smoke script instead, since neither is meaningfully
unit-testable without hardware."""

from __future__ import annotations

import json
import socket
import threading

import numpy as np
import pytest

from murmur.asr import ASRError, ParakeetEngine
from murmur.capture import TRIM_FRACTION, trim_trailing_silence
from murmur.config import Config
from murmur.inject import InjectionError, available, inject


# -- config ----------------------------------------------------------------


def test_config_defaults_when_missing(tmp_path):
    cfg = Config.load(tmp_path / "nope.toml")
    assert cfg.model.startswith("nvidia/parakeet")
    assert cfg.device == "cuda"
    assert cfg.sample_rate == 16_000


def test_config_reads_values(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('device = "cpu"\nsilence_timeout = 2.5\ninject = "clipboard"\n')
    cfg = Config.load(p)
    assert (cfg.device, cfg.silence_timeout, cfg.inject) == ("cpu", 2.5, "clipboard")


def test_bad_toml_falls_back_to_defaults_not_crash(tmp_path, capsys):
    p = tmp_path / "config.toml"
    p.write_text("this is not = = toml")
    cfg = Config.load(p)
    assert cfg.device == "cuda"
    assert "ignoring bad config" in capsys.readouterr().out


def test_unknown_keys_are_reported_and_dropped(tmp_path, capsys):
    p = tmp_path / "config.toml"
    p.write_text('device = "cpu"\nnonsense = 1\n')
    cfg = Config.load(p)
    assert cfg.device == "cpu"
    assert "unknown config keys" in capsys.readouterr().out


# -- silence trimming ------------------------------------------------------


def test_trim_removes_expected_fraction():
    sr, timeout = 16_000, 1.6
    audio = np.ones(sr * 5, dtype=np.float32)
    out = trim_trailing_silence(audio, timeout, sr)
    assert len(out) == len(audio) - int(timeout * sr * TRIM_FRACTION)


def test_trim_leaves_short_audio_alone():
    """A take shorter than the trim window must not become empty."""
    sr = 16_000
    audio = np.ones(1000, dtype=np.float32)
    assert len(trim_trailing_silence(audio, 1.6, sr)) == 1000


def test_trim_handles_empty_and_zero_timeout():
    sr = 16_000
    assert trim_trailing_silence(np.empty(0, dtype=np.float32), 1.6, sr).size == 0
    audio = np.ones(sr, dtype=np.float32)
    assert len(trim_trailing_silence(audio, 0.0, sr)) == sr


# -- injection -------------------------------------------------------------


def test_inject_ignores_blank_text(monkeypatch):
    called = []
    monkeypatch.setitem(
        __import__("murmur.inject", fromlist=["BACKENDS"]).BACKENDS,
        "wtype",
        lambda t: called.append(t),
    )
    inject("", "wtype")
    inject("   \n ", "wtype")
    assert called == []


def test_inject_rejects_unknown_backend():
    with pytest.raises(InjectionError):
        inject("hello", "telepathy")


def test_available_reports_missing_binaries():
    ok, detail = available("none")
    assert ok and detail == "no-op"


# -- daemon protocol -------------------------------------------------------


def test_daemon_handle_rejects_unknown_command():
    from murmur.daemon import Daemon

    d = Daemon(Config(), socket_path=None)  # never served; _handle is pure
    assert d._handle("frobnicate")["ok"] is False


def test_daemon_handle_accepts_known_commands():
    from murmur.daemon import Daemon

    d = Daemon(Config(), socket_path=None)
    for cmd in ("toggle", "start", "stop"):
        reply = d._handle(cmd)
        assert reply["ok"] and reply["accepted"] == cmd
    assert d._commands.qsize() == 3


def test_status_command_returns_snapshot():
    from murmur.daemon import Daemon

    d = Daemon(Config(), socket_path=None)
    reply = d._handle("status")
    assert reply["ok"] and reply["state"] == "starting"


def test_send_reports_when_daemon_absent(tmp_path):
    from murmur.daemon import send

    reply = send("status", socket_path=tmp_path / "absent.sock")
    assert reply["ok"] is False and "not running" in reply["error"]


# -- post-processing -------------------------------------------------------

from murmur.postprocess import clean, strip_fillers


def test_strips_real_world_take():
    """The literal first sentence murmur ever transcribed from Jane's voice."""
    got = strip_fillers(
        "Well it seems to be working. Um I guess if you're getting this "
        "message it worked."
    )
    assert got == (
        "Well it seems to be working. I guess if you're getting this "
        "message it worked."
    )


def test_recapitalises_after_sentence_initial_filler():
    assert strip_fillers("Um i think so.") == "I think so."
    assert strip_fillers("So. Uh maybe not.") == "So. Maybe not."


def test_strips_mid_sentence_filler_and_its_comma():
    assert strip_fillers("It was, um, fine.") == "It was, fine."


def test_does_not_touch_words_containing_fillers():
    for phrase in ("Umbrella policy.", "Uhura called.", "The summer hums."):
        assert strip_fillers(phrase) == phrase


def test_pure_filler_utterance_survives():
    """Never return empty when the model did hear something."""
    assert strip_fillers("Um.") == "Um."
    assert strip_fillers("uh") == "uh"


def test_clean_respects_the_opt_out():
    text = "Um I guess."
    assert clean(text, strip_filler_words=False) == text
    assert clean(text, strip_filler_words=True) == "I guess."


def test_clean_handles_empty():
    assert clean("", True) == ""
    assert clean("   ", True) == ""


# -- take history ----------------------------------------------------------

from murmur.history import record


def test_record_appends_one_line_per_take(tmp_path):
    p = tmp_path / "takes.jsonl"
    record("Um hello.", "Hello.", 2.0, 40, "silence", path=p)
    record("Second.", "Second.", 1.0, 30, "user", path=p)
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["raw"] == "Um hello." and first["cleaned"] == "Hello."
    assert first["stopped_by"] == "silence"


def test_record_omits_cleaned_when_unchanged(tmp_path):
    p = tmp_path / "takes.jsonl"
    record("No fillers here.", "No fillers here.", 1.0, 20, "user", path=p)
    assert "cleaned" not in json.loads(p.read_text().strip())


def test_record_is_written_owner_only(tmp_path):
    p = tmp_path / "takes.jsonl"
    record("x", "x", 1.0, 10, "user", path=p)
    assert oct(p.stat().st_mode)[-3:] == "600"


def test_record_never_raises_on_bad_path(tmp_path):
    bad = tmp_path / "nope"
    bad.write_text("i am a file, not a directory")
    record("x", "x", 1.0, 10, "user", path=bad / "takes.jsonl")  # must not raise


# -- asr engine -------------------------------------------------------------


class _RaisingModel:
    """Stands in for the NeMo model when the GPU context is unusable."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def transcribe(self, *a, **kw):
        raise self.exc


def _engine(tmp_path, model):
    eng = ParakeetEngine("nvidia/parakeet-tdt-0.6b-v3", "cuda", scratch=tmp_path)
    eng._model = model
    return eng


def test_warmup_failure_propagates(tmp_path):
    """A warmup that cannot decode means the GPU context is dead. Swallowing it
    lets the daemon announce itself ready with a model it cannot run."""
    eng = _engine(tmp_path, _RaisingModel(RuntimeError("CUDA error: unknown error")))
    with pytest.raises(ASRError, match="unknown error"):
        eng.warm()


def test_warmup_returns_elapsed_on_success(tmp_path):
    class _Ok:
        def transcribe(self, *a, **kw):
            return ["quiet"]

    assert _engine(tmp_path, _Ok()).warm() >= 0.0


def test_failed_decode_releases_vram(tmp_path, monkeypatch):
    """A decode that raises must not strand its allocations on the GPU."""
    eng = _engine(tmp_path, _RaisingModel(RuntimeError("CUDA error: unknown error")))
    freed = []
    monkeypatch.setattr(eng, "_empty_cache", lambda: freed.append(True))
    with pytest.raises(ASRError):
        eng.transcribe(np.zeros(8000, dtype=np.float32), 16_000)
    assert freed == [True], "failed decode did not release cached VRAM"


def test_successful_decode_does_not_empty_cache(tmp_path, monkeypatch):
    """empty_cache() on the happy path would throw away the allocator's blocks
    and make the next take slower for no reason."""

    class _Ok:
        def transcribe(self, *a, **kw):
            return ["hello"]

    eng = _engine(tmp_path, _Ok())
    freed = []
    monkeypatch.setattr(eng, "_empty_cache", lambda: freed.append(True))
    assert eng.transcribe(np.zeros(8000, dtype=np.float32), 16_000) == "hello"
    assert freed == []


# -- daemon start failure ---------------------------------------------------


def _dead_gpu_daemon(monkeypatch):
    from murmur.daemon import Daemon

    d = Daemon(Config(), socket_path=None)

    def boom():
        raise ASRError("decode failed: CUDA error: unknown error")

    monkeypatch.setattr(d.engine, "load", lambda: 0.0)
    monkeypatch.setattr(d.engine, "warm", boom)
    monkeypatch.setattr(d.recorder, "load_vad", lambda: None)
    return d


def test_fatal_start_exits_nonzero(monkeypatch):
    """Restart=on-failure only fires on a non-zero exit. A daemon that gives up
    at startup and exits 0 looks to systemd like a clean shutdown, so it is
    never retried and dictation stays dead until someone notices."""
    d = _dead_gpu_daemon(monkeypatch)
    d._controller()
    assert d._start_failed.is_set()
    assert d.exit_code == 1


def test_clean_shutdown_exits_zero(monkeypatch):
    from murmur.daemon import Daemon

    d = Daemon(Config(), socket_path=None)
    monkeypatch.setattr(d.engine, "load", lambda: 0.0)
    monkeypatch.setattr(d.engine, "warm", lambda: 0.0)
    monkeypatch.setattr(d.recorder, "load_vad", lambda: None)
    d._shutdown.set()  # exit the controller loop immediately
    d._controller()
    assert not d._start_failed.is_set()
    assert d.exit_code == 0
