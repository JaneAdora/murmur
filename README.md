# murmur

Local, low-latency dictation for COSMIC on Wayland. Press Super+D, talk, press
it again (or just stop talking). Text appears in whatever field has focus.
Nothing leaves the machine.

Built for muthur specifically: RTX 5080, Pop!_OS 24.04, cosmic-comp.

## Why it's built this way

**A resident daemon, not a per-utterance process.** Loading Parakeet takes ~11
seconds from a warm HuggingFace cache. Tools that load per utterance are the
reason local dictation on Linux feels bad, and it has nothing to do with model
quality. murmur loads once at login and stays warm, so a take costs ~45 ms.

**Parakeet, not Whisper.** Whisper pads every input to a fixed 30-second
window, so a four-second dictation costs nearly as much as a thirty-second one.
For short utterances, which is all dictation ever is, that padding dominates.
Parakeet also declines to hallucinate on non-speech, where Whisper famously
invents subtitles for silence.

**NeMo, not faster-whisper.** This machine is Blackwell (sm_120). CTranslate2
does not support sm_120 INT8 tensor-core ops, so faster-whisper and WhisperX
crash with `CUBLAS_STATUS_NOT_SUPPORTED` on their default compute type. Most
"just use faster-whisper" advice predates these cards. NeMo is plain PyTorch and
runs fine.

**wtype, not ydotool.** cosmic-comp advertises `zwp_virtual_keyboard_v1`
(confirmed with `wayland-info`), so there's no need to drop to kernel uinput.
This also avoids clobbering the clipboard, which would fight the clipboard
picker on Super+V.

## Install

Already installed on muthur. From scratch:

```sh
sudo apt install wtype libportaudio2 wayland-utils
cd ~/projects/murmur
uv venv --python 3.11 --system-site-packages .venv
uv pip install --python .venv/bin/python -e .
cp murmur.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now murmur
```

The Parakeet weights (~2.5 GB) download once into the HuggingFace cache on
first run.

## Use

| Action | How |
|---|---|
| Dictate | **Super+D**, speak, Super+D again |
| Stop by not speaking | just stop; VAD ends the take after 1.6 s of silence |
| Check state and timings | `murmur status` |
| Test without injecting | `murmur dry` then `murmur stop` |
| Diagnose | `murmur doctor` |
| Logs | `journalctl --user -u murmur -f` |

A take ends when you toggle again, when the VAD hears 1.6 s of silence, after
10 s if you never speak at all, or at the 300 s hard cap.

## Config

Optional, at `~/.config/murmur/config.toml`. Read once at daemon start, so
restart the service after editing.

```toml
model = "nvidia/parakeet-tdt-0.6b-v3"
device = "cuda"
source = ""              # PipeWire source; empty = system default
silence_timeout = 1.6    # silence that ends a take
no_speech_timeout = 10.0 # give up if you toggle on and never speak
max_seconds = 300        # hard cap
inject = "wtype"         # "wtype" | "clipboard" | "none"
strip_fillers = true     # drop standalone "um"/"uh" before injecting
transcript_log = true    # append every take to ~/.local/share/murmur/takes.jsonl
log_text = true          # include take text in the journal
```

`strip_fillers` removes only standalone filler tokens, never substrings, so
"umbrella" and "Uhura" survive. An utterance that is nothing but "um" is left
alone rather than vanishing.

## Measured on muthur

| | |
|---|---|
| Model load (warm HF cache) | 10.9 s, once at login |
| Startup warmup decode | 0.8 s |
| Warm decode, ~2 s of audio | 43-68 ms |
| Real speech, 3.6 s of audio | 63 ms |
| Real speech, 4.8 s of audio | **34 ms** (~140x realtime) |
| VRAM resident | 2.69 GB |

Without the startup warmup the first take of each session costs ~817 ms,
because CUDA kernel init is lazy. That is what the throwaway decode at startup
buys.

## The take corpus

Every take is appended to `~/.local/share/murmur/takes.jsonl` (mode 0600), with
the raw model output and, when post-processing changed it, the cleaned version:

```json
{"at": "2026-07-31T16:10:44", "audio_s": 2.6, "latency_ms": 34,
 "stopped_by": "user", "raw": "the fellow thing", "cleaned": "the fellow thing"}
```

This exists to seed a correction map from real vocabulary misses rather than
imagined ones. Client names, jargon and product names are where an ASR model
with no context will reliably fail, and guessing at the substitution list is
how you end up with corrections that fire on the wrong words.

To review what it has been getting wrong:

```sh
jq -r '.cleaned // .raw' ~/.local/share/murmur/takes.jsonl | less
```

It is a plaintext record of everything you dictate. `transcript_log = false`
turns it off; `log_text = false` separately stops take text reaching the
journal.

## Design notes

The audio callback does exactly one thing: copy the buffer onto a queue. All
VAD inference and stop-condition logic runs on a normal worker thread. Nothing
mutable is shared between the audio thread and anything else.

Socket handlers never do work either. They push a command onto a queue and
reply immediately, so the keyboard shortcut always returns instantly. A single
controller thread owns the recorder and the engine and processes commands
serially, which removes the need for locking around that state.

The daemon handles SIGTERM as well as SIGINT, so systemd stop and logout
finish the current take cleanly rather than dropping it.

## Known gaps

- **Injection uses the virtual-keyboard protocol, not input-method-v2.**
  cosmic-comp advertises `zwp_input_method_manager_v2`, which would insert text
  directly into the focused field with no keycode translation, and
  `set_preedit_string` would let text appear live while speaking. That needs a
  real Wayland client rather than a subprocess, so it's the next step rather
  than a blocker.
- **No hold-to-talk.** COSMIC's shortcut schema is a single
  `(modifiers, key) -> Action` map with no press/release distinction, so
  toggle plus a VAD backstop is the closest available shape.
- **No punctuation or formatting commands.** "new line", "comma" and friends
  are not interpreted. Parakeet punctuates declaratives and questions on its
  own, which covers most of the need.
- **Accuracy is spot-checked, not measured.** The first real takes came back
  clean, with correct punctuation, capitalisation and apostrophes. That is a
  good sign, not a word error rate.

## When the hotkey does nothing

Check the journal before anything else. The failure that looks like a dead
hotkey is usually a live hotkey and a dead decode:

```
murmur: recording
murmur: transcribe failed: decode failed: CUDA error: unknown error
```

`recording` means Super+D, the shortcut, the socket and the microphone are all
fine. Only the GPU side is broken.

`CUDA error: unknown error` on a daemon that loaded its model happily is a
poisoned CUDA context: memory copies still work, which is why the weights
loaded, but kernel launches do not. It was seen once, on 2026-09-15, when a
logout and login restarted the daemon four seconds into a compositor
bring-up, while cosmic-comp was still taking DRM master. Starting a CUDA
context in that window appears to produce one that never works.

To tell a poisoned daemon from a genuinely broken GPU, decode in a fresh
process:

```sh
.venv/bin/python scripts/smoke_asr.py
```

If that works and the daemon does not, the daemon's context is the problem
and `systemctl --user restart murmur` fixes it.

Two numbers worth knowing. A healthy daemon holds about 3.0 GiB of VRAM, so
substantially more than that means failed decodes have been stranding
allocations. And a warm take is tens of milliseconds, so anything near a
second is the lazy CUDA kernel init that the startup warmup is supposed to
have paid already.

Since the fixes in this repo's history, a daemon whose warmup cannot decode
exits non-zero rather than reporting itself ready, and systemd retries it
three times across five minutes. So this should now recover on its own, and
say so in the journal if it cannot.

## Tests

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

25 tests covering config parsing, silence trimming, filler stripping, take
logging, injection guards and the daemon command protocol. The audio thread and GPU decode are covered by
`murmur doctor` and `scripts/smoke_asr.py` instead, since neither is
meaningfully unit-testable without hardware.
