"""murmur CLI.

    murmur daemon     run the resident dictation daemon (systemd runs this)
    murmur toggle     start/stop a take - bind this to a COSMIC shortcut
    murmur start      begin a take
    murmur stop       end a take and transcribe
    murmur status     what the daemon is doing, plus last-take timings
    murmur doctor     check audio, injection, GPU and daemon health
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from .config import SOCKET_PATH, Config
from .daemon import Daemon, send
from .inject import available


def cmd_daemon(args) -> int:
    cfg = Config.load()
    return Daemon(cfg).run()


def cmd_send(args) -> int:
    reply = send(args.command)
    if args.json:
        print(json.dumps(reply))
    elif not reply.get("ok"):
        print(f"murmur: {reply.get('error', 'failed')}", file=sys.stderr)
    elif args.command == "status":
        for key in (
            "state", "model", "model_loaded", "takes",
            "last_audio_s", "last_latency_ms", "last_stopped_by", "last_error",
        ):
            if key in reply and reply[key] not in ("", None):
                print(f"{key:18} {reply[key]}")
        if reply.get("last_text"):
            print(f"{'last_text':18} {reply['last_text'][:100]!r}")
    return 0 if reply.get("ok") else 1


def cmd_doctor(args) -> int:
    cfg = Config.load()
    checks: list[tuple[str, bool, str]] = []

    ok, detail = available(cfg.inject)
    checks.append((f"injection ({cfg.inject})", ok, detail))

    try:
        import sounddevice as sd

        default_in = sd.query_devices(kind="input")
        checks.append(("audio input", True, default_in["name"]))
    except Exception as exc:
        checks.append(("audio input", False, str(exc)))

    try:
        import torch

        if cfg.device == "cuda" and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info()
            checks.append((
                "gpu",
                free / 1e9 > 3.0,
                f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}, "
                f"{free / 1e9:.1f}/{total / 1e9:.1f} GB free",
            ))
        elif cfg.device == "cuda":
            checks.append(("gpu", False, "device=cuda but no CUDA available"))
        else:
            checks.append(("gpu", True, f"device={cfg.device}"))
    except Exception as exc:
        checks.append(("gpu", False, str(exc)))

    reply = send("status")
    if reply.get("ok"):
        checks.append(("daemon", True, f"{reply['state']}, model_loaded={reply['model_loaded']}"))
    else:
        checks.append(("daemon", False, f"not running (socket {SOCKET_PATH})"))

    checks.append((
        "wayland",
        shutil.which("wayland-info") is not None,
        "wayland-info present" if shutil.which("wayland-info") else "wayland-utils not installed",
    ))

    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name:22} {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="murmur", description="local dictation for COSMIC")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("daemon", help="run the resident daemon")
    for name in ("toggle", "dry", "start", "stop", "status", "quit"):
        sub.add_parser(name, help=f"send {name} to the daemon")
    sub.add_parser("doctor", help="check the setup")

    args = parser.parse_args(argv)
    if args.cmd == "daemon":
        return cmd_daemon(args)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    args.command = args.cmd
    return cmd_send(args)


if __name__ == "__main__":
    raise SystemExit(main())
