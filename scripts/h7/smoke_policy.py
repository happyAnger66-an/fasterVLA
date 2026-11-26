#!/usr/bin/env python3
"""Connect to local :8013 and print server metadata (no GDK / no robot)."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("FASTERVLA_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("FASTERVLA_PORT", "8013")))
    args = p.parse_args()

    from fastervla.deploy.cli import _openpi_transport
    from fastervla.deploy.backends.openpi_ws import OpenPiWebsocketBackend, OpenPiWsConfig

    transport = _openpi_transport(args.host, args.port)
    cfg = OpenPiWsConfig(
        host=args.host,
        port=args.port,
        expect_checkpoint_step=89999,
        expect_norm_stats_name="badminton651",
        expect_prompt="Put the shuttlecock into the shuttlecock tube.",
        expect_num_steps=10,
        expect_rtc=True,
        expect_action_shape=(30, 24),
        expect_image_transport="jpeg",
    )
    backend = OpenPiWebsocketBackend(config=cfg, transport=transport)
    try:
        backend.warmup()
    except Exception as exc:  # noqa: BLE001
        meta = {}
        try:
            meta = transport.get_server_metadata()
        except Exception:  # noqa: BLE001
            pass
        print(json.dumps({"ok": False, "error": str(exc), "meta": meta}, ensure_ascii=False, default=str))
        return 1
    meta = transport.get_server_metadata()
    print(json.dumps({"ok": True, "host": args.host, "port": args.port, "meta": meta}, ensure_ascii=False, default=str))
    close = getattr(transport, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
