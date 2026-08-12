"""Local WebSocket relay for the ai4data_lab browser bridge.

The page connects first and identifies itself with `{"type": "ready"}`; the CLI
connects second with any frame. Frames are JSON. The relay only accepts one
client of each kind at a time and listens on loopback only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys

from websockets.asyncio.server import serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ai4data_lab.bridge")

PAGE_SOCKET = None
CLI_SOCKET = None


async def pump(ws, name: str) -> None:
    global PAGE_SOCKET, CLI_SOCKET
    try:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("non-JSON frame from %s", name)
                continue
            target = CLI_SOCKET if name == "page" else PAGE_SOCKET
            if target is None:
                continue
            try:
                await target.send(json.dumps(message))
            except Exception as exc:  # noqa: BLE001
                log.warning("send failed (%s -> %s): %s", name, "cli" if name == "page" else "page", exc)
    except Exception as exc:  # noqa: BLE001
        log.info("%s disconnected: %s", name, exc)
    finally:
        if name == "page" and PAGE_SOCKET is ws:
            PAGE_SOCKET = None
        if name == "cli" and CLI_SOCKET is ws:
            CLI_SOCKET = None


async def handler(ws) -> None:
    global PAGE_SOCKET, CLI_SOCKET
    try:
        hello = await ws.recv()
        message = json.loads(hello)
    except json.JSONDecodeError:
        await ws.close(code=1008, reason="Expected JSON hello")
        return
    except Exception:  # noqa: BLE001
        await ws.close(code=1011, reason="Failed to read hello")
        return

    if message.get("type") == "ready":
        if PAGE_SOCKET is not None:
            await ws.close(code=1008, reason="Page already connected")
            return
        PAGE_SOCKET = ws
        log.info("page connected")
        await pump(ws, "page")
    else:
        if CLI_SOCKET is not None:
            await ws.close(code=1008, reason="CLI already connected")
            return
        CLI_SOCKET = ws
        log.info("cli connected")
        await pump(ws, "cli")


async def main() -> None:
    async with serve(handler, "127.0.0.1", 8765, max_size=2**20):
        log.info("relay listening on ws://127.0.0.1:8765")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
