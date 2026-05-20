from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import websockets


class WebSocketBridge:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.clients: set[Any] = set()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="navai-websocket", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def publish(self, payload: dict[str, Any]) -> None:
        if not self.clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())
        self.loop.run_forever()

    async def _serve(self) -> None:
        async def handler(websocket: Any) -> None:
            self.clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.clients.discard(websocket)

        await websockets.serve(handler, self.host, self.port)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload)
        stale = []
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

