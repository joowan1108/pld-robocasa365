"""Minimal ZeroMQ REQ/REP transport (pickle payloads; localhost / trusted cluster only).

Request:  {"cmd": "infer", "obs": [per-env obs, ...]}   -> {"chunks": float32 (B, 16, 12), "latency_s": float}
          {"cmd": "info"}                                -> {"kind": ..., "model_path": ..., "seed": ...}
Per-env obs: {"video": {cam: uint8 (4, H, W, 3)}, "state": {key: float32 (4, D)}, "lang": str}
  (4 = history frames at relative steps [-6, -4, -2, 0], clamped to the episode start)
Chunk columns are in env action order: eef_pos(3) eef_rot(3) gripper(1) base_motion(4) control_mode(1).
"""

from __future__ import annotations

import pickle
import time
from typing import Any, Callable

import zmq


class Server:
    def __init__(self, host: str, port: int, handler: Callable[[list[dict]], Any], info: dict):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://{host}:{port}")
        self.handler = handler
        self.info = info

    def serve_forever(self):
        print(f"[server] ready: {self.info}", flush=True)
        while True:
            req = pickle.loads(self.sock.recv())
            cmd = req.get("cmd")
            try:
                if cmd == "infer":
                    t0 = time.time()
                    chunks = self.handler(req["obs"])
                    rep = {"chunks": chunks, "latency_s": time.time() - t0}
                elif cmd == "info":
                    rep = dict(self.info)
                else:
                    rep = {"error": f"unknown cmd {cmd}"}
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                rep = {"error": repr(e)}
            self.sock.send(pickle.dumps(rep, protocol=pickle.HIGHEST_PROTOCOL))


class Client:
    def __init__(self, host: str, port: int, timeout_s: float = 600.0):
        self.ctx = zmq.Context()
        self.addr = f"tcp://{host}:{port}"
        self.timeout_ms = int(timeout_s * 1000)
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, req: dict) -> dict:
        self.sock.send(pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL))
        rep = pickle.loads(self.sock.recv())
        if "error" in rep:
            raise RuntimeError(f"base-policy server error: {rep['error']}")
        return rep
