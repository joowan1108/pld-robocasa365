"""BasePolicy with 2 real ZeroMQ server replicas: routing env i -> replica i % 2, concurrent queries."""

import socket
import threading
import time

import numpy as np

from pld_rc365.base.client import BasePolicy
from pld_rc365.base.protocol import Server


def _free_port_pair():
    for _ in range(50):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        s2 = socket.socket()
        try:
            s2.bind(("127.0.0.1", p + 1))
            s2.close()
            return p
        except OSError:
            s2.close()
    raise RuntimeError("no port pair")


def test_two_replicas_route_and_run_concurrently():
    port = _free_port_pair()
    seen = {0: [], 1: []}

    def make(r):
        def handler(batch):
            time.sleep(0.3)  # emulate a slow model
            seen[r].append(len(batch))
            out = np.zeros((len(batch), 16, 12), np.float32)
            out[..., 0] = r  # encode the replica that answered
            return out
        return handler

    for r in (0, 1):
        srv = Server("127.0.0.1", port + r, make(r), info={"kind": "fake", "replica": r})
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    bp = BasePolicy("127.0.0.1", port, num_envs=4, exec_steps=8, replicas=2)
    assert [i["replica"] for i in bp.info["replicas"]] == [0, 1]
    pkt = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in
                      ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")},
           "state": {"base_position": np.zeros(3, np.float32)}, "lang": "x"}
    for i in range(4):
        bp.reset_env(i, pkt)
    t0 = time.time()
    bp.ensure([0, 1, 2, 3])
    dt = time.time() - t0
    assert [bp.peek(i)[0] for i in range(4)] == [0, 1, 0, 1]      # env i served by replica i % 2
    assert seen == {0: [2], 1: [2]}                                # one batched request per replica
    assert dt < 0.55, dt                                           # concurrent (~0.3 s), not serial (~0.6 s)
    assert len(bp.queues[0]) == 8


def test_async_submit_poll_with_real_server():
    port = _free_port_pair()

    def handler(batch):
        time.sleep(0.3)
        return np.ones((len(batch), 16, 12), np.float32) * len(batch)

    srv = Server("127.0.0.1", port, handler, info={"kind": "fake"})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    bp = BasePolicy("127.0.0.1", port, num_envs=3, exec_steps=8, replicas=1)
    pkt = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in
                      ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")},
           "state": {"base_position": np.zeros(3, np.float32)}, "lang": "x"}
    for i in range(3):
        bp.reset_env(i, pkt)
    assert bp.submit([0, 1]) == 2          # one request with envs 0 and 1
    assert bp.submit([2]) == 0             # replica busy: env 2 waits for the next round
    assert bp.poll() == 0                  # not answered yet (non-blocking)
    assert bp.poll(block=True) == 2
    assert bp.has_action(0) and bp.has_action(1) and not bp.has_action(2)
    assert bp.peek(0)[0] == 2              # answered as a batch of 2
    bp.ensure([2])                         # synchronous path still works
    assert bp.has_action(2) and bp.n_requests == 2
