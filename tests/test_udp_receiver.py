"""Regression tests for immediate UDP snapshot publication."""

from __future__ import annotations

import json
import unittest

from dora_openarm_vr.udp_receiver import JsonUdpReceiver


class _PacketSocket:
    def __init__(self) -> None:
        self.recv_count = 0

    def recvfrom(self, _buf_size: int) -> tuple[bytes, tuple[str, int]]:
        payload = json.dumps({"sequence": self.recv_count}).encode()
        self.recv_count += 1
        return payload, ("127.0.0.1", 5006)


class JsonUdpReceiverTest(unittest.TestCase):
    def test_receive_one_does_not_drain_additional_packets(self) -> None:
        receiver = object.__new__(JsonUdpReceiver)
        receiver._buf_size = 4096
        srv = _PacketSocket()

        latest, _recv_ns, monotonic_s = receiver._receive_one(srv)

        self.assertEqual(srv.recv_count, 1)
        self.assertEqual(latest, {"sequence": 0})
        self.assertIsNotNone(monotonic_s)


if __name__ == "__main__":
    unittest.main()
