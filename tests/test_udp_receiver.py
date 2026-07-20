"""Tests for latest-only UDP receiver state."""

import json
import unittest
from unittest.mock import patch

from dora_openarm_vr.udp_receiver import JsonUdpReceiver


class UdpReceiverTest(unittest.TestCase):
    @patch("dora_openarm_vr.udp_receiver.threading.Thread")
    def test_each_packet_atomically_replaces_latest_snapshot(self, thread) -> None:
        receiver = JsonUdpReceiver("127.0.0.1", 5006)

        receiver._store_packet(json.dumps({"frame": 1}).encode(), 100, 1.0)
        receiver._store_packet(json.dumps({"frame": 2}).encode(), 200, 2.0)

        self.assertEqual(receiver.latest(), {"frame": 2})
        self.assertEqual(receiver.latest_snapshot(), ({"frame": 2}, 2, 2.0))
        self.assertEqual(receiver.drain_recv_timestamps(), [100, 200])
        thread.return_value.start.assert_called_once_with()

    @patch("dora_openarm_vr.udp_receiver.threading.Thread")
    def test_invalid_packet_does_not_replace_latest(self, _thread) -> None:
        receiver = JsonUdpReceiver("127.0.0.1", 5006)
        receiver._store_packet(json.dumps({"frame": 1}).encode(), 100, 1.0)

        receiver._store_packet(b"not-json", 200, 2.0)

        self.assertEqual(receiver.latest_snapshot(), ({"frame": 1}, 1, 1.0))
        self.assertEqual(receiver.drain_recv_timestamps(), [100])


if __name__ == "__main__":
    unittest.main()
