# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import select
import socket
import threading
import time


class JsonUdpReceiver:
    """Background thread that binds a UDP socket and keeps the latest parsed JSON packet."""

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._host = host
        self._port = port
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_sequence = 0
        self._latest_monotonic_s: float | None = None
        self._recv_ts: collections.deque[int] = collections.deque(maxlen=512)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def latest_snapshot(self) -> tuple[dict | None, int, float | None]:
        """Return the latest packet, its receive sequence, and monotonic time."""
        with self._lock:
            return (
                self._latest,
                self._latest_sequence,
                self._latest_monotonic_s,
            )

    def drain_recv_timestamps(self) -> list[int]:
        """Return and clear the arrival timestamps (ns) collected since last call."""
        with self._lock:
            items = list(self._recv_ts)
            self._recv_ts.clear()
            return items

    def close(self) -> None:
        self._running = False

    def _parse_packet(self, data: bytes) -> dict | None:
        try:
            line = data.decode("utf-8", errors="replace").strip()
            if not line:
                return None
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _loop(self) -> None:
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as srv:
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind((self._host, self._port))
                    srv.settimeout(1.0)
                    print(f"[receiver] Listening on UDP {self._host}:{self._port}")

                    while self._running:
                        try:
                            data, _ = srv.recvfrom(self._buf_size)
                            recv_ns = time.time_ns()
                            recv_monotonic_s = time.perf_counter()
                            last_msg = self._parse_packet(data)
                            arrivals = [recv_ns] if last_msg is not None else []
                            last_monotonic_s = (
                                recv_monotonic_s if last_msg is not None else None
                            )

                            # Drain any queued datagrams, keep only the freshest
                            # pose, but record every packet's real arrival time.
                            while select.select([srv], [], [], 0.0)[0]:
                                data, _ = srv.recvfrom(self._buf_size)
                                recv_ns = time.time_ns()
                                recv_monotonic_s = time.perf_counter()
                                parsed = self._parse_packet(data)
                                if parsed is not None:
                                    arrivals.append(recv_ns)
                                    last_msg = parsed
                                    last_monotonic_s = recv_monotonic_s

                            with self._lock:
                                self._recv_ts.extend(arrivals)
                                if last_msg is not None:
                                    self._latest = last_msg
                                    self._latest_sequence += len(arrivals)
                                    self._latest_monotonic_s = last_monotonic_s

                        except TimeoutError:
                            continue
                        except Exception:
                            pass
            except OSError:
                if self._running:
                    time.sleep(1.0)
