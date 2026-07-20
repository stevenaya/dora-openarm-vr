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
import socket
import threading
import time


class JsonUdpReceiver:
    """Background thread that binds a UDP socket and keeps the latest parsed JSON packet."""

    _ERROR_LOG_INTERVAL_S = 1.0

    def __init__(self, host: str, port: int, buf_size: int = 4096) -> None:
        self._host = host
        self._port = port
        self._buf_size = buf_size
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_revision = 0
        self._latest_monotonic_s: float | None = None
        self._recv_ts: collections.deque[int] = collections.deque(maxlen=512)
        self._last_error_log_s = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest

    def latest_snapshot(self) -> tuple[dict | None, int, float | None]:
        """Return the latest packet, its revision, and monotonic receive time."""
        with self._lock:
            return (
                self._latest,
                self._latest_revision,
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

    def _receive_one(self, srv: socket.socket) -> tuple[dict | None, int, float | None]:
        """Receive and timestamp one UDP packet."""
        data, _ = srv.recvfrom(self._buf_size)
        recv_ns = time.time_ns()
        recv_monotonic_s = time.perf_counter()
        msg = self._parse_packet(data)
        sample_time = recv_monotonic_s if msg is not None else None
        return msg, recv_ns, sample_time

    def _log_error(self, context: str, exc: BaseException) -> None:
        now = time.monotonic()
        if now - self._last_error_log_s < self._ERROR_LOG_INTERVAL_S:
            return
        self._last_error_log_s = now
        print(f"[receiver] UDP {context} error: {type(exc).__name__}: {exc}")

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
                            msg, recv_ns, sample_time = self._receive_one(srv)

                            if msg is not None:
                                with self._lock:
                                    self._recv_ts.append(recv_ns)
                                    self._latest = msg
                                    self._latest_revision += 1
                                    self._latest_monotonic_s = sample_time

                        except TimeoutError:
                            continue
                        except Exception as exc:
                            self._log_error("receive", exc)
            except OSError as exc:
                if self._running:
                    self._log_error("socket", exc)
                    time.sleep(1.0)
