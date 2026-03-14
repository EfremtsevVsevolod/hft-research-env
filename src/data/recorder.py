"""Raw message recorder to Parquet files.

Buffers incoming messages and flushes to Parquet periodically.
Each row stores the raw JSON payload — no parsing or normalization.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


_COLUMNS = ["recv_ts", "exchange_ts", "event_type", "stream", "payload_json"]


class Recorder:
    """Buffer raw exchange messages and flush to Parquet files.

    Flushes the buffer when it reaches ``flush_every_events`` messages
    or every ``flush_every_seconds`` seconds, whichever comes first.

    Directory structure::

        base_dir/
          20260314_153000.parquet
          20260314_154000.parquet
          ...
    """

    def __init__(
        self,
        base_dir: Path,
        flush_every_events: int = 20000,
        flush_every_seconds: int = 10,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every_events = flush_every_events
        self._flush_every_seconds = flush_every_seconds

        self._buffer: list[dict] = []
        self._file_start_ts: float = time.time()

    def append(
        self,
        recv_ts: int,
        exchange_ts: int,
        event_type: str,
        stream: str,
        payload_json: str,
    ) -> None:
        """Add a raw message to the buffer.  Flushes automatically."""
        self._buffer.append({
            "recv_ts": recv_ts,
            "exchange_ts": exchange_ts,
            "event_type": event_type,
            "stream": stream,
            "payload_json": payload_json,
        })

        if len(self._buffer) >= self._flush_every_events:
            self.flush()
        elif time.time() - self._file_start_ts >= self._flush_every_seconds:
            self.flush()

    def flush(self) -> None:
        """Write buffered messages to a Parquet file."""
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer, columns=_COLUMNS)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._base_dir / f"{ts}.parquet"

        # Avoid overwriting if flushed within the same second.
        seq = 0
        while path.exists():
            seq += 1
            path = self._base_dir / f"{ts}_{seq}.parquet"

        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(path)
        self._buffer.clear()
        self._file_start_ts = time.time()

    def close(self) -> None:
        """Flush remaining buffer."""
        self.flush()
