"""Append-only JSONL cache for LLM calls. Key = sha256 of every input that affects the output."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def cache_key(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class JsonlCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._d: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated last line from an interrupted run
                    self._d[rec["key"]] = rec

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def put(self, key: str, value: dict) -> None:
        rec = {"key": key, **value}
        with self._lock:
            self._d[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append(rec)

    def _append(self, rec: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
