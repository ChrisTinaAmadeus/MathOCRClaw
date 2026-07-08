from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

class JsonCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(self.data, dict):
                    self.data = {}
            except Exception:
                self.data = {}

    def get(self, ns: str, key: str) -> Optional[Any]:
        try:
            v = self.data.get(ns, {}).get(key, None)
            return v
        except Exception:
            return None

    def set(self, ns: str, key: str, value: Any) -> None:
        if ns not in self.data or not isinstance(self.data.get(ns), dict):
            self.data[ns] = {}
        self.data[ns][key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

