import json
import time
import random
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class CachedResult:
    id: str
    data: Any
    size: int
    created_at: float
    source_path: str


class CovalentContext:
    def __init__(self):
        self._latest_result: Optional[CachedResult] = None

    def set_result(self, data: Any, source_path: str) -> CachedResult:
        result_id = self._generate_id()
        content = json.dumps(data)
        size = len(content.encode('utf-8'))

        result = CachedResult(
            id=result_id,
            data=data,
            size=size,
            created_at=time.time(),
            source_path=source_path
        )

        self._latest_result = result
        return result

    def get_result(self) -> Optional[CachedResult]:
        return self._latest_result

    def has_result(self) -> bool:
        return self._latest_result is not None

    def _generate_id(self) -> str:
        timestamp = int(time.time() * 1000)
        random_part = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=7))
        return f"result_{timestamp}_{random_part}"
