import math
import hashlib
import json
from pathlib import Path
from typing import Optional


class Bloom:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        capacity: expected number of elements to store (n)
        error_rate: desired false positive probability (p)
        """
        self.n = capacity
        self.p = error_rate

        # Automatically compute optimal parameters
        self.m = self._optimal_m(capacity, error_rate)
        self.k = self._optimal_k(self.m, capacity)

        self.bytes = (self.m + 7) // 8
        self.bit_array = bytearray(self.bytes)

    def _optimal_m(self, n, p):
        return int(-(n * math.log(p)) / (math.log(2) ** 2))

    def _optimal_k(self, m, n):
        return max(1, int((m / n) * math.log(2)))

    def _hashes(self, data: str):
        h = hashlib.sha256(data.encode()).digest()

        h1 = int.from_bytes(h[:16], 'big')
        h2 = int.from_bytes(h[16:], 'big')

        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, data: str):
        for pos in self._hashes(data):
            byte_index = pos >> 3
            bit_mask = 1 << (pos & 7)
            self.bit_array[byte_index] |= bit_mask

    def __contains__(self, data: str):
        for pos in self._hashes(data):
            byte_index = pos >> 3
            bit_mask = 1 << (pos & 7)
            if not (self.bit_array[byte_index] & bit_mask):
                return False
        return True

    def current_error_rate(self, inserted: int):
        return (1 - math.exp(-self.k * inserted / self.m)) ** self.k

    def info(self):
        return {
            "capacity": self.n,
            "error_rate_target": self.p,
            "bit_size": self.m,
            "byte_size": self.bytes,
            "hash_count": self.k,
        }

    def save_to_file(self, filepath: str | Path):
        import base64
        
        data = {
            "capacity": self.n,
            "error_rate": self.p,
            "bit_size": self.m,
            "hash_count": self.k,
            "bit_array": base64.b64encode(self.bit_array).decode('utf-8')
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str | Path) -> Optional['Bloom']:
        if not filepath:
            return None
        
        import base64
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bloom = cls(capacity=data["capacity"], error_rate=data["error_rate"])
        
        # Restore the bit array
        bloom.bit_array = bytearray(base64.b64decode(data["bit_array"]))
        
        # Verify the parameters match
        if bloom.m != data["bit_size"] or bloom.k != data["hash_count"]:
            return None
        
        return bloom

