
import datetime
import time
from loguru import logger


class Timer:
    def __init__(self, label="", metadata=None):
        self.label = label
        self.elapsed = 0.0
        self.metadata = metadata or {}
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_datetime = datetime.datetime.now()
        return self

    def elapsed_time(self):
        return time.perf_counter() - self.start_time

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.final_time = self.end_time - self.start_time
        logger.info(f"[Timer] {self.label} took {self.final_time:.4f} seconds")