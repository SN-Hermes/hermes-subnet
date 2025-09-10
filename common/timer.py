
import datetime
import time
from loguru import logger
from common.table_formatter import table_formatter


class Timer:
    def __init__(self, label="", metadata=None):
        self.response = ''
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
        
        # Create formatted output with table for answer
        output_lines = [self.label]
        
        if self.response:
            # Create answer table
            output_lines.append(table_formatter.create_single_column_table("💬 Answer", self.response))
        logger.info("\n".join(output_lines))
        logger.info(f"""⏱️ cost: {self.final_time:.4f}s""")