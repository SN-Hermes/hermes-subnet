
import datetime
import time
from common.table_formatter import table_formatter


class Timer:
    def __init__(self, label="", log=None, metadata=None):
        self.response = ''
        self.error = ''
        self.label = label
        self.elapsed = 0.0
        self.metadata = metadata or {}
        self.log = log

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
            rows = [self.response]
            if self.error:
                rows.append(f"⚠️ Error: {self.error}")
            output_lines.append(table_formatter.create_single_column_multiple_row_table("💬 Answer", rows))
        self.log and self.log.info("\n".join(output_lines))
        self.log and self.log.info(f"""⏱️ cost: {self.final_time:.4f}s""")