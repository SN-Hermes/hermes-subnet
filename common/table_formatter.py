from rich.console import Console
from rich.box import ROUNDED
from rich.table import Table
from loguru import logger


class TableFormatter:
    """Rich table formatter for consistent logging display"""
    
    def __init__(self):
        self.console = Console()
    
    def create_single_column_table(self, header: str, content: str, header_style: str = "bold") -> str:
        """Create a single column table with header and content"""
        table = Table(show_header=True, header_style=header_style)
        table.add_column(header, style="white")
        table.add_row(content)
        
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get().strip()
    
    def create_single_column_multiple_row_table(self, header: str, rows: list[str], header_style: str = "bold") -> str:
        """Create a single column table with header and content"""
        table = Table(show_header=True, header_style=header_style, box=ROUNDED)
        table.add_column(header, style="white")
        for row in rows:
            table.add_row(row)
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get().strip()
    
    def create_two_column_table(self, label: str, value: str, label_width: int = 15) -> str:
        """Create a two column table for label-value pairs"""
        table = Table(show_header=False)
        table.add_column("Label", style="cyan", no_wrap=True, width=label_width)
        table.add_column("Value", style="white")
        table.add_row(label, value)
        
        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get().strip()
    
    def create_miner_response_tables(self, uid: int, question: str, elapsed_time: float, challenge_id: str = "",
                                   miner_answer: str = None, ground_truth: str = None) -> str:
        """Create formatted tables for miner response display"""
        output_lines = [f"🔍 MINER RESPONSE [UID: {uid} ({challenge_id})]"]
        
        # Question table
        output_lines.append(self.create_single_column_table("❓ Question", question))
        
        # Response Time table (two columns)
        if miner_answer:
            output_lines.append(self.create_two_column_table("⏱️ Response Time", f"{elapsed_time:.2f}s"))
        
        if miner_answer:
            # Miner Answer table
            output_lines.append(self.create_single_column_table("✅ Miner Answer", miner_answer))
            
            # Ground Truth table
            if ground_truth:
                output_lines.append(self.create_single_column_table("📊 Ground Truth", ground_truth))
        else:
            # Status table for no response
            output_lines.append(self.create_two_column_table("Status", "No Response Received"))
        
        return "\n".join(output_lines)
    
    def create_synthetic_challenge_table(self, question: str, challenge_id: str = "") -> str:
        """Create table for synthetic challenge display"""
        return self.create_single_column_table("🤖 Synthetic Challenge" + f" ({challenge_id})", question, "bold green")

    def create_ground_truth_tables(self, ground_truth: str, generation_cost: float, challenge_id: str = "") -> str:
        """Create tables for ground truth display"""
        output_lines = []
        
        # Ground Truth table (single column)
        output_lines.append(self.create_single_column_table("🤖 Ground Truth" + f" ({challenge_id})", ground_truth))
        
        # Generation Cost table (two columns)
        output_lines.append(self.create_two_column_table("⏱️ Generation Cost", f"{generation_cost:.2f}s", 20))
        
        return "\n".join(output_lines)
    
    def log_with_newline(self, content: str, level: str = "info", **kwargs):
        """Log content with newline prefix, avoiding format string issues"""
        log_func = getattr(logger.opt(raw=True), level)
        log_func("\n{}\n", content, **kwargs)


# Global instance for easy access
table_formatter = TableFormatter()