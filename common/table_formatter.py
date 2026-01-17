from rich.console import Console
from rich.box import ROUNDED
from rich.table import Table
from loguru import logger
import bittensor as bt
from common.enums import ErrorCode
from common.protocol import OrganicNonStreamSynapse


class TableFormatter:
    """Rich table formatter for consistent logging display"""
    
    def __init__(self):
        self.console = Console()
    
    def create_single_column_table(self, header: str, rows: list[str], header_style: str = "bold", caption: str = "") -> str:
        """Create a single column table with header and content"""
        table = Table(show_header=True, header_style=header_style, caption=caption, caption_justify="left", box=ROUNDED)
        table.add_column(header, style="white")
        for row in rows:
            table.add_row(row)

        with self.console.capture() as capture:
            self.console.print(table)
        return capture.get().strip()
    
    def create_multiple_column_table(
            self,
            columns: list[str],
            rows: list[str],
            header_style: str = "bold",
            title: str = "",
            caption: str = "",
        ) -> str:
        """Create a multiple column table with headers and content"""
        table = Table(
            title=title,
            title_style="bold",
            title_justify="left",
            caption=caption,
            caption_justify="left",
            show_header=True,
            header_style=header_style,
            box=ROUNDED
        )
        for col in columns:
            table.add_column(col, style="white")
        for row in rows:
            table.add_row(*row)

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
    
    def create_ground_truth_tables(self, ground_truth: str, generation_cost: float, challenge_id: str = "") -> str:
        """Create tables for ground truth display"""
        output_lines = []
        
        # Ground Truth table (single column)
        output_lines.append(self.create_single_column_table("🤖 Ground Truth" + f" ({challenge_id})", ground_truth))
        
        # Generation Cost table (two columns)
        output_lines.append(self.create_two_column_table("⏱️ Generation Cost", f"{generation_cost:.2f}s", 20))
        
        return "\n".join(output_lines)
    
    def create_synthetic_challenge_table(
        self,
        round_id: str,
        challenge_id: str,
        cid: str,
        question: str,
        success: bool,
        ground_truth: str,
        ground_cost: float,
        metrics_data: dict | None = None
    ):
        header = "🤖 Synthetic Challenge" + f" ({round_id} | {challenge_id})"
        rows = [
            f"❓ Question: {question}\n",
            f"🎯 Ground Truth: {None if not success else ground_truth}\n",
            f"⚠️ {ground_truth}\n" if not success else "",
            f"📊 Metrics Data: { metrics_data}\n" if metrics_data else "",
            f"⏱️ Cost: {ground_cost}s"
        ]
        challenge_output = self.create_single_column_table(
            header=header,
            rows=rows,
            header_style="bold green",
            caption=f"cid: {cid}"
        )
        self.log_with_newline(challenge_output, "info")

    def create_synthetic_miners_response_table(
        self,
        round_id: str,
        challenge_id: str,
        uids: list[int],
        responses: list[bt.Synapse],
        ground_truth_scores: list[float],
        elapse_weights: list[float],
        zip_scores: list[float],
        cid: str,
        max_table_rows: int
    ):
        header = "🤖 Synthetic Challenge" + f" ({round_id} | {challenge_id})"
        rows = []

        # Separate successful and failed responses
        successful_indices = []
        failed_indices = []
        for idx, r in enumerate(responses):
            (successful_indices if r.is_success else failed_indices).append(idx)
        
        # Prioritize successful responses
        prioritized_indices = successful_indices + failed_indices
        limited_indices = prioritized_indices[:max_table_rows] if max_table_rows > 0 else prioritized_indices
        
        for idx in enumerate(limited_indices):
            uid = uids[idx]
            r = responses[idx]
            rstr = None
            if r.is_success:
                if r.status_code == ErrorCode.SUCCESS.value:
                    rstr = f"{r.response}"
                else:
                    rstr = f"⚠️ {r.status_code}: {r.error}"
            else:
                rstr = f"⚠️ {r.dendrite.status_code}"
                    
            # uid_hotkey = f"{uid}|{r.dendrite.hotkey}" if getattr(r.dendrite, 'hotkey', None) else f"{uid}"
            rows.append([
                f"{uid}",
                f"{rstr}",
                f"{r.elapsed_time}s",
                f"{ground_truth_scores[idx]}",
                f"{elapse_weights[idx]}",
                f"{zip_scores[idx]}",
            ])

        caption = f"cid: {cid}"
        if max_table_rows > 0 and len(uids) > max_table_rows:
            caption += f" (showing first {max_table_rows} of {len(uids)} miners)"
    
        miners_response_output = self.create_multiple_column_table(
            title=f"{header} - Miners Response",
            caption=caption,
            columns=[
                "UID",
                "Response",
                "Elapsed Time",
                "Truth Score",
                "Elapse Weight",
                "Score"
            ],
            rows=rows
        )
        self.log_with_newline(miners_response_output, "info")

    def create_synthetic_final_ranking_table(
        self,
        round_id: str,
        challenge_id: str,
        uids: list[int],
        hotkeys: list[str],
        workload_counts: list[int],
        quality_scores: list[list[float]],
        workload_score: list[float],
        new_ema_scores: dict[int, tuple[float, str]],
        max_table_rows: int

    ):
        header = "🤖 Synthetic Challenge" + f" ({round_id} | {challenge_id})"
        rows = []

        data = list(zip(uids, hotkeys, workload_counts, quality_scores, workload_score))

        # Sort by new_ema_scores[uid][0] in descending order
        sorted_data = sorted(data, key=lambda x: new_ema_scores[x[0]][0], reverse=True)
        
        # Limit to max_table_rows if specified
        limited_data = sorted_data[:max_table_rows] if max_table_rows > 0 else sorted_data

        for uid, hotkey, workload_count, quality_score, workload_s in limited_data:
            rows.append([
                f"{uid}",
                f"{hotkey}",
                f"{workload_count}",
                f"{', '.join(map(str, quality_score))}",
                f"{workload_s}",
                f"{new_ema_scores[uid][0]}"
            ])

        caption = f""
        if max_table_rows > 0 and len(uids) > max_table_rows:
            caption += f" (showing top {max_table_rows} of {len(uids)} miners by EMA score)"

        miners_response_output = table_formatter.create_multiple_column_table(
            title=f"{header} - Miners Final Score",
            caption=caption,
            columns=[
                "UID",
                "Hotkey",
                "Workload Count",
                "Workload Quality",
                "Workload Score",
                "Final EMA Score"
            ],
            rows=rows
        )
        table_formatter.log_with_newline(miners_response_output, "info")

    def create_organic_challenge_table(
        self,
        id: str,
        cid: str,
        question: str,
        response: OrganicNonStreamSynapse
    ):
        header = "🌿 Organic" + f" ({id})"
        rstr = None
        if response.is_success:
            if response.status_code == ErrorCode.SUCCESS.value:
                # response.response is now a simple string (final answer)
                rstr = f"💬 Answer: {response.response}"
            else:
                rstr = f"⚠️ {response.status_code}: {response.error}"
        else:
            rstr = f"⚠️ {response.dendrite.status_code}"
            
        rows = [
            f"❓ Question: {question}\n",
            f"{rstr}\n",
            f"⏱️ Cost: {response.elapsed_time}s"
        ]
        challenge_output = self.create_single_column_table(
            header=header,
            rows=rows,
            header_style="bold green",
            caption=f"cid: {cid}"
        )
        self.log_with_newline(challenge_output, "info")

    def create_workload_summary_table(
        self,
        round_id: str,
        challenge_id: str,
        ground_truth: str,
        uids: list[int],
        responses: list[OrganicNonStreamSynapse],
        ground_truth_scores: list[float],
        elapse_weights: list[float],
        zip_scores: list[float],
        cid: str
    ):
        header = "🤖 Organic Workload" + f" ({round_id} | {challenge_id})"
        rows = []
        for idx, uid in enumerate(uids):
            r = responses[idx]
            if r.is_success:
                if r.status_code == ErrorCode.SUCCESS.value:
                    rstr = f"{r.response}"
                else:
                    rstr = f"⚠️ {r.status_code}: {r.error}"
            else:
                rstr = f"⚠️ {r.dendrite.status_code}"
                    
            # uid_hotkey = f"{uid}|{r.dendrite.hotkey}" if getattr(r.dendrite, 'hotkey', None) else f"{uid}"
            rows.append([
                f"{uid}",
                f"{rstr}",
                f"{ground_truth}",
                f"{r.elapsed_time}s",
                f"{ground_truth_scores[idx]}",
                f"{elapse_weights[idx]}",
                f"{zip_scores[idx]}",
            ])
        miners_response_output = self.create_multiple_column_table(
            title=f"{header} - Miners Response",
            caption=f"cid: {cid}",
            columns=[
                "UID",
                "Response",
                "Ground Truth",
                "Elapsed Time",
                "Truth Score",
                "Elapse Weight",
                "Score"
            ],
            rows=rows
        )
        self.log_with_newline(miners_response_output, "info")

    def log_with_newline(self, content: str, level: str = "info", **kwargs):
        """Log content with newline prefix, avoiding format string issues"""
        log_func = getattr(logger.opt(raw=True), level)
        log_func("\n{}\n", content, **kwargs)


# Global instance for easy access
table_formatter = TableFormatter()