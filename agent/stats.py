from langchain_core.callbacks import BaseCallbackHandler
from enum import Enum
from langchain_core.messages import BaseMessage
from loguru import logger
import common.utils as utils
from datetime import datetime

class ToolCountHandler(BaseCallbackHandler):
    counter: dict[str, int] = {}
    def __init__(self):
        self.counter = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized.get("name")
                or serialized.get("id")
                or "unknown_tool")
        if name in ["graphql_schema_info", "graphql_query_validator", "graphql_execute", "graphql_type_detail"]:
            return
        self.counter[name] = self.counter.get(name, 0) + 1

    def stats(self) -> dict[str, int]:
        return self.counter
    

class ProjectCounter:

    # { cid -> [suc, fail] }
    counter: dict[str, list[int]] = {}
    def __init__(self):
        self.counter = {}

    def incr(self, cid: str, success: bool = True) -> dict[str, list[int]]:
        if cid not in self.counter:
            self.counter[cid] = [0, 0]
    
        self.counter[cid][0] += 1 if success else 0
        self.counter[cid][1] += 0 if success else 1

        return self.counter

    def stats(self) -> dict[str, list[int]]:
        return self.counter

class ToolCounter:
    counter: dict[str, int] = {}
    def __init__(self):
        self.counter = {}

    def incr(self, tool_name: str, count: int) -> dict[str, int]:
        self.counter[tool_name] = self.counter.get(tool_name, 0) + count
        return self.counter

    def stats(self) -> dict[str, int]:
        return self.counter


class ProjectUsageMetrics:

    def __init__(self):
        self._synthetic_tool_counter = ToolCounter()
        self._organic_tool_counter = ToolCounter()
        self._synthetic_project_counter = ProjectCounter()
        self._organic_project_counter = ProjectCounter()

    @property
    def synthetic_tool_usage(self) -> ToolCounter:
        return self._synthetic_tool_counter

    @property
    def organic_tool_usage(self) -> ToolCounter:
        return self._organic_tool_counter

    @property
    def synthetic_project_usage(self) -> ProjectCounter:
        return self._synthetic_project_counter

    @property
    def organic_project_usage(self) -> ProjectCounter:
        return self._organic_project_counter
    
    def stats(self) -> dict[str, any]:
        return {
            "synthetic_tool_usage": self.synthetic_tool_usage.stats(),
            "organic_tool_usage": self.organic_tool_usage.stats(),
            "synthetic_project_usage": self.synthetic_project_usage.stats(),
            "organic_project_usage": self.organic_project_usage.stats()
        }


class Phase(Enum):
    GENERATE_QUESTION = "generate_question"
    GENERATE_GROUND_TRUTH = "generate_ground_truth"
    GENERATE_MINER_GROUND_TRUTH_SCORE = "ground_truth_score"

    MINER_SYNTHETIC = "miner_synthetic_challenge"
    MINER_ORGANIC_NONSTREAM = "miner_organic_nonstream_challenge"
    MINER_ORGANIC_STREAM = "miner_organic_stream_challenge"

class TokenUsageMetrics:
    datas: list[any] = []
    count: int

    def __init__(self, datas: list = None):
        self.datas = datas if datas is not None else []
        self.count = 0

    def parse(
            self,
            cid_hash: str,
            phase: Phase,
            response: BaseMessage | dict[str, any],
            extra: dict = {}
        ) -> dict[str, any]:
        extra_input_tokens = 0
        extra_input_cache_read_tokens = 0
        extra_output_tokens = 0

        if isinstance(response, dict):
            messages = response.get('messages', [])
            extra_input_tokens = response.get('intermediate_graphql_agent_input_token_usage', 0)
            extra_input_cache_read_tokens = response.get('intermediate_graphql_agent_input_cache_read_token_usage', 0)
            extra_output_tokens = response.get('intermediate_graphql_agent_output_token_usage', 0)
        else:
            messages = [response]

        input_tokens, input_cache_read_tokens, output_tokens = utils.extract_token_usage(messages)
        tool_calls = utils.extract_tool_calls(messages)
        logger.info(f"[TokenUsageMetrics] - append cid_hash: {cid_hash}, phase: {phase}, input_tokens: {input_tokens}, input_cache_read_tokens: {input_cache_read_tokens} output_tokens: {output_tokens}, extra_input_tokens: {extra_input_tokens}, extra_input_cache_read_tokens: {extra_input_cache_read_tokens}, extra_output_tokens: {extra_output_tokens}, tool_calls: {tool_calls}")

        data = {
            "cid_hash": cid_hash,
            "phase": phase.value,
            "input_tokens": input_tokens + extra_input_tokens,
            "input_cache_read_tokens": input_cache_read_tokens + extra_input_cache_read_tokens,
            "output_tokens": output_tokens + extra_output_tokens,
            "tool_calls": tool_calls,
            "timestamp":  int(datetime.now().timestamp())
        }
        data.update(extra)
        return data

    def append(
            self,
            data: dict[str, any]
        ) -> dict[str, any]:
        if not data:
            return None
        
        self.datas.append(data)
        self.count += 1

        # trim old records if count exceeds threshold
        if self.count > 10:
            self.count = 0
            current_time = int(datetime.now().timestamp())
            twenty_four_hours_ago = current_time - (24 * 60 * 60)  # 24 hours in seconds
            
            original_count = len(self.datas)
            
            # For shared lists (manager.list), we need to modify in-place
            if hasattr(self.datas, '_callmethod'):  # Check if it's a manager.list
                # Remove items in reverse order to avoid index shifting
                for i in range(len(self.datas) - 1, -1, -1):
                    if self.datas[i]["timestamp"] <= twenty_four_hours_ago:
                        del self.datas[i]
            else:
                # For regular lists
                self.datas = [d for d in self.datas if d["timestamp"] > twenty_four_hours_ago]
            
            trimmed_count = original_count - len(self.datas)
            
            if trimmed_count > 0:
                logger.info(f"[TokenUsageMetrics] Trimmed {trimmed_count} records older than 24 hours (original: {original_count}, remaining: {len(self.datas)})")

        return data

    def stats(self, since_timestamp: int) -> list[any]:
        return [data for data in self.datas if data["timestamp"] > since_timestamp]
