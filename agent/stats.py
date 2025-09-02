from typing import Dict
from langchain_core.callbacks import BaseCallbackHandler


class ToolCountHandler(BaseCallbackHandler):
    counter: Dict[str, int] = {}
    def __init__(self):
        self.counter = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized.get("name")
                or serialized.get("id")
                or "unknown_tool")
        self.counter[name] = self.counter.get(name, 0) + 1

    def stats(self) -> Dict[str, int]:
        return self.counter