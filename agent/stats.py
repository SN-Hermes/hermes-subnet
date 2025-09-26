from langchain_core.callbacks import BaseCallbackHandler


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


class Metrics:

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