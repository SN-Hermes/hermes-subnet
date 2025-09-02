from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field


class MultiInput(BaseModel):
    a: int = Field(description="first integer")
    b: int = Field(description="second integer")


class MultiTool(BaseTool):
    __version__ = "1.0.0"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "multiply"
    description: str = "multiply two integers."
    args_schema: Optional[ArgsSchema] = MultiInput

    def __init__(self):
        super().__init__()

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> int:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        return self._run(a, b, run_manager=run_manager.get_sync())

tools = [MultiTool()]