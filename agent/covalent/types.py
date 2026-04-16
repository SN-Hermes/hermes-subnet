from typing import Optional, Protocol
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0


@dataclass
class CovalentAgentConfig:
    llm: LLMConfig
    verbose: int = 0
    

@dataclass
class CovalentConfig:
    base_url: str
    authorization: Optional[str] = None


class CovalentAgent(Protocol):
    async def invoke(self, question: str) -> str:
        ...
