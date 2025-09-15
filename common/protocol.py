
import bittensor as bt
from typing import Optional, List
from pydantic import BaseModel, Field

from loguru import logger

# ===============  openai ================
class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatCompletionMessage] = Field(..., description="List of messages")
    stream: bool = Field(default=False, description="Whether to stream responses")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")


class SyntheticSynapse(bt.Synapse):
    time_elapsed: int = 0
    question: str | None = None
    response: Optional[dict] = None

class CapacitySynapse(bt.Synapse):
    time_elapsed: int = 0
    response: Optional[dict] = None

class SyntheticStreamSynapse(bt.StreamingSynapse):
    time_elapsed: int = 0
    question: str | None = None
    response: Optional[dict] = None

    async def process_streaming_response(self, response: "ClientResponse"):
        logger.info(f"Processing streaming response: {response}")

        buffer = ""
        async for chunk in response.content.iter_any():
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text
            logger.info(f"Streaming response part: {text}")
            yield text

        self._buffer = buffer


    def extract_response_json(self, r: "ClientResponse") -> dict:
        logger.info(f"Extracting JSON from response: {r}")
        self.response = {"final_text": getattr(self, "_buffer", "")}
        return self.response


    def deserialize(self):
        return '[end]'

class SyntheticNonStreamSynapse(bt.Synapse):
    id: str | None = None
    elapsed_time: float | None = 0.0
    project_id: str | None = None
    question: str | None = None
    response: str | None = ''
    error: str | None = None

class OrganicStreamSynapse(bt.StreamingSynapse):
    time_elapsed: int = 0
    project_id: str | None = None
    completion: ChatCompletionRequest | None = None
    response: Optional[dict] = None

    async def process_streaming_response(self, response: "ClientResponse"):
        logger.info(f"Processing streaming response2: {response}")

        buffer = ""
        async for chunk in response.content.iter_any():
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text
            logger.info(f"Streaming response part: {text}")
            yield text

        self._buffer = buffer

    def extract_response_json(self, r: "ClientResponse") -> dict:
        logger.info(f"Extracting JSON from response: {r}")
        self.response = {"final_text": getattr(self, "_buffer", "")}
        return self.response


    def deserialize(self):
        return '[end]'
    
class OrganicNonStreamSynapse(bt.Synapse):
    id: str | None = None
    elapsed_time: float | None = 0.0
    project_id: str | None = None
    completion: ChatCompletionRequest | None = None
    response: Optional[dict] = None