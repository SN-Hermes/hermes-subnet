import json
import bittensor as bt
from typing import Optional, List
import fastapi
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from loguru import logger

from agent.stats import Metrics
from common.sqlite_manager import SQLiteManager


class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, sqlite_manager: SQLiteManager, metrics: Metrics):
        super().__init__(app)
        self.sqlite_manager = sqlite_manager
        self.metrics = metrics

    def handle_stats_html(self):
        with open(f"common/stats.html", "r", encoding="utf-8") as f:
            html = f.read()
        return fastapi.Response(content=html, media_type="text/html")

    def handle_stats_data(self, since_id: int = 0):
        if since_id > 0:
            data = self.sqlite_manager.fetch_newer_than(since_id)
        else:
            data = self.sqlite_manager.fetch_all()

        return fastapi.Response(content=json.dumps({"data": data, "usage": self.metrics.stats()}), media_type="application/json")

    async def dispatch(
        self, request: "fastapi.Request", call_next: "RequestResponseEndpoint"
    ) -> fastapi.Response:
        path = request.url.path
        if path == '/favicon.ico':
            return fastapi.Response(status_code=404)
        elif path == '/stats':
            return self.handle_stats_html()
        elif path == '/stats/data':
            return self.handle_stats_data(int(request.query_params.get("since_id", 0)))

        return await call_next(request)


# ===============  openai ================
class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    id: str = Field(default=None, description="Unique identifier for the request")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatCompletionMessage] = Field(..., description="List of messages")
    stream: bool = Field(default=False, description="Whether to stream responses")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")


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

class BaseSynapse(bt.Synapse):
    id: str | None = None
    project_id: str | None = None
    cid: str | None = None
    status_code: int | None = 200
    error: str | None = None
    elapsed_time: float | None = 0.0

class SyntheticNonStreamSynapse(BaseSynapse):
    question: str | None = None
    response: str | None = ''

    def get_question(self):
        return self.question

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
    
class OrganicNonStreamSynapse(BaseSynapse):
    completion: ChatCompletionRequest | None = None
    response: Optional[dict] = None

    def get_question(self):
        if not self.completion:
            return None
        user_messages = [msg for msg in self.completion.messages if msg.role == "user"]
        user_input = user_messages[-1].content
        return user_input