import os
import sys
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import json
import bittensor as bt
from typing import Any

from loguru import logger

from common.protocol import ChatCompletionRequest, OrganicStreamSynapse, SyntheticStreamSynapse

sys.path.append(os.path.abspath("/Users/demon/Desktop/work/onf/subql-graphql-agent/examples"))
from common.protocol import CapacitySynapse, SyntheticStreamSynapse
from server import stream_chat_completion


app = FastAPI()
router = APIRouter()

@router.post("/miner_availabilities")
async def get_miner_availabilities(request: Request, uids: list[int] | None = None):
  return [1,2, 3]

@router.post("/organic/generate")
async def organic_generate(request: Request):
    payload: dict[str, Any] = await request.json()
    message = payload.get("message", "")

    logger.info(f"Received  organic_generate message: {message}")
    v: Any = request.app.state.validator

    async def streamer():
        miner_uid = 3
        synapse = SyntheticStreamSynapse(question=message)
        responses = await v.dendrite.forward(
            axons=v.settings.metagraph.axons[miner_uid],
            synapse=synapse,
            deserialize=True,
            timeout=60*3,
            streaming=True,
        )

        async for part in responses:
            logger.info(f"V3 got part: {part}, type: {type(part)}")

            '''
            json.dumps() 只能序列化 Python 的基本类型（如 dict、list、str、int、float、bool、None), 不能直接序列化自定义类的实例。
            
            import json
            class A:
                def __init__(self):
                    self.x = 1

            a = A()
            json.dumps(a)  # ❌ 会报 TypeError: Object of type A is not JSON serializable

            如果你想序列化自定义对象，有几种办法：
            1. 转换成字典（手动或用 __dict__):
                json.dumps(a.__dict__)  # ✅ 输出 {"x": 1}

            2. 实现自定义序列化器:
                json.dumps(a, default=lambda o: o.__dict__)

            3. 使用 Pydantic 的 dict() 方法（如果你的类继承自 BaseModel 或类似 Munch):
               json.dumps(a.dict())  # ✅ 输出 {"x": 1}
            '''
            chunk = json.dumps({"text": part})

            yield (chunk + "\n").encode("utf-8")
    return StreamingResponse(streamer(), media_type="application/json")

@router.post("/{cid}/chat/completions")
async def chat(cid: str, request: Request, body: ChatCompletionRequest):
    logger.info(f"Received chat completion request for cid: {cid}, body: {body}")
    v: Any = request.app.state.validator
    dendrite: bt.Dendrite = v.dendrite

    # user_messages = [msg for msg in body.messages if msg.role == "user"]
    # if not user_messages:
    #     raise HTTPException(status_code=400, detail="No user message found")
    # user_input = user_messages[-1].content
    
    # return StreamingResponse(
    #   stream_chat_completion(v.agent, user_input, body),
    #   media_type="text/plain"
    # )

    async def streamer():
        miner_uid = 3
        synapse = OrganicStreamSynapse(completion=body)
        responses = await dendrite.forward(
            axons=v.settings.metagraph.axons[miner_uid],
            synapse=synapse,
            deserialize=True,
            timeout=60*3,
            streaming=True,
        )
        async for part in responses:
            logger.info(f"V3 got part: {part}, type: {type(part)}")
            yield part
    return StreamingResponse(streamer(), media_type="text/plain")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router, prefix="/miners")
