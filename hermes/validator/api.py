from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import bittensor as bt
from typing import Any
from loguru import logger
from common.protocol import ChatCompletionRequest, OrganicNonStreamSynapse, OrganicStreamSynapse, SyntheticStreamSynapse
from common.protocol import SyntheticStreamSynapse


app = FastAPI()
router = APIRouter()

@router.post("/{cid}/chat/completions")
async def chat(cid: str, request: Request, body: ChatCompletionRequest):
    logger.info(f"Received chat completion request for cid: {cid}, body: {body}")
    v: Any = request.app.state.validator
    dendrite: bt.Dendrite = v.dendrite

    if body.stream:
        async def streamer():
            miner_uid = 3
            synapse = OrganicStreamSynapse(project_id=cid, completion=body)
            responses = await dendrite.forward(
                axons=v.settings.metagraph.axons[miner_uid],
                synapse=synapse,
                deserialize=True,
                timeout=60*3,
                streaming=True,
            )
            async for part in responses:
                # logger.info(f"V3 got part: {part}, type: {type(part)}")
                yield part
        return StreamingResponse(streamer(), media_type="text/plain")

    miner_uid = 3
    synapse = OrganicNonStreamSynapse(project_id=cid, completion=body)
    response = await dendrite.forward(
        axons=v.settings.metagraph.axons[miner_uid],
        synapse=synapse,
        deserialize=False,
        timeout=60*3,
    )
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router, prefix="/miners")
