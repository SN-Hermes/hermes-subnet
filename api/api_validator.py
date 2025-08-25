
import asyncio
import threading
from loguru import logger
import requests
import netaddr
import uvicorn
from fastapi import FastAPI, APIRouter, Request
from bittensor.core.extrinsics.serving import serve_extrinsic
from common.subtensor import settings

app = FastAPI()
router = APIRouter()

@router.post("/miner_availabilities")
async def get_miner_availabilities(request: Request, uids: list[int] | None = None):
  return [1,2, 3]

app.include_router(router, prefix="/miners")

@app.get("/health")
def health():
    return {"status": "ok"}


async def serve_api():
  try:
      settings.inspect()

      logger.info(f"Starting Scoring API on https://0.0.0.0:{settings.port}")
      config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.port,
        loop="asyncio",
        reload=False,
      )
      server = uvicorn.Server(config)
      await server.serve()
  except Exception as e:
      logger.warning(f"Failed to serve scoring api to chain: {e}")

async def start_api():
    logger.info("Starting API...")
    try:

        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        logger.info(f"external_ip: {external_ip}")

        netaddr.IPAddress(external_ip)

        
        # serve_success = serve_extrinsic(
        #   subtensor=subtensor,
        #   wallet=wallet,
        #   ip=external_ip,
        #   port=port,
        #   protocol=4,
        #   netuid=netuid,
        # )
        # logger.debug(f"Serve success: {serve_success}")

        await asyncio.create_task(serve_api())

    except Exception as e:
        logger.warning(f"Failed to serve scoring api to chain: {e}")
    logger.info("API started.")
