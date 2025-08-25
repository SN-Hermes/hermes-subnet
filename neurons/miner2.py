# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import os
import sys
import time
import bittensor as bt
from fastapi import APIRouter, FastAPI, Request
from loguru import logger
import netaddr
import requests
import uvicorn
from common.subtensor import Settings
from bittensor.core.extrinsics.serving import serve_extrinsic
from bittensor.core.axon import FastAPIThreadedServer


sys.path.append(os.path.abspath("/Users/demon/Desktop/work/onf/subql-graphql-agent/examples"))
from working_example import GraphQLAgent

class Miner:
    settings: Settings
    should_exit: bool
    agent: GraphQLAgent | None

    def __init__(self, config=None):
        Settings.load_env_file('miner2')
        self.settings = Settings()
        self.should_exit = False
        self.agent = None

    async def start_graphql_agent(self):
        logger.info("Starting GraphQL agent...")
        try:
            self.agent = GraphQLAgent(
                "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
            )

        except Exception as e:
            logger.warning(f"Failed to start GraphQL agent: {e}")

    async def start(self):
        external_ip = os.environ.get("EXTERNAL_IP", None)
        if not external_ip:
            try:
                external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
                netaddr.IPAddress(external_ip)
            except Exception:
                logger.error("Failed to get external IP")

        logger.info(
            f"Serving miner endpoint {external_ip}:{self.settings.port} on network: {self.settings.subtensor_network} with netuid: {self.settings.netuid}"
        )

        serve_success = serve_extrinsic(
            subtensor=self.settings.subtensor,
            wallet=self.settings.wallet,
            ip=external_ip,
            port=self.settings.port,
            protocol=4,
            netuid=self.settings.netuid,
        )
        if not serve_success:
            logger.error("Failed to serve endpoint")
            return
        
        await self.start_graphql_agent()

        app = FastAPI()
        router = APIRouter()
        router.add_api_route(
            "/v1/chat/completions",
            self.create_chat_completion,
            # dependencies=[Depends(self.verify_request)],
            methods=["POST"],
        )
        router.add_api_route(
            "/availability",
            self.check_availability,
            methods=["POST"],
        )
        router.add_api_route(
            "/health",
            self.check_health,
            methods=["GET"],
        )
        app.include_router(router)
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.settings.port,
            log_level="info",
            loop="asyncio",
            workers=4,
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        logger.info(f"Miner starting at block: {self.settings.subtensor.block}")

        # Main execution loop.
        try:
            while not self.should_exit:
                time.sleep(1)
        except Exception as e:
            logger.error(str(e))

    async def create_chat_completion(self, request: Request):
        data = await request.json()
        # response: str = await self.agent.query(data["question"])
        # logger.info(f"\n🤖 [Miner] Agent: {response}")
        return {"response": "I don't know the answer to that."}

    def check_availability(self):
        return {"error": "check_availability"}

    def check_health(self):
        return {"status": "ok"}

if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())
