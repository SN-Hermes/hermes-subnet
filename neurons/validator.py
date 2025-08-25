# The MIT License (MIT)
# Copyright © 2025 Subquery

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
import copy
import os
import sys

from fastapi import APIRouter, FastAPI, Request
import httpx
from loguru import logger
import netaddr
import numpy as np
import requests
import bittensor as bt
import torch
import uvicorn
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from common.subtensor import Settings



sys.path.append(os.path.abspath("/Users/demon/Desktop/work/onf/subql-graphql-agent/examples"))
from working_example import GraphQLAgent



app = FastAPI()
router = APIRouter()

@router.post("/miner_availabilities")
async def get_miner_availabilities(request: Request, uids: list[int] | None = None):
  return [1,2, 3]

app.include_router(router, prefix="/miners")

@app.get("/health")
def health():
    return {"status": "ok"}


class Validator:
    settings: Settings
    should_exit: bool
    
    miners: list[int] | None
    agent: GraphQLAgent | None
    llm: ChatOpenAI | None
    hotkeys: dict[int, str]  # uid to hotkey mapping
    scores: torch.Tensor
    device: str

    def __init__(self):
        Settings.load_env_file('validator')
        self.settings = Settings()
        self.should_exit = False

        self.miners = None
        self.agent = None

        self.hotkeys = copy.deepcopy(self.settings.metagraph.hotkeys)
        self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        self.device = 'cpu'

    async def start(self):
        tasks = [
            asyncio.create_task(
                self.refresh_uids()
            ),
            asyncio.create_task(
                self.start_api()
            ),
            asyncio.create_task(
                self.start_graphql_agent()
            ),
        ]
        await asyncio.gather(*tasks)

    async def start_api(self):
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

            await asyncio.create_task(self.serve_api())

        except Exception as e:
            logger.warning(f"Failed to serve scoring api to chain: {e}")
        logger.info("API started.")

    async def serve_api(self):
        try:
            self.settings.inspect()
            logger.info(f"Starting Scoring API on https://0.0.0.0:{self.settings.port}")
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.settings.port,
                loop="asyncio",
                reload=False,
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.warning(f"Failed to serve scoring api to chain: {e}")

    async def refresh_uids(self):
        while True:
            miners = self.settings.miners()
            logger.info(f"miners: {miners}")
            self.miners = miners
            if miners != self.miners:
                self.miners = miners
                logger.info(f"Updated miners: {self.miners}")
            await asyncio.sleep(20)

    async def start_graphql_agent(self):
        logger.info("Starting GraphQL agent...")
        try:
            self.agent = GraphQLAgent(
                "https://index-api.onfinality.io/sq/subquery/subquery-mainnet"
            )

            # await self.query()
            await self.loop_query()
        except Exception as e:
            logger.warning(f"Failed to start GraphQL agent: {e}")

    async def query(self):
        response = await self.agent.query('What projects are available?')
        logger.info(f"\n🤖 Agent: {response}")

    async def loop_query(self):
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=1
        )
        schema_file_path = "/Users/demon/Desktop/work/onf/subql-graphql-agent/examples/schema.graphql"
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            entity_schema = f.read()
    
        prompt = f"""You are given the following document as background context:
{entity_schema}
Your task:
1. Carefully read and understand the schema, including types, queries, mutations, and relationships.
2. Generate ONE natural question that a user might ask based on this file.
3. The question must be related to a numerical value (e.g., quantity, percentage, date, amount, measurement) that appears in the file.
4. The question must explicitly relate to "indexer".
5. Output only the question, nothing else.
6. Do not include explanations, answers, or more than one question.

Now generate the question:

"""
        await asyncio.sleep(10)  # wait for miners to be populated
        while True:
            summary_response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"\n🤖 LLM question: {summary_response.content}")

            question = summary_response.content.strip()

            # generate ground truth
            ground_truth = await self.generate_ground_truth(question)
            if not ground_truth:
                logger.warning("Failed to generate ground truth, skipping this round.")
                continue

            logger.info(f"\n🤖 LLM ground_truth: {ground_truth}")

            # query all miner
            tasks = []
            uids = self.settings.miners()
            for uid in uids:
                tasks.append(
                    asyncio.create_task(self.query_miner(uid, question))
                )
            responses = await asyncio.gather(*tasks)
            logger.info(f"responses: {responses}")

            # score result
            tasks = []
            for r in responses:
                tasks.append(
                    asyncio.create_task(self.get_score(ground_truth, r))
                )
            scores = await asyncio.gather(*tasks)
            scores = [float(s) for s in scores]
            logger.info(f"score result: {scores}")

            # keep score 
            self.set_weights(uids, scores)

            await asyncio.sleep(60 * 5)

    async def generate_ground_truth(self, question: str):
        response = await self.agent.query(question)
        return response

    async def query_miner(self, uid: int, question: str):
        try:
            axon_info = self.settings.metagraph.axons[uid]
            url = f"http://{axon_info.ip}:{axon_info.port}/v1/chat/completions"
            timeout = httpx.Timeout(3 * 60, connect=10, read=2 * 60)

            # payload = {
            #     "model": "gpt-4o-mini",
            #     "messages": [
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": question}
            #     ],
            #     "temperature": 1
            # }
            payload = { "question": question }
            logger.info(f"Querying miner {uid} at {url} with payload: {payload}")

            text = {}
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                text = response.json()

            logger.info(f"Miner {uid} response: {text}")
            response.raise_for_status()
            return text.get('response', '')

        except Exception as e:
            logger.warning(f"Failed to query miner {uid}: {e}")
            return ''


    async def get_score(self, ground_truth: str, miner_answer: str):
        prompt = f"""You are a strict fact-checking evaluator.  
Given a [Reference Answer] and a [Response], evaluate how factually close the Response is to the Reference Answer.  

Rules:  
1. Judge only based on factual correctness, not tone or style.  
2. Provide a single integer score between 0 and 10, where 0 = completely inconsistent, and 10 = perfectly consistent.  
3. You may use at most one decimal place (e.g., 7, 8.5, 10).
4. Output only the score as a number. Do not provide explanations or any extra text.  

[Reference Answer]:  
{ground_truth}  

[Response]:  
{miner_answer}  

Score:

        """
        summary_response = self.llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"\n🤖 LLM get_score: {summary_response.content}")
        return summary_response.content

    def set_weights(self, uids: list[int], scores: list[float]):
        logger.info(f"set_weights for uids: {uids}, scores: {scores}")

        scattered_scores: torch.FloatTensor = self.scores.scatter(
            0, torch.tensor(uids).to(self.device), torch.tensor(scores, dtype=torch.float32).to(self.device)
        ).to(self.device)
        
        logger.info(f"scattered_scores: {scattered_scores}")

        raw_weights = torch.nn.functional.normalize(scattered_scores, p=1, dim=0)
        logger.info(f"raw_weights: {raw_weights}")

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids = np.array(self.settings.metagraph.uids, dtype=np.int64),
                weights = np.array(raw_weights, dtype=np.float32),
                netuid=self.settings.netuid,
                subtensor=self.settings.subtensor,
                metagraph=self.settings.metagraph,
        )
        logger.info(f"processed_weight_uids: {processed_weight_uids}")
        logger.info(f"processed_weights: {processed_weights}")

        [suc, msg] = self.settings.subtensor.set_weights(
            wallet=self.settings.wallet,
            netuid=self.settings.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=10010,
        )
        logger.info(f"processed_weights: {suc, msg}")


    def check_registered(self):
        if not self.settings.subtensor.is_hotkey_registered(
            netuid=self.settings.netuid,
            hotkey_ss58=self.settings.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.settings.wallet} is not registered on netuid {self.settings.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()

    
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.start())