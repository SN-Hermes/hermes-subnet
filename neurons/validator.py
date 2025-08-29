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
import httpx
from loguru import logger
import numpy as np
import bittensor as bt
import torch
import uvicorn
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from common.prompt_template import SCORE_PROMPT, SYNTHETIC_PROMPT
from common.protocol import SyntheticNonStreamSynapse
from common.utils import try_get_external_ip
from herms.validator.api import app
from herms.base import BaseNeuron


class Validator(BaseNeuron):
    version: str = '4'

    dendrite: bt.Dendrite
    miners: list[int] | None
    llm: ChatOpenAI | None
    hotkeys: dict[int, str]  # uid to hotkey mapping
    scores: torch.Tensor
    device: str

    @property
    def role(self) -> str:
        return "validator"
    
    def __init__(self):
        super().__init__()
        self.miners = None

        self.hotkeys = copy.deepcopy(self.settings.metagraph.hotkeys)
        self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        self.device = 'cpu'

        self.dendrite = bt.dendrite(wallet=self.settings.wallet)

    async def start(self):
        super().start()

        tasks = [
            asyncio.create_task(
                self.refresh_miners()
            ),
            asyncio.create_task(
                self.serve_api()
            ),
            asyncio.create_task(
                self.loop_query()
            )
        ]
        await asyncio.gather(*tasks)
            
    async def serve_api(self):
        try:
            external_ip = try_get_external_ip()
            logger.info(f"external_ip: {external_ip}")

            logger.info(f"Starting serve API on http://0.0.0.0:{self.settings.port}")
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.settings.port,
                loop="asyncio",
                reload=False,
            )
            app.state.validator = self

            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.warning(f"Failed to serve API: {e}")

    async def refresh_miners(self):
        while True:
            miners = self.settings.miners()
            logger.info(f"miners: {miners}")
            self.miners = miners
            if miners != self.miners:
                self.miners = miners
                logger.info(f"Updated miners: {self.miners}")
            await asyncio.sleep(30)

    async def loop_query(self):
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=1
        )
        schema_file_path = "/Users/demon/Desktop/work/onf/subql-graphql-agent/examples/schema.graphql"
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            entity_schema = f.read()
    
        question_prompt = SYNTHETIC_PROMPT.format(entity_schema=entity_schema)
        await asyncio.sleep(10)

        while True:
            # generate challenge
            summary_response = self.llm.invoke([HumanMessage(content=question_prompt)])
            question = summary_response.content.strip()
            logger.info(f"\n🤖 generate sythetic challenge: {question}")

            # generate ground truth
            ground_truth = await self.generate_ground_truth(question)
            if not ground_truth:
                logger.warning("Failed to generate ground truth, skipping this round.")
                continue

            logger.info(f"\n🤖 generate ground_truth: {ground_truth}")

            # query all miner
            tasks = []
            uids = self.settings.miners()
            for uid in uids:
                if uid == self.uid:
                    continue
                tasks.append(
                    asyncio.create_task(self.query_miner(uid, question))
                )
            responses = await asyncio.gather(*tasks)
            logger.info(f"responses: {responses}")

            # # score result
            tasks = []
            for r in responses:
                tasks.append(
                    asyncio.create_task(self.get_score(ground_truth, r))
                )
            scores = await asyncio.gather(*tasks)
            scores = [float(s) for s in scores]
            logger.info(f"score result: {scores}")

            # # keep score 
            self.set_weights(uids, scores)

            await asyncio.sleep(60 * 5)

    async def generate_ground_truth(self, question: str):
        response = await self.exampleAgent.query(question)
        return response

    async def query_miner(self, uid: int, question: str):
        try:
            synapse = SyntheticNonStreamSynapse(question=question)
            response = await self.dendrite.forward(
                axons=self.settings.metagraph.axons[uid],
                synapse=synapse,
                deserialize=False,
                timeout=60*3,
            )
            logger.info(f"query_miner miner {uid}, question: {question} response: {response}")

            return response.response

        except Exception as e:
            logger.warning(f"Failed to query miner {uid}: {e}")
            return ''

    async def get_score(self, ground_truth: str, miner_answer: str):
        question_prompt = SCORE_PROMPT.format(
            ground_truth=ground_truth, 
            miner_answer=miner_answer
        )
        summary_response = self.llm.invoke([HumanMessage(content=question_prompt)])
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


