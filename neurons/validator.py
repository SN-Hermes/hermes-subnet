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
from pathlib import Path
import time
from typing import Any
from loguru import logger
import numpy as np
import bittensor as bt
import torch
import uvicorn
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from common.project_manager import ProjectManager
from common.prompt_template import SCORE_PROMPT
from common.protocol import SyntheticNonStreamSynapse
import common.utils as utils
from hermes.validator.question_generator import question_generator
from hermes.validator.api import app
from hermes.base import BaseNeuron
import agent.graphql_agent as subAgent


SUBQL_CID = 'QmfUNJC1Qz8m3F67sQmxrwjuSAu4WaCR1iBdPPdzBruQ7P'
class Validator(BaseNeuron):
    version: str = '5'

    server_agent: Any
    dendrite: bt.Dendrite
    miners: list[int] | None
    llm: ChatOpenAI | None
    scoreLLM: ChatOpenAI | None
    project_manager: ProjectManager | None
    hotkeys: dict[int, str]  # uid to hotkey mapping
    scores: torch.Tensor
    device: str

    @property
    def role(self) -> str:
        return "validator"
    
    def __init__(self):
        super().__init__()
        
        # Configure loguru to intercept and control third-party logging
        utils.configure_loguru()
        
        self.miners = []

        self.hotkeys = copy.deepcopy(self.settings.metagraph.hotkeys)
        self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        self.device = 'cpu'

        self.dendrite = bt.dendrite(wallet=self.settings.wallet)

    async def start(self):
        super().start()

        await self.init_project()

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

    async def init_project(self):
        model_name = os.getenv("LLM_MODEL", "gpt-5")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=1
        )

        score_model_name = os.getenv("SCORE_LLM_MODEL", "o3")
        self.scoreLLM = ChatOpenAI(
            model=score_model_name,
            temperature=1
        )
        logger.info(f"Using LLM model: {model_name} for synthetic challenge")
        logger.info(f"Using LLM model: {score_model_name} for scoring")

        current_dir = Path(__file__).parent
        project_dir = current_dir.parent / "projects" / "validator"
        self.project_manager = ProjectManager(project_dir)
        await self.project_manager.pull()

        self.server_agent = subAgent.initServerAgentWithConfig(self.project_manager.get_project(SUBQL_CID))
        # self.non_stream_chat_completion = subAgent.non_stream_chat_completion

    async def serve_api(self):
        try:
            external_ip = utils.try_get_external_ip()
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
            # logger.info(f"miners: {miners}")
            self.miners = miners
            if miners != self.miners:
                self.miners = miners
                logger.info(f"Updated miners: {self.miners}")
            await asyncio.sleep(30)

    async def loop_query(self):
        entity_schema = self.project_manager.get_project(SUBQL_CID).schema_content
        await asyncio.sleep(10)
    
        while True:
            # generate challenge
            question = question_generator.generate_question(SUBQL_CID, entity_schema, self.llm)
            trace_id = str(uuid4())

            logger.info(f"\n🤖 generate synthetic challenge: {question}", traceId=trace_id)

            # generate ground truth
            start_time = time.perf_counter()
            ground_truth: str = await self.generate_ground_truth(question)
            if not ground_truth:
                logger.warning("Failed to generate ground truth, skipping this round.", traceId=trace_id)
                await asyncio.sleep(60)
                continue

            # TODO: check ground truth has real content
            
            end_time = time.perf_counter()
            ground_cost = end_time - start_time
            logger.info(f"\n🤖 generate ground_truth: {ground_truth} cost: {ground_cost}s", traceId=trace_id)

            # query all miner
            tasks = []
            uids = self.settings.miners()
            uids = [uid for uid in uids if uid != self.uid]
            logger.info(f"query miners: {uids}")
            for uid in uids:
                tasks.append(
                    asyncio.create_task(self.query_miner(uid, trace_id, question, ground_truth))
                )
            responses = await asyncio.gather(*tasks)

            # score result
            tasks = []
            for r in responses:
                tasks.append(
                    asyncio.create_task(self.get_score(ground_truth, r))
                )
            scores = await asyncio.gather(*tasks)
            truth_scores = [float(s) for s in scores]
            logger.info(f" ground_truth scores: {truth_scores}")

            elapse_time = [r.elapsed_time for r in responses]
            logger.info(f" elapse_time: {elapse_time}")

            elapse_weights = [utils.get_elapse_weight_quadratic(r.elapsed_time, ground_cost) for r in responses]
            logger.info(f" elapse_weights: {elapse_weights}")

            weighted_scores = [s * w for s, w in zip(truth_scores, elapse_weights)]
            logger.info(f" zip scores: {weighted_scores}")

            # # # keep score 
            self.set_weights(uids, weighted_scores)

            await asyncio.sleep(60)

    async def generate_ground_truth(self, question: str):
        try:
            # response = await self.non_stream_chat_completion(
            #     self.server_agent,
            #    [{"role": "user", "content": question}],
            #     ChatCompletionRequest(
            #         messages=[{"role": "user", "content": question}],
            #         model="gpt-4o",
            #     )
            # )
            # logger.info(f"Generated ground truth response: {response.choices[0].message.content}")
            # return response.choices[0].message.content

            response = await self.server_agent.query_no_stream(question)
            # logger.info(f"Generated ground truth response: {response}")
            # todo: deal response
            return response.get('messages', [])[-1].content
            

        except Exception as e:
            logger.error(f"Error generating ground truth: {e}")
        return ''

    async def query_miner(self, uid: int, task_id: str, question: str, ground_truth: str):
        try:
            start_time = time.perf_counter()
            synapse = SyntheticNonStreamSynapse(id=task_id, projectId=SUBQL_CID, question=question)
            r = await self.dendrite.forward(
                axons=self.settings.metagraph.axons[uid],
                synapse=synapse,
                deserialize=False,
                timeout=60*3,
            )
            end_time = time.perf_counter()
            synapse.response = r.response
            logger.info(f"""
query_miner 
  miner: {uid}\n
  question: {question}\n
  answer: {synapse.response}\n
  ground_truth: {ground_truth}\n
  cost: {end_time - start_time}s
""")
            synapse.elapsed_time = end_time - start_time
            return synapse

        except Exception as e:
            logger.warning(f"Failed to query miner {uid}: {e}")
            return ''

    async def get_score(self, ground_truth: str, miner_synapse: SyntheticNonStreamSynapse):
        question_prompt = SCORE_PROMPT.format(
            ground_truth=ground_truth, 
            miner_answer=miner_synapse.response
        )
        # logger.debug(f"Generated question prompt for get_score: {question_prompt}")
        summary_response = self.scoreLLM.invoke([HumanMessage(content=question_prompt)])
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
                weights = raw_weights.detach().cpu().numpy().astype(np.float32),
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


if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.start())


