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
from typing import Any, Dict
from loguru import logger
import numpy as np
import bittensor as bt
import torch
import uvicorn
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from common.table_formatter import table_formatter

from common.project_manager import ProjectManager
from common.prompt_template import SCORE_PROMPT
from common.protocol import SyntheticNonStreamSynapse
import common.utils as utils
from hermes.validator.question_generator import question_generator
from hermes.validator.api import app
from hermes.base import BaseNeuron
import agent.graphql_agent as subAgent
from hermes.validator.ema import EMAUpdater


class Validator(BaseNeuron):
    version: str = '5'

    server_agent: Any
    server_agents: Dict[str, subAgent.GraphQLAgent]
    dendrite: bt.Dendrite
    miners: list[int] | None
    llm: ChatOpenAI | None
    scoreLLM: ChatOpenAI | None
    project_manager: ProjectManager | None
    hotkeys: dict[int, str]  # uid to hotkey mapping
    scores: torch.Tensor
    device: str
    last_score: list[float]
    ema: EMAUpdater
    _last_set_weight_time: float

    @property
    def role(self) -> str:
        return "validator"
    
    def __init__(self):
        super().__init__()
        
        self._last_set_weight_time = time.time()
        self.ema = EMAUpdater(alpha=0.7)
        self.miners = []

        self.hotkeys = copy.deepcopy(self.settings.metagraph.hotkeys)
        self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        self.device = 'cpu'

        self.dendrite = bt.dendrite(wallet=self.settings.wallet)
        
        # Configure synthetic challenge loop interval (default: 10 minutes)
        self.challenge_interval = int(os.getenv("CHALLENGE_INTERVAL", 600))  # seconds
        self.set_weight_interval = int(os.getenv("SET_WEIGHT_INTERVAL", 60 * 30))  # seconds
        logger.info(f"Synthetic challenge interval set to {self.challenge_interval} seconds")
        logger.info(f"Set weight interval set to {self.set_weight_interval} seconds")

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
            ),
            asyncio.create_task(
                self.set_weight()
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
        project_dir = current_dir.parent / "projects" / self.role
        self.project_manager = ProjectManager(project_dir)
        await self.project_manager.pull()

        self.init_agents()

    def init_agents(self):
        self.server_agents = {}
        for cid, project_config in self.project_manager.get_projects().items():
            self.server_agents[cid] = subAgent.initServerAgentWithConfig(project_config)
        logger.info(f"Initialized server_agents for projects: {list(self.server_agents.keys())}")

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
                log_config=None,  # Disable uvicorn's default logging config
                access_log=False,  # Disable access logs to reduce noise
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
        await asyncio.sleep(10)
    
        while True:
            projects = self.project_manager.get_projects()
            if not projects:
                logger.warning("No projects found, skipping this round.")
                await asyncio.sleep(self.challenge_interval)
                continue

            uids = [uid for uid in self.settings.miners() if uid != self.uid]

            project_score = []

            for cid, project_config in projects.items():
                # generate challenge
                question = question_generator.generate_question(cid, project_config.schema_content, self.llm)
                # question = await question_generator.generate_question_with_agent(cid, project_config.schema_content, self.server_agents[cid])
                trace_id = str(uuid4())

                # Create synthetic challenge table
                challenge_output = table_formatter.create_synthetic_challenge_table(question)
                table_formatter.log_with_newline(challenge_output, "info", traceId=trace_id)

                # generate ground truth
                start_time = time.perf_counter()
                ground_truth: str = await self.generate_ground_truth(cid, question)
                if not ground_truth:
                    project_score.append([1] * len(uids))
                    logger.warning("Failed to generate ground truth.", traceId=trace_id)
                    continue

                # TODO: check ground truth has real content
            
                end_time = time.perf_counter()
                ground_cost = end_time - start_time
                
                # Create ground truth tables
                ground_truth_output = table_formatter.create_ground_truth_tables(ground_truth, ground_cost)
                table_formatter.log_with_newline(ground_truth_output, "info", traceId=trace_id)

                logger.info(f"query miners: {uids}")
                # query all miner
                tasks = []
                for uid in uids:
                    tasks.append(
                        asyncio.create_task(self.query_miner(uid, cid, trace_id, question, ground_truth))
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

                zip_scores = [s * w for s, w in zip(truth_scores, elapse_weights)]
                logger.info(f" zip scores: {zip_scores}")

                project_score.append(zip_scores)


                project_score = np.array(project_score)
                logger.info(f"project_score: {project_score}")

                project_score = project_score.sum(axis=0)
                logger.info(f"project sum score: {project_score}")

                self.ema.update(uids, project_score.tolist())

            await asyncio.sleep(self.challenge_interval)


    async def set_weight(self):
        while True:
            await asyncio.sleep(10)
            if time.time() - self._last_set_weight_time > self.set_weight_interval:
                try:
                    uids = list(self.ema.last_scores.keys())
                    if not uids:
                        continue
                    scores = list(self.ema.last_scores.values())
                    self._set_weights(uids, scores)

                    self._last_set_weight_time = time.time()
                except Exception as e:
                    logger.error(f"Failed to set_weight: {e}")

    async def generate_ground_truth(self, cid: str, question: str):
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

            if cid not in self.server_agents:
                logger.warning(f"No server agent found for cid: {cid}")
                return ''

            server_agent = self.server_agents[cid]
            response = await server_agent.query_no_stream(question)
            # logger.info(f"Generated ground truth response: {response}")
            # todo: deal response
            return response.get('messages', [])[-1].content
            

        except Exception as e:
            logger.error(f"Error generating ground truth: {e}")
        return ''

    async def query_miner(self, uid: int, cid: str, task_id: str, question: str, ground_truth: str):
        try:
            start_time = time.perf_counter()
            synapse = SyntheticNonStreamSynapse(id=task_id, projectId=cid, question=question)
            r = await self.dendrite.forward(
                axons=self.settings.metagraph.axons[uid],
                synapse=synapse,
                deserialize=False,
                timeout=60*3,
            )
            end_time = time.perf_counter()
            synapse.response = r.response
            elapsed_time = end_time - start_time
            
            # Check if miner provided a response
            miner_answer = synapse.response.strip() if synapse.response and synapse.response.strip() else None
            miner_output = table_formatter.create_miner_response_tables(
                uid=uid,
                question=question,
                elapsed_time=elapsed_time,
                miner_answer=miner_answer,
                ground_truth=ground_truth if miner_answer else None
            )
            logger.info(miner_output)
            
            synapse.elapsed_time = elapsed_time
            return synapse

        except Exception as e:
            logger.warning("🔍 MINER RESPONSE [UID: {}] - ❌ Failed to query: {}", uid, e)
            return ''

    async def get_score(self, ground_truth: str, miner_synapse: SyntheticNonStreamSynapse):
        question_prompt = SCORE_PROMPT.format(
            ground_truth=ground_truth, 
            miner_answer=miner_synapse.response
        )
        # logger.debug(f"Generated question prompt for get_score: {question_prompt}")
        summary_response = self.scoreLLM.invoke([HumanMessage(content=question_prompt)])
        return summary_response.content

    def _set_weights(self, uids: list[int], scores: list[float]):
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


