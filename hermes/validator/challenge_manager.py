import asyncio
import os
from pathlib import Path
import time
from typing import Tuple
from uuid import uuid4
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from common.agent_manager import AgentManager
from common.protocol import OrganicNonStreamSynapse, SyntheticNonStreamSynapse
from common.settings import Settings
from common.table_formatter import table_formatter
from common.timer import Timer
from hermes.validator.question_generator import question_generator
from hermes.validator.scorer_manager import ScorerManager
from hermes.validator.workload_manager import WorkloadManager


class ChallengeManager:
    settings: Settings
    save_project_dir: Path
    uid: int
    challenge_interval: int
    dendrite: bt.Dendrite
    llm_synthetic: ChatOpenAI
    llm_score: ChatOpenAI
    agent_manager: AgentManager
    scorer_manager: ScorerManager
    workload_manager: WorkloadManager

    def __init__(
        self, 
        settings: Settings, 
        save_project_dir: str | Path, 
        uid: int, 
        dendrite: bt.Dendrite,
        synthetic_model_name: str | None = None,
        score_model_name: str | None = None,
    ):
        self.settings = settings

        # Configure synthetic challenge loop interval (default: 10 minutes)
        self.challenge_interval = int(os.getenv("CHALLENGE_INTERVAL", 600))  # seconds
        logger.info(f"[ChallengeManager] Synthetic challenge interval set to {self.challenge_interval} seconds")


        self.uid = uid
        self.dendrite = dendrite

        synthetic_model_name = synthetic_model_name or os.getenv("LLM_MODEL", "gpt-5")
        self.llm_synthetic = ChatOpenAI(
            model=synthetic_model_name,
            temperature=1
        )

        score_model_name = score_model_name or os.getenv("SCORE_LLM_MODEL", "o3")
        self.llm_score = ChatOpenAI(
            model=score_model_name,
            temperature=1
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm_synthetic,
            llm_score=self.llm_score
        )

        self.scorer_manager = ScorerManager(llm_score=self.llm_score)
        self.workload_manager = WorkloadManager(self)

        logger.info(f"[ChallengeManager] Using LLM model: {synthetic_model_name} for synthetic challenge")
        logger.info(f"[ChallengeManager] Using LLM model: {score_model_name} for scoring")

    async def start(self):
        # pull projects & init agents
        await self.agent_manager.start(pull=False)

        self.task = asyncio.create_task(self.workload_manager.compute_organic_task())

        while True:
            await asyncio.sleep(self.challenge_interval)

            projects = self.agent_manager.get_projects()
            if not projects:
                logger.warning("No projects found, skipping this round.")
                await asyncio.sleep(self.challenge_interval)
                continue

            uids = [uid for uid in self.settings.miners() if uid != self.uid]
            if not uids:
                logger.warning("No available miners for challenge.")
                await asyncio.sleep(self.challenge_interval)
                continue

            project_score_matrix = []

            for cid, project_config in projects.items():
                trace_id = str(uuid4())
                
                # generate challenge
                question = question_generator.generate_question(cid, project_config.schema_content, self.llm_synthetic)
                if not question:
                    continue

                # Create synthetic challenge table
                challenge_output = table_formatter.create_synthetic_challenge_table(question)
                table_formatter.log_with_newline(challenge_output, "info", traceId=trace_id)

                # generate ground truth
                success, ground_truth, ground_cost = await self.generate_ground_truth(cid, question)
                if not success:
                    logger.warning(f"Failed to generate ground truth. {trace_id}, {ground_truth}")
                    continue
                
                # Create ground truth tables
                ground_truth_output = table_formatter.create_ground_truth_tables(ground_truth, ground_cost)
                table_formatter.log_with_newline(ground_truth_output, "info", traceId=trace_id)

                # query all miner
                logger.info(f"query miners: {uids}")
                responses = await asyncio.gather(
                    *(self.query_miner(uid, cid, trace_id, question, ground_truth) for uid in uids)
                )

                # score result
                zip_scores, _, _ = await self.scorer_manager.compute_challenge_score(ground_truth, ground_cost, responses)
                project_score_matrix.append(zip_scores)

            workload_score = self.workload_manager.compute_workload_score(uids)
            logger.info(f"workload score: {workload_score}")
            
            self.scorer_manager.update_scores(uids, project_score_matrix, workload_score)
            # await asyncio.sleep(self.challenge_interval)

    async def generate_ground_truth(self, cid: str, question: str) -> Tuple[bool, str, int]:
        start_time = time.perf_counter()
        success = False
        ground_truth = ""
        try:
            agent = self.agent_manager.get_agent(cid)
            if not agent:
                ground_truth = f"No server agent found for cid: {cid}"
            else:
                response = await agent.query_no_stream(question)
                success = True
                ground_truth = response.get('messages', [])[-1].content
        except Exception as e:
            ground_truth = str(e)

        finally:
            return [success, ground_truth, time.perf_counter() - start_time]

    async def query_miner(
        self, 
        uid: int, 
        cid: str, 
        task_id: str, 
        question: str, 
        ground_truth: str
    ):
        synapse = SyntheticNonStreamSynapse(id=task_id, project_id=cid, question=question)
        try:
            with Timer() as t:
                r = await self.dendrite.forward(
                    axons=self.settings.metagraph.axons[uid],
                    synapse=synapse,
                    deserialize=False,
                    timeout=60*1,
                )
            elapsed_time = t.final_time
            synapse.response = r.response
            
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

        except Exception as e:
            logger.warning("🔍 MINER RESPONSE [UID: {}] - ❌ Failed to query: {}", uid, e)
            synapse.error = str(e)
        finally:
            return synapse

    def tick_organic(self, uid: int, response: OrganicNonStreamSynapse):
        self.workload_manager.collect(uid, response)