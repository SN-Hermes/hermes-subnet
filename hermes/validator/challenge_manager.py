import asyncio
import os
from pathlib import Path
import time
import traceback
from typing import Tuple, TYPE_CHECKING
from uuid import uuid4
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from multiprocessing.synchronize import Event
import numpy as np
import torch
if TYPE_CHECKING:
    from neurons.validator import Validator
from agent.stats import Phase, TokenUsageMetrics
from common.agent_manager import AgentManager
from common.errors import ErrorCode
from common.protocol import SyntheticNonStreamSynapse
from common.settings import Settings
from common.table_formatter import table_formatter
import common.utils as utils
from hermes.validator.question_generator import question_generator
from hermes.validator.scorer_manager import ScorerManager
from hermes.validator.workload_manager import WorkloadManager


class ChallengeManager:
    settings: Settings
    uid: int
    round_id: int
    challenge_interval: int
    dendrite: bt.Dendrite
    llm_synthetic: ChatOpenAI
    llm_score: ChatOpenAI
    agent_manager: AgentManager
    scorer_manager: ScorerManager
    workload_manager: WorkloadManager
    synthetic_score: list
    miners_dict: dict
    event_stop: Event
    scores: torch.Tensor
    token_usage_metrics: TokenUsageMetrics

    def __init__(
        self, 
        settings: Settings, 
        save_project_dir: str | Path, 
        uid: int, 
        dendrite: bt.Dendrite,
        organic_score_queue: list,
        synthetic_score: list,
        miners_dict: dict,
        synthetic_model_name: str | None = None,
        score_model_name: str | None = None,
        event_stop: Event = None,
        synthetic_token_usage: list = None,
        v: "Validator" = None,
    ):
        self.settings = settings

        # Configure synthetic challenge loop interval (default: 10 minutes)
        self.challenge_interval = int(os.getenv("CHALLENGE_INTERVAL", 60 * 10))  # seconds
        self.refresh_agents_interval = int(os.getenv("REFRESH_AGENTS_INTERVAL", 60 * 5))  # seconds

        self.forward_miner_timeout = int(os.getenv("FORWARD_MINER_TIMEOUT", 60 * 3))  # seconds
        logger.info(f"[ChallengeManager] Synthetic challenge interval set to {self.challenge_interval} seconds")

        self.uid = uid
        self.round_id = 1
        self.dendrite = dendrite
        self.token_usage_metrics = TokenUsageMetrics(datas=synthetic_token_usage)

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
        )

        self.scorer_manager = ScorerManager(
            llm_score=self.llm_score,
            score_state_path=Path(self.settings.base_dir) / ".data" / f"{v.role}_score_state.pt"
        )

        self.workload_manager = WorkloadManager(
            challenge_manager=self,
            organic_score_queue=organic_score_queue,
            work_state_path=Path(self.settings.base_dir) / ".data" / f"{v.role}_workload_state.pt",
            token_usage_metrics=self.token_usage_metrics
        )

        self.synthetic_score = synthetic_score
        self.miners_dict = miners_dict
        self.event_stop = event_stop

        self._last_set_weight_time = 0
        # self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        # self.device = 'cpu'
        self.set_weight_interval = int(os.getenv("SET_WEIGHT_INTERVAL", 60 * 30))  # seconds
        
        logger.info(f"[ChallengeManager] Set weight interval to {self.set_weight_interval} seconds")

        logger.info(f"[ChallengeManager] Using LLM model: {synthetic_model_name} for synthetic challenge")
        logger.info(f"[ChallengeManager] Using LLM model: {score_model_name} for scoring")
        logger.info(f"[ChallengeManager] Using KEY: {utils.format_openai_key()}")

    async def start(self):
        try:
            mode = os.getenv("PROJECT_PULL_MODE", "pull")

            # pull projects & init agents
            await self.agent_manager.start(mode == "pull", role="validator")

            self.task = [
                asyncio.create_task(self.workload_manager.compute_organic_task()),
                asyncio.create_task(self.set_weight()),
                asyncio.create_task(self.challenge_loop()),
                # asyncio.create_task(self.test_ground_truth_token()),
                asyncio.create_task(self.refresh_agents()),
            ]
            await asyncio.gather(*self.task)
        except KeyboardInterrupt:
            logger.info("[ChallengeManager] Starting process interrupted by user")
            # Cancel all running tasks
            if hasattr(self, 'task'):
                for task in self.task:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to complete cancellation
                await asyncio.gather(*self.task, return_exceptions=True)
            logger.info("[ChallengeManager] All tasks cancelled successfully")
            raise  # Re-raise to allow graceful shutdown at higher level
        except Exception as e:
            logger.error(f"[ChallengeManager] Failed to start challenge manager: {e}\n{traceback.format_exc()}")
            raise

    async def challenge_loop(self):
        try:
            while not self.event_stop.is_set():
                await asyncio.sleep(self.challenge_interval)

                projects = self.agent_manager.get_projects()
                if not projects:
                    logger.warning("[ChallengeManager] No projects found, skipping this round.")
                    await asyncio.sleep(self.challenge_interval)
                    continue

                miner_uids = []
                miner_hotkeys = []
                for uid, miner_info in self.miners_dict.items():
                    miner_uids.append(uid)
                    miner_hotkeys.append(miner_info["hotkey"])
                
                uids = []
                hotkeys = []
                for idx, u in enumerate(miner_uids):
                    if u != self.uid:
                        uids.append(u)
                        hotkeys.append(miner_hotkeys[idx])

                skip_query_miner = os.getenv("SKIP_QUERY_MINER", "false").lower() == "true"

                if not skip_query_miner and not uids:
                    logger.warning("[ChallengeManager] No available miners for challenge, skipping this round.")
                    await asyncio.sleep(self.challenge_interval)
                    continue

                project_score_matrix = []

                for cid_hash, project_config in projects.items():
                    allowed_cid_hashs = os.getenv("ALLOWED_PROJECT_CID_HASHS", "").split(",")
                    if allowed_cid_hashs and cid_hash not in allowed_cid_hashs:
                        logger.info(f"[ChallengeManager] - {cid_hash} Skipping project not in allowed list")
                        continue

                    # Retry loop: attempt to generate a valid challenge for this project
                    max_retries = int(os.getenv("CHALLENGE_GENERATION_MAX_RETRIES", 3))
                    challenge_generated = False

                    for attempt in range(max_retries):
                        challenge_id = str(uuid4())

                        # generate challenge
                        question = await question_generator.generate_question(
                            cid_hash, 
                            project_config.schema_content, 
                            self.llm_synthetic,
                            self.token_usage_metrics,
                            round_id=self.round_id
                        )
                        if not question:
                            logger.warning(f"[ChallengeManager] - {cid_hash} Failed to generate question (attempt {attempt + 1}/{max_retries})")
                            continue


                        success, ground_truth, ground_cost, metrics_data = await self.generate_ground_truth(cid_hash, question, self.token_usage_metrics,  round_id=self.round_id)

                        is_valid = success and utils.is_ground_truth_valid(ground_truth)

                        # Create challenge table
                        table_formatter.create_synthetic_challenge_table(
                            round_id=self.round_id,
                            challenge_id=challenge_id,
                            cid=cid_hash,
                            question=question,
                            success=is_valid,
                            ground_truth=ground_truth,
                            ground_cost=ground_cost,
                            metrics_data=utils.pick(metrics_data, ["phase", "input_tokens", "input_cache_read_tokens", "output_tokens", "timestamp", "round_id"])
                        )

                        if not is_valid:
                            logger.warning(f"[ChallengeManager] - {challenge_id} Invalid ground truth (attempt {attempt + 1}/{max_retries}): {ground_truth}")
                            continue

                        # Valid challenge generated, break retry loop
                        challenge_generated = True
                        break
                    # Skip this project if all retries failed
                    if not challenge_generated:
                        logger.error(f"[ChallengeManager] - {cid_hash} Failed to generate valid challenge after {max_retries} attempts, skipping project")
                        continue

                    if skip_query_miner:
                        continue

                    # query all miner
                    logger.info(f"[ChallengeManager] - {challenge_id} query miners: {uids}")
                    responses = await asyncio.gather(
                        *(self.query_miner(
                            uid=uid,
                            cid_hash=cid_hash,
                            challenge_id=challenge_id,
                            question=question
                        ) for uid in uids)
                    )

                    # score result
                    zip_scores, ground_truth_scores, elapse_weights = await self.scorer_manager.compute_challenge_score(
                        ground_truth,
                        ground_cost,
                        responses,
                        challenge_id=challenge_id,
                        cid_hash=cid_hash,
                        token_usage_metrics=self.token_usage_metrics,
                        round_id=self.round_id
                    )
                    project_score_matrix.append(zip_scores)

                    table_formatter.create_synthetic_miners_response_table(
                        round_id=self.round_id,
                        challenge_id=challenge_id,
                        uids=uids,
                        responses=responses,
                        ground_truth_scores=ground_truth_scores,
                        elapse_weights=elapse_weights,
                        zip_scores=zip_scores,
                        cid=cid_hash
                    )

                if not project_score_matrix:
                    logger.warning("[ChallengeManager] No valid project score matrix, skipping this round.")
                    continue

                workload_score, workload_counts, log_quality_scores = await self.workload_manager.compute_workload_score(uids, hotkeys, challenge_id=challenge_id)
                new_ema_scores = self.scorer_manager.update_scores(
                    uids,
                    hotkeys,
                    project_score_matrix,
                    workload_score,
                    challenge_id=challenge_id
                )
                self.synthetic_score[0] = self.scorer_manager.get_last_synthetic_scores()

                table_formatter.create_synthetic_final_ranking_table(
                    round_id=self.round_id,
                    challenge_id=challenge_id,
                    uids=uids,
                    hotkeys=hotkeys,
                    workload_counts=workload_counts,
                    quality_scores=log_quality_scores,
                    workload_score=workload_score,
                    new_ema_scores=new_ema_scores
                )
                self.round_id += 1

        except KeyboardInterrupt:
            logger.info("[ChallengeManager] Challenge loop interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[ChallengeManager] Challenge loop error: {e}\n{traceback.format_exc()}")
            raise

    async def test_ground_truth_token(self):
        try:
            questions = [
                'Which indexer currently has the highest total stake across all their delegations?',
                'What is the average commission rate (as a percentage) charged by indexers who have updated their rates within the last 10 eras?',
                'Which deployment has received the highest cumulative consumer boost in net tokens added minus removed?',
                # 'Which indexer currently has the highest total stake across all their delegations plus self-stake?'
            ]
            
            projects = self.agent_manager.get_projects()

            for cid_hash, project_config in projects.items():
                logger.info(f"start testing: {cid_hash}")
                
                total_input_tokens = 0
                total_input_cache_tokens = 0
                total_output_tokens = 0

                for q in questions:
                    challenge_id = str(uuid4())
                    success, ground_truth, ground_cost, metrics_data = await self.generate_ground_truth(cid_hash, q, self.token_usage_metrics,  round_id=self.round_id)

                    is_valid = success and utils.is_ground_truth_valid(ground_truth)

                    # Create challenge table
                    table_formatter.create_synthetic_challenge_table(
                        round_id=self.round_id,
                        challenge_id=challenge_id,
                        cid=cid_hash,
                        question=q,
                        success=is_valid,
                        ground_truth=ground_truth,
                        ground_cost=ground_cost,
                        metrics_data=metrics_data
                    )
                    self.round_id += 1

                    question_input = metrics_data.get("input_tokens", 0)
                    question_cache = metrics_data.get("input_cache_read_tokens", 0)
                    question_output = metrics_data.get("output_tokens", 0)
                    
                    total_input_tokens += question_input
                    total_input_cache_tokens += question_cache
                    total_output_tokens += question_output

                    await asyncio.sleep(2)  # brief pause between questions

                # calculate total cost for the project
                total_cost_info = utils.calculate_token_cost(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    input_cache_tokens=total_input_cache_tokens,
                    model_name=self.llm_synthetic.model_name
                )

                logger.info(f"=== project {cid_hash} summary ===")
                logger.info(f"Total Token: {total_cost_info['total_tokens']}")
                logger.info(f"Input Token: {total_input_tokens} (Actual: {total_cost_info['actual_input_tokens']}, cache: {total_input_cache_tokens})")
                logger.info(f"Output Token: {total_output_tokens}")
                logger.info(f"Total Cost: ${total_cost_info['total_cost']:.6f}")
                logger.info(f"Average Token Price: ${total_cost_info['avg_token_price']:.8f}")
                logger.info(f"Cost Breakdown - Input: ${total_cost_info['input_cost']:.6f}, Cache: ${total_cost_info['cache_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}")

        except KeyboardInterrupt:
            logger.info("[ChallengeManager] Challenge loop interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[ChallengeManager] Challenge loop error: {e}\n{traceback.format_exc()}")
            raise

    async def generate_ground_truth(
            self,
            cid_hash: str,
            question: str,
            token_usage_metrics: TokenUsageMetrics | None = None,
            round_id: int = 0
        ) -> Tuple[bool, str | None, int, dict | None]:
        start_time = time.perf_counter()
        success = False
        result = None
        metrics_data = None
        try:
            agent = self.agent_manager.get_graphql_agent(cid_hash)
            if not agent:
                raise ValueError(f"No server agent found for cid: {cid_hash}")

            response = await agent.query_no_stream(question, prompt_cache_key=f"{cid_hash}_{start_time}", is_synthetic=True)

            if os.getenv("LOG_GROUND_TRUTH", "").lower() == "true":
                logger.info(f'------------------- Ground Truth Response for CID {cid_hash} ------------------ {response}')

            result = response.get('messages', [])[-1].content

            if token_usage_metrics is not None:
                metrics_data = token_usage_metrics.append(cid_hash, phase=Phase.GENERATE_GROUND_TRUTH, response=response, extra = {"round_id": round_id})

            if not result:
                error = utils.try_get_invalid_tool_messages(response.get('messages', []))
                if error:
                    raise RuntimeError(f"[ChallengeManager] - {cid_hash} Failed to generate ground truth. {error}")

            success = True

        except KeyboardInterrupt:
            logger.info(f"[ChallengeManager] generate_ground_truth interrupted by user for cid: {cid_hash}")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            # Handle specific rate limit errors differently
            if isinstance(e, (dict, str)) and ('429' in str(e) or 'RATE_LIMIT_EXCEEDED' in str(e)):
                logger.warning(f"[ChallengeManager] Rate limit exceeded for cid: {cid_hash}. Will retry later. Error: {e}")
                raise  # Re-raise rate limit errors to allow retry logic
            else:
                logger.error(f"[ChallengeManager] generate_ground_truth error for cid: {cid_hash} {e}\n{traceback.format_exc()}")
                raise

        finally:
            return [success, result, utils.fix_float(time.perf_counter() - start_time), metrics_data]

    async def query_miner(
        self,
        uid: int,
        cid_hash: str,
        challenge_id: str,
        question: str,
    ):
        synapse = SyntheticNonStreamSynapse(id=challenge_id, cid_hash=cid_hash, question=question)
        start_time = time.perf_counter()

        # Initialize response object with error defaults
        r = SyntheticNonStreamSynapse(id=challenge_id, cid_hash=cid_hash, question=question)
        r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
        r.error = "Unknown error"

        try:
            r: SyntheticNonStreamSynapse = await self.dendrite.forward(
                axons=self.settings.metagraph.axons[uid],
                synapse=synapse,
                deserialize=False,
                timeout=self.forward_miner_timeout,
            )
            logger.debug(f"🔍 [ChallengeManager] - {challenge_id} MINER RESPONSE [UID: {uid}] - ✅ is_success: {r.is_success} - {r.dendrite.status_code}")
        except KeyboardInterrupt:
            logger.info(f"[ChallengeManager] - {challenge_id} Miner query interrupted by user [UID: {uid}]")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"🔍 [ChallengeManager] - {challenge_id} MINER RESPONSE [UID: {uid}] - ❌ Failed to query: {e}\n{traceback.format_exc()}")
            if not hasattr(r, 'status_code'):
                r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
            if not hasattr(r, 'error'):
                r.error = str(e)
        finally:
            r.elapsed_time = utils.fix_float(time.perf_counter() - start_time)
            return r

    async def set_weight(self):
        while True:
            await asyncio.sleep(10)
            if time.time() - self._last_set_weight_time > self.set_weight_interval:
                try:
                    scores_dict = self.scorer_manager.get_last_overall_scores()
                    uids = list(scores_dict.keys())
                    scores = list(scores_dict.values())
                    if not uids:
                        continue
                    scores = [s[0] for s in scores]
                    self._set_weights(uids, scores)

                    self._last_set_weight_time = time.time()
                except Exception as e:
                    logger.error(f"[ChallengeManager] Failed to set_weight: {e}")

    def _set_weights(self, uids: list[int], scores: list[float]):
        logger.info(f"[ChallengeManager] set_weights for uids: {uids}, scores: {scores}")
        scores_np = np.array(scores, dtype=np.float32)

        if np.all(scores_np == 0):
            logger.warning("[ChallengeManager] All scores are zero, skipping set weight.")
            return
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids = np.array(uids, dtype=np.int64),
                # weights = raw_weights.detach().cpu().numpy().astype(np.float32),
                weights = scores_np,
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
        logger.info(f"processed_weights result: {suc, msg}")

    async def refresh_agents(self):
        try:
            while not self.event_stop.is_set():
                await asyncio.sleep(self.refresh_agents_interval)
                await self.agent_manager.start(pull=True, role="validator")
        except Exception as e:
            logger.error(f"[ChallengeManager] refresh_agents error: {e}")
