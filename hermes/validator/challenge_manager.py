import asyncio
import os
from pathlib import Path
import random
import time
import traceback
from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING, Optional
from uuid import uuid4
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from multiprocessing.synchronize import Event
import numpy as np
import torch

from hermes.validator.benchmark import BenchMark
if TYPE_CHECKING:
    from neurons.validator import Validator
from agent.stats import Phase, TokenUsageMetrics
from common.agent_manager import AgentManager
from common.enums import ChallengeType, ErrorCode, FailureType
from common.protocol import SyntheticNonStreamSynapse
from common.settings import Settings
from common.table_formatter import table_formatter
import common.utils as utils
from hermes.validator.question_generator import question_generator
from hermes.validator.scorer_manager import ScorerManager
from hermes.validator.workload_manager import WorkloadManager
from hermes.validator.dendrite import HighConcurrencyDendrite
from hermes.validator.multiprocess_query import query_miners_multiprocess


@dataclass
class EpochInfo:
    current_block: int
    tempo: int
    blocks_since_last_step: int
    epoch_start_block: int
    epoch_index: int
    next_epoch_start_block: int
    blocks_until_next_epoch: int


class ChallengeManager:
    settings: Settings
    uid: int
    round_id: int
    challenge_interval: int
    dendrite: HighConcurrencyDendrite
    llm_synthetic: ChatOpenAI
    llm_score: ChatOpenAI
    agent_manager: AgentManager
    scorer_manager: ScorerManager
    workload_manager: WorkloadManager
    ipc_synthetic_score: list
    ipc_miners_dict: dict
    ipc_meta_config: dict
    event_stop: Event
    scores: torch.Tensor
    token_usage_metrics: TokenUsageMetrics
    V: "Validator"

    def __init__(
        self, 
        settings: Settings, 
        save_project_dir: str | Path, 
        uid: int, 
        dendrite: HighConcurrencyDendrite,
        organic_score_queue: list,
        ipc_synthetic_score: list,
        ipc_miners_dict: dict,
        synthetic_model_name: str | None = None,
        score_model_name: str | None = None,
        ipc_meta_config: dict = None,
        ipc_common_config: dict = None,
        event_stop: Event = None,
        ipc_synthetic_token_usage: list = None,
        score_state_path: str | Path = None,
        work_state_path: str | Path = None,
        v: "Validator" = None,
    ):
        self.settings = settings

        # Configure synthetic challenge loop interval (default: 10 minutes)
        self.challenge_interval = int(os.getenv("CHALLENGE_INTERVAL", 60 * 20))  # seconds
        self.refresh_agents_interval = int(os.getenv("REFRESH_AGENTS_INTERVAL", 60 * 5))  # seconds

        self.forward_miner_timeout = int(os.getenv("FORWARD_MINER_TIMEOUT", 60 * 3))  # seconds
        logger.info(f"[ChallengeManager] Synthetic challenge interval set to {self.challenge_interval} seconds")

        self.uid = uid
        self.round_id = 1
        self.dendrite = dendrite
        self.token_usage_metrics = TokenUsageMetrics(datas=ipc_synthetic_token_usage)
        self.benchmark = BenchMark(self.settings.wallet, ipc_meta_config)

        synthetic_model_name = synthetic_model_name or os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
        self.llm_synthetic = ChatOpenAI(
            model=synthetic_model_name,
            temperature=1
        )

        score_model_name = score_model_name or os.getenv("SCORE_LLM_MODEL", "google/gemini-3-flash-preview")
        score_model_base_url = os.getenv("SCORE_LLM_MODEL_BASE_URL", None)
        score_model_api_key = os.getenv("SCORE_LLM_MODEL_API_KEY", None)
        score_model_args = {}
        if score_model_base_url:
            score_model_args["base_url"] = score_model_base_url
        if score_model_api_key:
            score_model_args["api_key"] = score_model_api_key

        self.llm_score = ChatOpenAI(
            model=score_model_name,
            temperature=0,
            **score_model_args
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm_synthetic,
            ipc_common_config=ipc_common_config,
        )

        self.scorer_manager = ScorerManager(
            llm_score=self.llm_score,
            score_state_path=score_state_path,
            ipc_meta_config=ipc_meta_config
        )

        self.workload_manager = WorkloadManager(
            challenge_manager=self,
            organic_score_queue=organic_score_queue,
            work_state_path=work_state_path,
            token_usage_metrics=self.token_usage_metrics,
            ipc_meta_config=ipc_meta_config or {},
            benchmark=self.benchmark,
            event_stop=event_stop,
            v=v,
        )

        self.ipc_synthetic_score = ipc_synthetic_score
        self.ipc_miners_dict = ipc_miners_dict
        self.ipc_meta_config = ipc_meta_config
        self.event_stop = event_stop
        self.V = v

        self._last_set_weight_time = 0
        self._last_epoch_submitted: Optional[int] = None
        self.block_time_seconds = float(os.getenv("CHAIN_BLOCK_TIME_SECONDS", 12))
        self.epoch_submission_buffer_seconds = int(os.getenv("EPOCH_SUBMISSION_BUFFER_SECONDS", 60))
        if self.block_time_seconds <= 0:
            logger.warning("[ChallengeManager] Invalid CHAIN_BLOCK_TIME_SECONDS, defaulting to 12 seconds")
            self.block_time_seconds = 12.0
        buffer_blocks = int(self.epoch_submission_buffer_seconds / self.block_time_seconds)
        self.epoch_submission_buffer_blocks = max(1, buffer_blocks)
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
            await self.benchmark.upload_os_info(
                uid=self.uid,
                address=self.settings.wallet.hotkey.ss58_address,
                version=self.settings.version,
                cpu_count=self.settings.cpu_count
            )

            self.task = [
                asyncio.create_task(self.workload_manager.compute_organic_task()),
                asyncio.create_task(self.set_weight()),
                asyncio.create_task(self.challenge_loop()),
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
            block_cache: dict[str, int] = {}
            miners_counter: dict[int, tuple[int, int]] = {}  # uid -> [success_count, total_count]
            challenge_interval = self.challenge_interval

            while not self.event_stop.is_set():
                await asyncio.sleep(challenge_interval)

                projects = self.agent_manager.get_projects()
                if not projects:
                    logger.warning("[ChallengeManager] No projects found, skipping this round.")
                    challenge_interval = 30
                    continue

                # Randomly shuffle miners
                miners_list = list(self.ipc_miners_dict.items())
                random.shuffle(miners_list)
                
                uids = []
                hotkeys = []
                ips = []
                axons: list[str] = []
                seen_ips = {}  # ip -> first uid that used it
                for uid, miner_info in miners_list:
                    if uid != self.uid:
                        uids.append(uid)
                        hotkeys.append(miner_info["hotkey"])
                        axons.append(miner_info["axon"])
                        ip = miner_info.get("ip")
                        ips.append(ip)
                        if ip and ip not in seen_ips:
                            seen_ips[ip] = uid

                skip_query_miner = os.getenv("SKIP_QUERY_MINER", "false").lower() == "true"

                if not skip_query_miner and not uids:
                    logger.warning("[ChallengeManager] No available miners for challenge, skipping this round.")
                    challenge_interval = 30
                    continue

                project_score_matrix = []
                organic_success_score_threshold = self.ipc_meta_config.get("organic_success_score_threshold", 5)

                for cid_hash, project_config in projects.items():
                    allowed_cid_hashs_str = os.getenv("ALLOWED_PROJECT_CID_HASHS", "").strip()
                    if allowed_cid_hashs_str:
                        allowed_cid_hashs = allowed_cid_hashs_str.split(",")
                        if cid_hash not in allowed_cid_hashs:
                            logger.info(f"[ChallengeManager] - {cid_hash} Skipping project not in allowed list")
                            continue

                    # Retry loop: attempt to generate a valid challenge for this project
                    max_retries = int(os.getenv("CHALLENGE_GENERATION_MAX_RETRIES", 3))
                    challenge_generated = False
                    error_msgs = []
                    weight_a = self.ipc_meta_config.get("weight_a", 70)
                    weight_b = self.ipc_meta_config.get("weight_b", 30)

                    for attempt in range(max_retries):
                        challenge_id = str(uuid4())

                        # get latest block
                        latest_block = await utils.get_latest_block(project_config.endpoint, project_config.node_type)
                        if latest_block is None and block_cache.get(cid_hash, None) is None:
                            logger.warning(f"[ChallengeManager] - {cid_hash} Failed to get latest block (attempt {attempt + 1}/{max_retries})")
                            error_msgs.append(f"(round: {self.round_id}, attempt: {attempt + 1}/{max_retries}, {cid_hash}) Failed to get latest block.")
                            continue
                        
                        if latest_block is not None:
                            block_cache[cid_hash] = latest_block - 1000
                        
                        # generate challenge
                        question, error = await question_generator.generate_question(
                            cid_hash, 
                            project_config,
                            self.llm_synthetic,
                            self.token_usage_metrics,
                            round_id=self.round_id,
                            weight_a=weight_a,    # normal
                            weight_b=weight_b     # tool
                        )

                        if not question:
                            logger.warning(f"[ChallengeManager] - {cid_hash} Failed to generate question (attempt {attempt + 1}/{max_retries})")
                            error_msgs.append(f"(round: {self.round_id}, attempt: {attempt + 1}/{max_retries}, {cid_hash}) {error}")
                            continue
                        logger.info(f"[ChallengeManager] - {cid_hash} Selected block height: {block_cache[cid_hash]}")

                        success, ground_truth, ground_cost, metrics_data, model_name = await self.generate_ground_truth(
                            cid_hash=cid_hash,
                            question=question,
                            token_usage_metrics=self.token_usage_metrics,
                            round_id=self.round_id,
                            block_height=block_cache[cid_hash]
                        )

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
                            # metrics_data=utils.pick(metrics_data, ["phase", "input_tokens", "input_cache_read_tokens", "output_tokens", "timestamp", "round_id"])
                            metrics_data=metrics_data
                        )

                        if not is_valid:
                            logger.warning(f"[ChallengeManager] - {challenge_id} Invalid ground truth (attempt {attempt + 1}/{max_retries}): {ground_truth}")
                            error_msgs.append(f"(round: {self.round_id}, attempt: {attempt + 1}/{max_retries}, {cid_hash}) Invalid ground truth: {ground_truth}")
                            continue

                        # Valid challenge generated, break retry loop
                        challenge_generated = True
                        break

                    # Skip this project if all retries failed
                    if not challenge_generated:
                        logger.error(f"[ChallengeManager] - {cid_hash} Failed to generate valid challenge after {max_retries} attempts")
                        await self.benchmark.add_failure(
                            uid= self.uid,
                            round_id= self.round_id,
                            address= self.settings.wallet.hotkey.ss58_address,
                            version= self.settings.version,
                            failure_type= FailureType.GENERATE_CHALLENGE.value,
                            cid_hash= cid_hash,
                            error_msgs= error_msgs
                        )
                        project_score_matrix.append([0.0] * len(uids))
                        continue

                    if skip_query_miner:
                        project_score_matrix.append([0.0] * len(uids))
                        continue

                    # query all miner
                    logger.info(f"[ChallengeManager] - {challenge_id} query miners: {uids}")

                    responses = await query_miners_multiprocess(
                        uids=uids,
                        hotkeys=hotkeys,
                        axons=axons,
                        ips=ips,
                        seen_ips=seen_ips,
                        cid_hash=cid_hash,
                        challenge_id=challenge_id,
                        question=question,
                        block_height=block_cache[cid_hash],
                        timeout=self.forward_miner_timeout,
                        settings=self.settings,
                    )

                    logger.info(f"[ChallengeManager] - {challenge_id} query miners done")
                    self.token_usage_metrics.append(metrics_data)

                    # score result
                    zip_scores, ground_truth_scores, elapse_weights, miners_elapse_time = await self.scorer_manager.compute_challenge_score(
                        ground_truth,
                        ground_cost,
                        responses,
                        challenge_id=challenge_id,
                        cid_hash=cid_hash,
                        token_usage_metrics=self.token_usage_metrics,
                        min_latency_improvement_ratio=self.ipc_meta_config.get("min_latency_improvement_ratio", 0.2),
                        round_id=self.round_id
                    )
                    project_score_matrix.append(zip_scores)

                    # update miners counter
                    for uid, truth_score in zip(uids, ground_truth_scores):
                        success_count, total_count = miners_counter.get(uid, (0, 0))
                        if truth_score >= organic_success_score_threshold:
                            success_count += 1
                        total_count += 1
                        miners_counter[uid] = (success_count, total_count)

                    table_formatter.create_synthetic_miners_response_table(
                        round_id=self.round_id,
                        challenge_id=challenge_id,
                        uids=uids,
                        hotkeys=hotkeys,
                        responses=responses,
                        ground_truth_scores=ground_truth_scores,
                        elapse_weights=elapse_weights,
                        zip_scores=zip_scores,
                        cid=cid_hash,
                        max_table_rows=int(os.getenv("MAX_TABLE_ROWS", 50))
                    )

                    await self.benchmark.upload(
                        uid=self.V.uid,
                        address=self.settings.wallet.hotkey.ss58_address,
                        version=self.settings.version,
                        cid=cid_hash.split('_')[0],
                        challenge_type=ChallengeType.SYNTHETIC.value,
                        challenge_id=challenge_id,
                        question=question,
                        question_generator_model_name=self.llm_synthetic.model_name,
                        ground_truth_model_name=model_name,
                        score_model_name=self.llm_score.model_name,
                        ground_truth=ground_truth[:500] if ground_truth else None,
                        ground_cost=ground_cost,
                        ground_truth_tools=[
                            parsed for t in metrics_data.get("tool_calls", []) if (parsed := utils.safe_json_loads(t)) is not None
                        ],
                        ground_input_tokens=metrics_data.get("input_tokens", 0),
                        ground_input_cache_read_tokens=metrics_data.get("input_cache_read_tokens", 0),
                        ground_output_tokens=metrics_data.get("output_tokens", 0),
                        
                        miners_answer=[
                            {
                                "uid": uid,
                                "address": hotkey,
                                "minerModelName": resp.miner_model_name[:50] if resp.miner_model_name else "",
                                "graphqlAgentModelName": resp.graphql_agent_model_name[:50] if resp.graphql_agent_model_name else "",
                                "elapsed": elapse_time,
                                "truthScore": truth_score,
                                "statusCode": resp.status_code,
                                "error": resp.error,
                                "answer": resp.response[:500] if resp.response else "",
                                "inputTokens": resp.usage_info.get("input_tokens", 0) if resp.usage_info else 0,
                                "inputCacheReadTokens": resp.usage_info.get("input_cache_read_tokens", 0) if resp.usage_info else 0,
                                "outputTokens": resp.usage_info.get("output_tokens", 0) if resp.usage_info else 0,
                                "toolCalls": [
                                    parsed for t in resp.usage_info.get("tool_calls", []) if (parsed := utils.safe_json_loads(t)) is not None
                                ] if resp.usage_info else [],
                                
                                "graphqlAgentInnerToolCalls": [
                                    parsed for t in resp.graphql_agent_inner_tool_calls 
                                    if (parsed := utils.safe_json_loads(t)) is not None
                                ] if resp.graphql_agent_inner_tool_calls else [],
                            }
                            for uid, hotkey, elapse_time, truth_score, resp in zip(uids, hotkeys, miners_elapse_time, ground_truth_scores, responses)
                            if resp.status_code != ErrorCode.NOT_HEALTHY.value
                        ],
                    )

                if not project_score_matrix:
                    logger.warning(f"[ChallengeManager] No valid project score matrix {self.round_id}")
                    challenge_interval = 30
                    continue

                workload_score, workload_counts, log_quality_scores = await self.workload_manager.compute_workload_score(uids, hotkeys, challenge_id=challenge_id)
                new_ema_scores = self.scorer_manager.update_scores(
                    uids,
                    hotkeys,
                    project_score_matrix,
                    workload_score,
                    challenge_id=challenge_id
                )
                self.ipc_synthetic_score[0] = self.scorer_manager.get_last_synthetic_scores()
                self.ipc_synthetic_score[1] = miners_counter

                table_formatter.create_synthetic_final_ranking_table(
                    round_id=self.round_id,
                    challenge_id=challenge_id,
                    uids=uids,
                    hotkeys=hotkeys,
                    workload_counts=workload_counts,
                    quality_scores=log_quality_scores,
                    workload_score=workload_score,
                    new_ema_scores=new_ema_scores,
                    max_table_rows=int(os.getenv("MAX_TABLE_ROWS", 50))
                )
                await self.benchmark.upload_ema(
                    uid=self.uid,
                    address=self.settings.wallet.hotkey.ss58_address,
                    version=self.settings.version,
                    round_id=self.round_id,
                    new_ema_scores=new_ema_scores,
                )
                self.round_id += 1
                challenge_interval = self.challenge_interval

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
            round_id: int = 0,
            block_height: int = 0,
        ) -> Tuple[bool, str | None, int, dict | None, str]:
        start_time = time.perf_counter()
        success = False
        result = None
        metrics_data = None
        model_name = ""
        try:
            agent = self.agent_manager.get_graphql_agent(cid_hash)
            if not agent:
                raise ValueError(f"No server agent found for cid: {cid_hash}")

            model_name = agent.llm.model_name
            response, _, _ = await agent.query_no_stream(
                question,
                prompt_cache_key=f"{cid_hash}_{start_time}",
                is_synthetic=True,
                block_height=block_height
            )

            if os.getenv("LOG_GROUND_TRUTH", "").lower() == "true":
                logger.info(f'------------------- Ground Truth Response for CID {cid_hash} ------------------ {response}')

            result = response.get('messages', [])[-1].content

            if token_usage_metrics is not None:
                metrics_data = token_usage_metrics.parse(cid_hash, phase=Phase.GENERATE_GROUND_TRUTH, response=response, extra={"round_id": round_id})
            if not result:
                error = utils.try_get_invalid_tool_messages(response.get('messages', []))
                raise RuntimeError(f"[ChallengeManager] - {cid_hash} Failed to generate ground truth. {error}")

            # data = utils.form_training_data(question, block_height, response.get('messages', []), metrics_data)
            # now = time.strftime("%Y-%m-%d", time.localtime())
            # utils.append_to_jsonl(f"./.data/dataset_validate_{now}_.jsonl", data)

            success = True

        except KeyboardInterrupt:
            logger.info(f"[ChallengeManager] generate_ground_truth interrupted by user for cid: {cid_hash}")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            # Handle specific rate limit errors differently
            if isinstance(e, (dict, str)) and ('429' in str(e) or 'RATE_LIMIT_EXCEEDED' in str(e)):
                logger.warning(f"[ChallengeManager] Rate limit exceeded for cid: {cid_hash}. Will retry later. Error: {e}")
            else:
                logger.error(f"[ChallengeManager] generate_ground_truth error for cid: {cid_hash} {e}\n{traceback.format_exc()}")
            
            result = f"{e}"

        finally:
            return [success, result, utils.fix_float(time.perf_counter() - start_time), metrics_data, model_name]

    async def query_miner(
        self,
        uid: int,
        hotkey: str,
        cid_hash: str,
        challenge_id: str,
        question: str,
        block_height: int = 0,
        is_ip_duplicated: bool = False,
    ):
        synapse = SyntheticNonStreamSynapse(id=challenge_id, uid=uid, cid_hash=cid_hash, question=question, block_height=block_height)
        start_time = time.perf_counter()

        # Initialize response object with error defaults
        r = SyntheticNonStreamSynapse(id=challenge_id, uid=uid, cid_hash=cid_hash, question=question, block_height=block_height)
        r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
        r.error = "Unknown error"

        try:
            if not hotkey:
                r.dendrite = bt.TerminalInfo(status_code=200)
                r.status_code = ErrorCode.NOT_HEALTHY.value
                r.error = "Miner is not healthy"
            elif is_ip_duplicated:
                r.dendrite = bt.TerminalInfo(status_code=200)
                r.status_code = ErrorCode.DUPLICATED_IP.value
                r.error = "Miner has duplicated IP"
            else:
                r: SyntheticNonStreamSynapse = await self.dendrite.forward(
                    axons=self.settings.metagraph.axons[uid],
                    synapse=synapse,
                    deserialize=False,
                    timeout=self.forward_miner_timeout,
                )
                logger.debug(f"🔍 [ChallengeManager] - {challenge_id} MINER RESPONSE [UID: {uid}] - ✅ is_success: {r.is_success} - {r.dendrite.status_code} - {r.dendrite.status_message}")
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
            r.uid = uid
            r.elapsed_time = utils.fix_float(time.perf_counter() - start_time)
            return r

    async def set_weight(self):
        while not self.event_stop.is_set():
            await asyncio.sleep(10)
            epoch_info = self._get_epoch_info()
            should_force_epoch_submission = self._should_force_epoch_submission(epoch_info)
            if not should_force_epoch_submission and time.time() - self._last_set_weight_time <= self.set_weight_interval:
                continue

            reason = "epoch-guard" if should_force_epoch_submission else "interval"
            try:
                uids, scores = self._prepare_scores_for_submission()
                if not uids:
                    uids, scores = self._build_fallback_uniform_weights()
                    if not uids:
                        logger.warning("[ChallengeManager] No miners available for fallback weight submission, skipping.")
                        self._last_set_weight_time = time.time()
                        continue
                    logger.info("[ChallengeManager] No historical scores available. Submitting uniform fallback weights.")

                self._set_weights(uids, scores)
                self._last_set_weight_time = time.time()
                if epoch_info:
                    self._last_epoch_submitted = epoch_info.epoch_index
                logger.info(f"[ChallengeManager] Submitted weights due to {reason}.")
            except Exception as e:
                logger.error(f"[ChallengeManager] Failed to set_weight: {e}")

    def _get_epoch_info(self) -> Optional[EpochInfo]:
        try:
            meta_info: bt.MetagraphInfo = self.settings.subtensor.get_metagraph_info(
                netuid=self.settings.netuid,
                field_indices=[
                    bt.SelectiveMetagraphIndex.Block,
                    bt.SelectiveMetagraphIndex.Tempo,
                    bt.SelectiveMetagraphIndex.BlocksSinceLastStep,
                ]
            )
        except Exception as e:
            logger.warning(f"[ChallengeManager] Unable to get epoch info: {e}")
            return None

        if meta_info is None:
            logger.warning("[ChallengeManager] Meta info is unavailable.")
            return None

        current_block = meta_info.block
        tempo = meta_info.tempo
        blocks_since_last_step = meta_info.blocks_since_last_step

        if not current_block:
            current_block = getattr(self.settings.subtensor, "block", None)

        if not current_block:
            logger.warning("[ChallengeManager] Current block is unavailable.")
            return None

        epoch_start_block = current_block - blocks_since_last_step
        epoch_index = epoch_start_block // tempo if tempo else 0
        next_epoch_start_block = epoch_start_block + tempo
        blocks_until_next_epoch = max(0, next_epoch_start_block - current_block)

        return EpochInfo(
            current_block=current_block,
            tempo=tempo,
            blocks_since_last_step=blocks_since_last_step,
            epoch_start_block=epoch_start_block,
            epoch_index=epoch_index,
            next_epoch_start_block=next_epoch_start_block,
            blocks_until_next_epoch=blocks_until_next_epoch,
        )

    def _should_force_epoch_submission(self, epoch_info: Optional[EpochInfo]) -> bool:
        if epoch_info is None:
            return False

        if self._last_epoch_submitted is None:
            return epoch_info.blocks_until_next_epoch <= self.epoch_submission_buffer_blocks

        if epoch_info.epoch_index > self._last_epoch_submitted and epoch_info.blocks_until_next_epoch <= self.epoch_submission_buffer_blocks:
            return True

        return False

    def _prepare_scores_for_submission(self) -> tuple[list[int], list[float]]:
        scores_dict = self.scorer_manager.get_last_overall_scores()
        if not scores_dict:
            return [], []
        sorted_scores = sorted(scores_dict.items(), key=lambda item: item[0])
        uids = [uid for uid, _ in sorted_scores]
        scores = [score for _, (score, _) in sorted_scores]
        return uids, scores

    def _build_fallback_uniform_weights(self) -> tuple[list[int], list[float]]:
        miner_uids = [uid for uid in list(self.ipc_miners_dict.keys()) if uid != self.uid]
        if not miner_uids:
            return [], []

        count = len(miner_uids)
        # uniform_weight = 1.0 / count
        return miner_uids, [0] * count

    def _set_weights(self, uids: list[int], scores: list[float]):
        logger.info(f"[ChallengeManager] set_weights for uids: {uids}, scores: {scores}")
        scores_np = np.array(scores, dtype=np.float32)

        if np.all(scores_np == 0):
            logger.warning("[ChallengeManager] All scores are zero, burning weights.")
            burn_uid = self.settings.burn_uid
            burn_uids_np = np.array([burn_uid], dtype=np.int64)
            burn_weights_np = np.array([1.0], dtype=np.float32)
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=burn_uids_np,
                weights=burn_weights_np,
                netuid=self.settings.netuid,
                subtensor=self.settings.subtensor,
                metagraph=self.settings.metagraph,
            )
        else:
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
                self.settings.reread()
                await self.agent_manager.start(pull=True, role="validator", silent=True)
        except Exception as e:
            logger.error(f"[ChallengeManager] refresh_agents error: {e}")
