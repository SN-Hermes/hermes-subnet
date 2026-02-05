import asyncio
from collections import deque
import json
import os
from pathlib import Path
import time
from collections import defaultdict
import threading
import traceback
from loguru import logger
from typing import TYPE_CHECKING
import torch
from multiprocessing.synchronize import Event
from agent.stats import TokenUsageMetrics
from common.enums import ChallengeType
from common.table_formatter import table_formatter
import common.utils as utils
from common.protocol import OrganicNonStreamSynapse
from hermes.validator.benchmark import BenchMark
if TYPE_CHECKING:
    from hermes.validator.challenge_manager import ChallengeManager
    from neurons.validator import Validator

class BucketCounter:
    def __init__(self, uid: int, hotkey: str, window_hours=3):
        self.uid = uid
        self.hotkey = hotkey
        self.bucket_seconds = 3600 # 1 hour per bucket
        self.window_buckets = window_hours
        self.buckets = defaultdict(int)  # {bucket_id: count}
        self._lock = threading.Lock()

    def tick(self, hotkey: str) -> int:
        now = int(time.time())
        bucket_id = now // self.bucket_seconds
        with self._lock:
            if hotkey != self.hotkey:
                self.buckets = defaultdict(int)
                self.hotkey = hotkey
        
            self.buckets[bucket_id] += 1
            return self.buckets[bucket_id]

    def count(self, hotkey: str | None = None):
        now = int(time.time())
        current_bucket = now // self.bucket_seconds
        total = 0
        with self._lock:
            if hotkey and hotkey != self.hotkey:
                self.buckets = defaultdict(int)
                self.hotkey = hotkey

            # calculate total in the last `window_buckets` buckets
            for i in range(self.window_buckets):
                total += self.buckets.get(current_bucket - i, 0)
        return total

    def cleanup(self):
        # Periodically clean up expired buckets to save memory
        now = int(time.time())
        min_bucket = (now // self.bucket_seconds) - self.window_buckets
        with self._lock:
            self.buckets = defaultdict(int, {k: v for k, v in self.buckets.items() if k >= min_bucket})

    def serialize(self) -> dict:
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            "buckets": dict(self.buckets)
        }
    
    @staticmethod
    def deserialize(data: dict) -> "BucketCounter":
        obj = BucketCounter(data["uid"], data["hotkey"])
        obj.buckets = defaultdict(int, data["buckets"])
        return obj

class WorkloadManager:
    uid_organic_workload_counter: dict[int, BucketCounter]
    challenge_manager: "ChallengeManager"
    organic_score_queue: list

    uid_sample_scores: dict[int, deque[float]]
    organic_task_compute_interval: int  # seconds
    organic_task_concurrency: int
    organic_task_sample_rate: int
    organic_workload_counter_full_purge_interval: int
    last_full_purge_time: int = int(time.time())
    work_state_path: str | Path = None
    token_usage_metrics: TokenUsageMetrics | None = None
    collect_count: int

    def __init__(
        self, 
        challenge_manager: "ChallengeManager", 
        organic_score_queue: list,
        work_state_path: str | Path = None,
        token_usage_metrics: TokenUsageMetrics = None,
        ipc_meta_config: dict = {},
        benchmark: BenchMark = None,
        event_stop: Event = None,
        v: "Validator" = None,
    ):
        self.challenge_manager = challenge_manager
        self.organic_score_queue = organic_score_queue
        self.token_usage_metrics = token_usage_metrics
        self.ipc_meta_config = ipc_meta_config
        self.benchmark = benchmark
        self.event_stop = event_stop
        self.V = v

        self.uid_sample_scores = {}
        # self.uid_organic_workload_counter = defaultdict(BucketCounter)
        self.uid_organic_workload_counter = {}

        self._purge_lock = asyncio.Lock()

        self.organic_task_compute_interval = int(os.getenv("WORKLOAD_ORGANIC_TASK_COMPUTE_INTERVAL", 30))
        self.organic_task_concurrency = int(os.getenv("WORKLOAD_ORGANIC_TASK_CONCURRENCY", 5))
        self.organic_task_sample_rate = int(os.getenv("WORKLOAD_ORGANIC_TASK_SAMPLE_RATE", 1))
        self.organic_workload_counter_full_purge_interval = int(os.getenv("WORKLOAD_ORGANIC_WORKLOAD_COUNTER_FULL_PURGE_INTERVAL", 3600))
        self.work_state_path = work_state_path
        self.collect_count = 0
        self.round_id = 1
        self.load_state()

    async def collect(self, uid: int, hotkey: str):
         async with self._purge_lock:
            if uid not in self.uid_organic_workload_counter:
                self.uid_organic_workload_counter[uid] = BucketCounter(uid, hotkey)

            cur = self.uid_organic_workload_counter[uid].tick(hotkey)

            self.collect_count += 1
            if self.collect_count % 10 == 0:
                self.save_state()
                self.collect_count = 0

            return cur

    async def purge(self, uids: list[int], hotkeys: list[str]):
        for uid, _ in zip(uids, hotkeys):
            if uid in self.uid_organic_workload_counter:
                self.uid_organic_workload_counter[uid].cleanup()

        now = int(time.time())
        if now - self.last_full_purge_time > self.organic_workload_counter_full_purge_interval:
            async with self._purge_lock:
                to_delete = []
                for uid, counter in list(self.uid_organic_workload_counter.items()):
                    counter.cleanup()
                    if counter.count() == 0:
                        to_delete.append(uid)
                for uid in to_delete:
                    del self.uid_organic_workload_counter[uid]
                self.last_full_purge_time = now
    
    async def compute_workload_score(
        self,
        uids: list[int],
        hotkeys: list[str],
        challenge_id: str = ""
    ) -> tuple[list[float], list[int], list[list[float]]]:
        await self.purge(uids, hotkeys)

        workload_counts = []
        for uid, hotkey in zip(uids, hotkeys):
            if uid not in self.uid_organic_workload_counter:
                self.uid_organic_workload_counter[uid] = BucketCounter(uid, hotkey)
            workload_counts.append(self.uid_organic_workload_counter[uid].count(hotkey))

        min_workload = min(workload_counts) if workload_counts else 0
        max_workload = max(workload_counts) if workload_counts else 1

        log_quality_scores = []

        scores = [0.0] * len(uids)
        for idx, uid in enumerate(uids):
            quantity = workload_counts[idx]
            uid_quality_scores = self.uid_sample_scores.get(uid, [])
            log_quality_scores.append(list(uid_quality_scores))

            # quality score（EMA）
            if not uid_quality_scores:
                quality_ema = 0.0
            else:
                alpha = 0.7
                quality_ema = None
                for score in uid_quality_scores:
                    if quality_ema is None:
                        quality_ema = score
                    else:
                        quality_ema = alpha * score + (1 - alpha) * quality_ema

            # normalized workload score
            if max_workload == min_workload:
                normalized_workload = 0 if min_workload == 0 else 0.5
            else:
                normalized_workload = (quantity - min_workload) / (max_workload - min_workload)

            scores[idx] = utils.fix_float(min(0.5 * quality_ema + 0.5 * normalized_workload, 5))

        logger.debug(f"[WorkloadManager] - {challenge_id} workload_counts: {workload_counts}, quality_scores: {log_quality_scores}, compute_workload_score: {scores}")
        return scores, workload_counts, log_quality_scores

    async def compute_organic_task(self):
        debug = os.getenv("DEBUG_ORGANIC_COUNTER", "0") == "1"

        while not self.event_stop.is_set():
            await asyncio.sleep(self.organic_task_compute_interval)

            if debug:
                info_lines = []
                for uid, counter in self.uid_organic_workload_counter.items():
                    info_lines.append(f"UID: {uid}, hotkey: {counter.hotkey}, buckets: {dict(counter.buckets)}")
                if len(info_lines) > 0:
                    logger.info("\n".join(info_lines))

            try:
                for i in range(self.organic_task_concurrency):
                    logger.debug(f"[WorkloadManager] Round {i+1}/{self.organic_task_concurrency} of computing organic workload scores")
                    
                    if self.organic_score_queue:
                        miner_uid, hotkey, resp_dict = self.organic_score_queue.pop(0)

                        logger.debug(f"[WorkloadManager] Processing organic task for miner: {miner_uid}, resp_dict id: {resp_dict}")
                        response = OrganicNonStreamSynapse(**resp_dict)

                        miner_uid_work_load = await self.collect(miner_uid, hotkey)
                        if miner_uid_work_load % self.organic_task_sample_rate != 0:
                            logger.debug(f"[WorkloadManager] Skipping organic task computation for miner: {miner_uid} at count {miner_uid_work_load}")
                            continue

                        question = response.get_question()
                        logger.debug(f"[WorkloadManager] compute organic task({response.id}) for miner: {miner_uid}, response: {response}. question: {question}")

                        success, ground_truth, ground_cost, metrics_data, model_name = await self.challenge_manager.generate_ground_truth(
                            cid_hash=response.cid_hash,
                            question=question,
                            token_usage_metrics=self.token_usage_metrics,
                            round_id=f"Organic-{self.round_id}",
                            block_height=response.block_height
                        )
                        # Validate ground truth content
                        is_valid = success and utils.is_ground_truth_valid(ground_truth)
                        if not is_valid:
                            logger.warning(f"[WorkloadManager] Invalid ground truth for task({response.id}), skipping quality scoring. Ground truth: {ground_truth}")
                            continue

                        logger.debug(f"[WorkloadManager] Generated task({response.id}) ground truth: {ground_truth}, cost: {ground_cost}, miner.response: {response.response}")

                        zip_scores, ground_truth_scores, elapse_weights, miners_elapse_time = await self.challenge_manager.scorer_manager.compute_challenge_score(
                            ground_truth,
                            ground_cost,
                            [response],
                            challenge_id=response.id,
                            cid_hash=response.cid_hash,
                            token_usage_metrics=self.token_usage_metrics,
                            min_latency_improvement_ratio=self.ipc_meta_config.get("min_latency_improvement_ratio", 0.2),
                            round_id=f"Organic-{self.round_id}",
                        )

                        table_formatter.create_workload_summary_table(
                            round_id=self.round_id,
                            challenge_id=response.id,
                            ground_truth=ground_truth,
                            uids=[miner_uid],
                            responses=[response],
                            ground_truth_scores=ground_truth_scores,
                            elapse_weights=elapse_weights,
                            zip_scores=zip_scores,
                            cid=response.cid_hash
                        )

                        await self.benchmark.upload(
                            uid=self.V.uid,
                            address=self.V.settings.wallet.hotkey.ss58_address,
                            version=self.V.settings.version,
                            cid=response.cid_hash.split('_')[0],
                            challenge_id=response.id,
                            challenge_type=ChallengeType.ORGANIC_STREAM.value,
                            question=response.get_question(),

                            question_generator_model_name='',
                            ground_truth_model_name=model_name[:50],
                            score_model_name=self.challenge_manager.scorer_manager.llm_score.model_name[:50],

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
                                "minerModelName": resp.miner_model_name[:50],
                                "graphqlAgentModelName": resp.graphql_agent_model_name[:50],
                                "elapsed": elapse_time,
                                "truthScore": truth_score,
                                "statusCode": resp.status_code,
                                "error": resp.error,
                                "answer": resp.response[:500] if resp.response and resp.status_code == 200 else None,
                                "inputTokens": resp.usage_info.get("input_tokens", 0) if resp.usage_info else 0,
                                "inputCacheReadTokens": resp.usage_info.get("input_cache_read_tokens", 0) if resp.usage_info else 0,
                                "outputTokens": resp.usage_info.get("output_tokens", 0) if resp.usage_info else 0,
                                "toolCalls": [
                                    parsed for t in resp.usage_info.get("tool_calls", []) if (parsed := utils.safe_json_loads(t)) is not None
                                ] if resp.usage_info else [],

                                "graphqlAgentInnerToolCalls": [
                                    parsed for t in resp.graphql_agent_inner_tool_calls if (parsed := utils.safe_json_loads(t)) is not None
                                ] if resp.graphql_agent_inner_tool_calls else [],
                            }
                            for uid, hotkey, elapse_time, truth_score, resp in zip([miner_uid], [hotkey], miners_elapse_time, ground_truth_scores, [response])
                        ],
                    )

                        if miner_uid not in self.uid_sample_scores:
                            self.uid_sample_scores[miner_uid] = deque(maxlen=20)

                        self.uid_sample_scores[miner_uid].append(zip_scores[0])
                        logger.info(f"[WorkloadManager] Updated organic workload score for uid {miner_uid},{zip_scores[0]}, {self.uid_sample_scores}")

                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"[WorkloadManager] Error computing organic workload scores: {e}\n{traceback.format_exc()}")

    def load_state(self):
        try:
            if not self.work_state_path or not os.path.exists(self.work_state_path):
                return

            state: dict = torch.load(str(self.work_state_path))
            timestamp = state.get("timestamp", 0)
            # only load state within 3 days
            if abs(int(time.time()) - timestamp) > 3 * 24 * 3600:
                    return
            
            if "works" in state:
                self.uid_organic_workload_counter = {
                    uid: BucketCounter.deserialize(counter_data) 
                    for uid, counter_data in state["works"].items()
                }
                workload_info = []
                for uid, counter in self.uid_organic_workload_counter.items():
                    workload_info.append(f"UID: {uid}, hotkey: {counter.hotkey}, total_workload: {counter.count(counter.hotkey)}, buckets: {dict(counter.buckets)}")
                logger.info(f"[WorkloadManager] Load state from {self.work_state_path}, works: {list(state['works'].keys())}\n" + "\n".join(workload_info))

        except Exception as e:
            logger.error(f"[WorkloadManager] Load state error: {e}")
    
    def save_state(self):
        try:
            if not self.work_state_path:
                return

            dir_path = os.path.dirname(self.work_state_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            state = {
                "timestamp": int(time.time()),
                "works": {uid: counter.serialize() for uid, counter in self.uid_organic_workload_counter.items()}
            }
            torch.save(state, str(self.work_state_path))
            logger.info(f"[WorkloadManager] Save state to {self.work_state_path}, works: {list(state['works'].keys())}")

        except Exception as e:
            logger.error(f"[WorkloadManager] Save state error: {e}")
