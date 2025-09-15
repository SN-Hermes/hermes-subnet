import asyncio
from collections import deque
from typing import Any
import time
from collections import defaultdict
import threading

from loguru import logger

from typing import TYPE_CHECKING

from common.protocol import OrganicNonStreamSynapse
if TYPE_CHECKING:
    from hermes.validator.challenge_manager import ChallengeManager

class BucketCounter:
    def __init__(self, window_hours=3):
        self.bucket_seconds = 3600 # 1 hour per bucket
        self.window_buckets = window_hours
        self.buckets = defaultdict(int)  # {bucket_id: count}
        self._lock = threading.Lock()

    def tick(self) -> int:
        now = int(time.time())
        bucket_id = now // self.bucket_seconds
        with self._lock:
            self.buckets[bucket_id] += 1
            return self.buckets[bucket_id]

    def count(self):
        now = int(time.time())
        current_bucket = now // self.bucket_seconds
        total = 0
        with self._lock:
            # calculate total in the last `window_buckets` buckets
            for i in range(self.window_buckets):
                total += self.buckets.get(current_bucket - i, 0)
        return total

    def cleanup(self):
        # Periodically clean up expired buckets to save memory
        now = int(time.time())
        min_bucket = (now // self.bucket_seconds) - self.window_buckets
        with self._lock:
            self.buckets = {k: v for k, v in self.buckets.items() if k >= min_bucket}


class WorkloadManager:
    uid_organic_response_history: dict[int, deque[OrganicNonStreamSynapse]]
    uid_organic_workload_counter: dict[int, BucketCounter]
    challenge_manager: "ChallengeManager"

    uid_sample_scores: dict[int, deque[float]]
    interval: int = 30  # seconds

    def __init__(self, challenge_manager: "ChallengeManager"):
        self.uid_organic_workload_counter = defaultdict(BucketCounter)
        self.uid_organic_response_history = defaultdict(lambda: deque(maxlen=20))
        
        self.uid_sample_scores = {}
        self.challenge_manager = challenge_manager

    def collect(self, uid: int, response: OrganicNonStreamSynapse):
        cur = self.uid_organic_workload_counter[uid].tick()

        # sample every 5th response
        if cur % 1 == 0:
            if uid not in self.uid_organic_response_history:
                self.uid_organic_response_history[uid] = deque(maxlen=20)
            self.uid_organic_response_history[uid].append(response)

        logger.info('after collect, uid_organic_response_history: {}', self.uid_organic_response_history)
        logger.info('uid, cur {}', (uid, cur))

    def compute_workload_score(self, uids):
        workload_counts = [self.uid_organic_workload_counter[uid].count() for uid in uids]
        min_workload = min(workload_counts) if workload_counts else 0
        max_workload = max(workload_counts) if workload_counts else 1

        scores = [0.0] * len(uids)
        for idx, uid in enumerate(uids):
            quantity = workload_counts[idx]
            quality_scores = self.uid_sample_scores.get(uid, [])

            # quality score（EMA）
            if not quality_scores:
                quality_ema = 0.0
            else:
                alpha = 0.7
                quality_ema = None
                for score in quality_scores:
                    if quality_ema is None:
                        quality_ema = score
                    else:
                        quality_ema = alpha * score + (1 - alpha) * quality_ema

            # normalized workload score
            if max_workload == min_workload:
                normalized_workload = 0.5 if len(uids) > 1 else 1.0
            else:
                normalized_workload = (quantity - min_workload) / (max_workload - min_workload)

            total_score = 0.5 * quality_ema + 0.5 * normalized_workload
            scores[idx] = total_score

        return scores

    async def compute_organic_task(self):
        while True:
            await asyncio.sleep(self.interval)

            try:
                logger.info(f"[WorkloadManager] Computing organic workload scores...{self.uid_organic_response_history}")

                for uid, responses in self.uid_organic_response_history.items():
                    if not responses:
                        logger.warning(f"[WorkloadManager] No responses to process for uid {uid}")
                        continue
                    r = responses.popleft()
                    q = r.completion.messages[-1].content

                    logger.info(f"[WorkloadManager] Computing organic workload score for uid {uid}, question: {q}")

                    success, ground_truth, ground_cost = await self.challenge_manager.generate_ground_truth(r.project_id, q)
                    if not success:
                        logger.warning(f"[WorkloadManager] Failed to generate ground truth {ground_truth}")
                        continue
                    logger.info(r)
                    logger.info(r.response)
                    zip_scores, _, _ = await self.challenge_manager.scorer_manager.compute_challenge_score(ground_truth, ground_cost, [r])

                    if uid not in self.uid_sample_scores:
                        self.uid_sample_scores[uid] = deque(maxlen=20)

                    self.uid_sample_scores[uid].append(zip_scores[0])
                    logger.info(f"[WorkloadManager] Updated organic workload score for uid {uid},{zip_scores[0]}, {self.uid_sample_scores}")
            except Exception as e:
                logger.error(f"[WorkloadManager] Error computing organic workload scores: {e}")