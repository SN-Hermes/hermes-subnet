import asyncio
import os
from pathlib import Path
import time
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain_core.messages import HumanMessage
import numpy as np
import torch
from agent.stats import Phase, TokenUsageMetrics
from common import utils
from common.enums import ErrorCode
from common.prompt_template import SCORE_PROMPT, create_scoring_json
from common.prompt_injection_defense import sanitize_for_evaluation
from common.protocol import SyntheticNonStreamSynapse
from hermes.validator.ema import EMAUpdater


class ScorerManager:
    llm_score: ChatOpenAI
    overall_ema: EMAUpdater
    synthetic_ema: EMAUpdater
    score_state_path: str | Path

    def __init__(self, llm_score: ChatOpenAI, score_state_path: str | Path = None, ipc_meta_config: dict = None):
        self.overall_ema = EMAUpdater(alpha=0.7)
        self.synthetic_ema = EMAUpdater(alpha=0.7)
        self.llm_score = llm_score
        self.score_state_path = score_state_path
        self.ipc_meta_config = ipc_meta_config
        self.load_state()

    async def compute_challenge_score(
        self,
        ground_truth: str,
        ground_cost: float,
        miner_synapses: List[SyntheticNonStreamSynapse],
        challenge_id: str = "",
        cid_hash: str = "",
        token_usage_metrics: TokenUsageMetrics | None = None,
        min_latency_improvement_ratio: float = 0.2,
        round_id: int = 0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        elapse_time = [r.elapsed_time for r in miner_synapses]
        elapse_weights = [
            utils.fix_float(
                utils.get_elapse_weight_quadratic(
                    r.elapsed_time,
                    ground_cost,
                    min_latency_improvement_ratio
                )
            ) for r in miner_synapses
        ]
        
        # Only calculate ground truth scores for miners with non-zero elapse weights
        valid_miners = [(r, i) for i, (r, w) in enumerate(zip(miner_synapses, elapse_weights)) if w > 0]
        
        if valid_miners:
            valid_scores = await asyncio.gather(
                *(self.cal_ground_truth_score(ground_truth, r, cid_hash, token_usage_metrics, round_id=round_id) for r, _ in valid_miners)
            )
        else:
            valid_scores = []
        
        # Reconstruct ground_truth_scores_raw in original order
        ground_truth_scores_raw = ["0.0"] * len(miner_synapses)
        for (_, i), score in zip(valid_miners, valid_scores):
            ground_truth_scores_raw[i] = score
        
        ground_truth_scores = [min(utils.fix_float(utils.safe_float_convert(s)), 10.0) for s in ground_truth_scores_raw]
        zip_scores = [utils.fix_float(s * w) for s, w in zip(ground_truth_scores, elapse_weights)]

        logger.debug(f"[ScorerManager] - {challenge_id} ground_truth_scores: {ground_truth_scores_raw}, elapse_time: {elapse_time}, elapse_weights: {elapse_weights}, zip_scores: {zip_scores}")
        return zip_scores, ground_truth_scores, elapse_weights, elapse_time

    async def cal_ground_truth_score(
            self,
            ground_truth: str,
            miner_synapse: SyntheticNonStreamSynapse,
            cid_hash: str = "",
            token_usage_metrics: TokenUsageMetrics | None = None,
            round_id: int = 0
        ):
        if not miner_synapse.response or miner_synapse.status_code != 200:
            logger.debug(f"[ScorerManager] - cal_ground_truth_score: empty or error response from miner_synapse, status_code: {miner_synapse.status_code}, response: {miner_synapse.response}")
            return 0.0

        suspicious_uids = self.ipc_meta_config.get("suspicious_uids", []) if self.ipc_meta_config else []
        if miner_synapse.uid in suspicious_uids:
            logger.debug(f"[ScorerManager] - cal_ground_truth_score: miner_synapse {miner_synapse.uid} is in suspicious_uids")
            miner_synapse.status_code = ErrorCode.SUSPICIOUS.value
            miner_synapse.error = "Miner is suspicious"
            return 0.0

        # SECURITY: Sanitize miner response to detect/log prompt injection attempts
        # sanitized_response = sanitize_for_evaluation(miner_synapse.response, max_length=5000)
        
        json_data = create_scoring_json(ground_truth, miner_synapse.response)
        
        # Directly insert JSON data into the template to avoid format() conflicts with JSON braces
        question_prompt = SCORE_PROMPT.template.replace("{json_data}", json_data)
        try :
            summary_response = await self.llm_score.ainvoke([HumanMessage(content=question_prompt)])
            if token_usage_metrics is not None:
                d = token_usage_metrics.parse(
                    cid_hash, phase=Phase.GENERATE_MINER_GROUND_TRUTH_SCORE, response=summary_response, extra={"round_id": round_id}
                )
                token_usage_metrics.append(d)

        except Exception as e:
            logger.error(f"[ScorerManager] - LLM scoring error: {e}")
            return 0.0
        return summary_response.content.strip() if summary_response.content else "0.0"
    
    def update_scores(self, 
        uids: List[int], 
        hotkeys: List[str],
        project_score_matrix: List[List[float]],
        workload_score: List[float] | None,
        challenge_id: str = ""
    ):
        logger.debug(f"[ScorerManager] - {challenge_id} update_scores called with uids: {uids}, hotkeys: {hotkeys}, project_score_matrix: {project_score_matrix}, workload_score: {workload_score}")
        if not uids or not project_score_matrix:
            return

        suspicious_uids = self.ipc_meta_config.get("suspicious_uids", []) if self.ipc_meta_config else []
        synthetic_scores = np.array(project_score_matrix).sum(axis=0).tolist()
        self.synthetic_ema.update(uids, hotkeys, synthetic_scores, suspicious_uids)

        if workload_score is not None:
            merged = project_score_matrix + [workload_score]
        else:
            merged = project_score_matrix

        score_matrix = np.array(merged)
        score_matrix = score_matrix.sum(axis=0)
        
        new_scores = self.overall_ema.update(uids, hotkeys, score_matrix.tolist(), suspicious_uids)
        self.save_state(new_scores)
        logger.debug(f"[ScorerManager] - {challenge_id} uids: {uids}, project_score_matrix: {project_score_matrix}, workload_score: {workload_score}, merged: {merged}, score_matrix: {score_matrix.tolist()}, updated_ema_scores: {new_scores}")
        return new_scores


    def get_last_overall_scores(self):
        return self.overall_ema.last_scores

    def get_last_synthetic_scores(self):
        return self.synthetic_ema.last_scores

    def load_state(self):
        try:
            if not self.score_state_path or not os.path.exists(self.score_state_path):
                return

            state: dict = torch.load(str(self.score_state_path))
            timestamp = state.get("timestamp", 0)

            # only load state within 3 days
            if abs(int(time.time()) - timestamp) > 3 * 24 * 3600:
                return
            
            if "scores" in state:
                self.overall_ema.load(state["scores"])
                logger.info(f"[ScorerManager] Load state from {self.score_state_path}, scores: {state['scores']}")

        except Exception as e:
            logger.error(f"[ScorerManager] Load state error: {e}")

    def save_state(self, new_scores: dict[str, tuple[float, str]]):
        try:
            if not self.score_state_path:
                return

            dir_path = os.path.dirname(self.score_state_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            torch.save(
                {
                    "timestamp": int(time.time()),
                    "scores": new_scores,
                },
                str(self.score_state_path)
            )
        except Exception as e:
            logger.error(f"[ChallengeManager] Save state error: {e}")