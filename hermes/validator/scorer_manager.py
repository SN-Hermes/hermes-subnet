import asyncio
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from loguru import logger
from langchain.schema import HumanMessage
import numpy as np
from common import utils
from common.prompt_template import SCORE_PROMPT
from common.protocol import SyntheticNonStreamSynapse
from testing.ema import EMAUpdater


class ScorerManager:
    llm_score: ChatOpenAI
    ema: EMAUpdater

    def __init__(self, llm_score: ChatOpenAI):
        self.ema = EMAUpdater(alpha=0.7)
        self.llm_score = llm_score

    async def compute_challenge_score(self, 
        ground_truth: str, 
        ground_cost: float, 
        miner_synapses: List[SyntheticNonStreamSynapse]
    ) -> Tuple[List[float], List[float], List[float]]:
        ground_truth_scores = await asyncio.gather(
            *(self.cal_ground_truth_score(ground_truth, r) for r in miner_synapses)
        )
        ground_truth_scores = [float(s) for s in ground_truth_scores]
        logger.info(f" ground_truth scores: {ground_truth_scores}")

        elapse_time = [r.elapsed_time for r in miner_synapses]
        logger.info(f" elapse_time: {elapse_time}")

        elapse_weights = [utils.get_elapse_weight_quadratic(r.elapsed_time, ground_cost) for r in miner_synapses]
        logger.info(f" elapse_weights: {elapse_weights}")

        zip_scores = [s * w for s, w in zip(ground_truth_scores, elapse_weights)]
        logger.info(f" zip scores: {zip_scores}")

        return zip_scores, ground_truth_scores, elapse_weights

    async def cal_ground_truth_score(self, ground_truth: str, miner_synapse: SyntheticNonStreamSynapse):
        question_prompt = SCORE_PROMPT.format(
            ground_truth=ground_truth, 
            miner_answer=miner_synapse.response
        )
        summary_response = self.llm_score.invoke([HumanMessage(content=question_prompt)])
        return summary_response.content
    
    def update_scores(self, 
        uids: List[int], 
        project_score_matrix: List[List[float]],
        workload_score: List[float] | None
    ):
        if not uids or not project_score_matrix:
            return

        if workload_score is not None:
            merged = project_score_matrix + [workload_score]
        else:
            merged = project_score_matrix

        logger.info(f"project_score_matrix; {project_score_matrix}")

        logger.info(f"merged; {merged}")
        score_matrix = np.array(merged)
        logger.info(f"score_matrix: {score_matrix}")

        score_matrix = score_matrix.sum(axis=0)
        logger.info(f"project sum score: {score_matrix}")

        self.ema.update(uids, score_matrix.tolist())

    def get_last_scores(self):
        return self.ema.last_scores