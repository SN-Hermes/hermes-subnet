from typing import List
from loguru import logger


class EMAUpdater:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

         # {uid: score}
        self.last_scores = {}

    def update(self, cur_uids: List[int], cur_scores: List[float]):
        cur_dict = dict(zip(cur_uids, cur_scores))
        new_scores = {}

        # find out all possible uids (including last and cur)
        all_uids = set(self.last_scores.keys()) | set(cur_dict.keys())
        
        for uid in all_uids:
            last_val = self.last_scores.get(uid, None)
            cur_val = cur_dict.get(uid, None)

            if last_val is None and cur_val is not None:
                # new uid -> last defaults to 0
                last_val = cur_val
            elif last_val is not None and cur_val is None:
                # disappeared uid -> cur defaults to 0
                cur_val = 0

            # calculate EMA
            new_scores[uid] = (1 - self.alpha) * last_val + self.alpha * cur_val

        self.last_scores = new_scores
        logger.info(f"EMA updated scores: {self.last_scores}")

        return new_scores
