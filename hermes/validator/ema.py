from loguru import logger

class EMAUpdater:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

        # {uid: (score, hotkey) }
        self._last_scores = {}

    def update(self, cur_uids: list[int], cur_hotkeys: list[str], cur_scores: list[float]):
        cur_dict_score = dict(zip(cur_uids, cur_scores))
        cur_dict_hotkeys = dict(zip(cur_uids, cur_hotkeys))
        new_scores = {}

        # find out all possible uids (including last and cur)
        all_uids = set(self._last_scores.keys()) | set(cur_dict_score.keys())
        
        for uid in all_uids:
            last_val, last_hk = self._last_scores.get(uid, (None, None))
            cur_val = cur_dict_score.get(uid, None)

            if last_val is None and cur_val is not None:
                # new uid -> last defaults to 0
                last_val = cur_val

            elif last_val is not None and cur_val is None:
                # disappeared uid -> cur defaults to 0
                cur_val = 0

            cur_hk = cur_dict_hotkeys.get(uid, None)
            if cur_hk and last_hk and cur_hk != last_hk:
                logger.info(f"UID {uid} hotkey changed from {last_hk} to {cur_hk}. Resetting EMA.")
                # hotkey changed, reset last_val to cur_val
                last_val = cur_val

            # calculate EMA
            new_scores[uid] = ((1 - self.alpha) * last_val + self.alpha * cur_val, cur_hk)

        self._last_scores = new_scores

        # logger.info(f"EMA updated scores: {self._last_scores}")
        return new_scores

    def load(self, state: dict[str, tuple[float, str]]):
        if not state:
            return
        self._last_scores = state
        return self._last_scores

    @property
    def last_scores(self):
        return self._last_scores

# if __name__ == "__main__":
#     ema = EMAUpdater(alpha=0.7)

#     print("first:")
#     # {1: 5.0, 2: 7.0, 3: 9.0}
#     print(ema.update([1, 2, 3], [5, 7, 9]))

#     print("second (add 4):")
#     # {1: 5.699999999999999, 2: 7.7, 3: 9.7, 4: 12.0}
#     print(ema.update([1, 2, 3, 4], [6, 8, 10, 12]))

#     print("third (remove 3):")
#     # {1: 8.71, 2: 16.310000000000002, 3: 2.91, 4: 3.6000000000000005}
#     print(ema.update([1, 2], [10, 20]))

#     print("fourth (3 disappeared, 4 returned):")
#     # {1: 8.213000000000001, 2: 13.293, 3: 0.8730000000000002, 4: 4.58}
#     print(ema.update([1, 2, 4], [8, 12, 5]))