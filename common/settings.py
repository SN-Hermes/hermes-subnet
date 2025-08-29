
import os
import time
from typing import List
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from loguru import logger
import dotenv
import bittensor as bt


class Settings:
    _subtensor: Subtensor | None = None
    _wallet: bt.wallet | None = None
    _last_metagraph: Metagraph = None
    _last_update_time: float = 0

    @classmethod
    def load_env_file(cls, role: str):
        env_file = f".env.{role}"
        dotenv.load_dotenv(env_file)
        logger.info(f"Loaded {env_file} file")

    @property
    def subtensor(self) -> Subtensor:
        if self._subtensor is not None:
            return self._subtensor
        subtensor_network = os.environ.get("SUBTENSOR_NETWORK", "local")
        if subtensor_network.lower() == "local":
            subtensor_network = os.environ.get("SUBTENSOR_CHAIN_ENDPOINT")
        else:
            subtensor_network = subtensor_network.lower()
        logger.info(f"Instantiating subtensor with network: {subtensor_network}")
        self._subtensor = Subtensor(network=subtensor_network)
        return self._subtensor
    
    @property
    def metagraph(self) -> Metagraph:
        if time.time() - self._last_update_time > 1200:
            try:
                logger.info(f"Fetching new METAGRAPH for NETUID={self.netuid}")
                meta = self.subtensor.metagraph(netuid=self.netuid)
                self._last_metagraph = meta
                self._last_update_time = time.time()
                return meta
            except Exception as e:
                logger.error(f"Failed to fetch new METAGRAPH for NETUID={self.NETUID}: {e}")
                if self._last_metagraph is not None:
                    logger.warning("Falling back to the previous METAGRAPH.")
                    return self._last_metagraph
                else:
                    logger.error("No previous METAGRAPH is available; re-raising exception.")
                    raise
        else:
            return self._last_metagraph
        
    @property
    def wallet(self):
        if self._wallet is not None:
            return self._wallet

        wallet_name = os.environ.get("WALLET_NAME")
        hotkey = os.environ.get("HOTKEY")
        logger.info(f"Instantiating wallet with name: {wallet_name}, hotkey: {hotkey}")
        self._wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
        return self._wallet

    @property
    def netuid(self) -> int:
        return int(os.environ.get("NETUID"))

    @property
    def port(self) -> int:
        return int(os.environ.get("PORT", "8085"))

    @property
    def external_ip(self) -> str | None:
        return os.environ.get("EXTERNAL_IP", None)

    @property
    def subtensor_network(self) -> str | None:
        return os.environ.get("SUBTENSOR_NETWORK", None)

    def miners(self) -> List[int]:
        uids = []
        print('all uids:', self.metagraph.uids)
        for uid in self.metagraph.uids:
            print('uid:', uid, 'is_serving:', self.metagraph.axons[uid].is_serving, 'permit:', self.metagraph.validator_permit[uid], 'stake:', self.metagraph.S[uid])
            if not self.metagraph.axons[uid].is_serving:
              continue
            if self.metagraph.validator_permit[uid] and self.metagraph.S[uid] > 1024:
                continue
            uids.append(int(uid))
        return uids
    

    def inspect(self):
        uids = self.metagraph.uids
        logger.info(f"Inspecting METAGRAPH UIDs: {uids}")

