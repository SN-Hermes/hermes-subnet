
import os
import time
from typing import List, Tuple
from bittensor.core.metagraph import Metagraph
from bittensor.core.subtensor import Subtensor
from loguru import logger
import dotenv
import bittensor as bt
from common import utils
from common.enums import RoleFlag


class Settings:
    _subtensor: Subtensor | None = None
    _wallet: bt.wallet | None = None
    _last_metagraph: Metagraph = None
    _last_update_time: int = 0
    _env_file: str | None = None
    _external_ip: str | None = None

    def load_env_file(self, role: str | None = None):
        env_file = f".env.{role}" if role else ".env"
        try:
            dotenv.load_dotenv(env_file)
            self._env_file = env_file
            logger.info(f"Loaded {env_file} file")
        except Exception as e:
            logger.error(f"Failed to load {env_file} file: {e}")

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
        if int(time.time()) - self._last_update_time > 60 * 10:
            try:
                logger.info(f"Syncing METAGRAPH for NETUID={self.netuid}")
                if self._last_metagraph is None:
                    # Create metagraph ONCE on first access
                    self._last_metagraph = Metagraph(netuid=self.netuid, network=self.subtensor.network)
                # Use sync() to update existing metagraph - avoids the memory leak!
                self._last_metagraph.sync(subtensor=self.subtensor, lite=True)
                self._last_update_time = int(time.time())
                return self._last_metagraph
            except Exception as e:
                logger.error(f"Failed to sync METAGRAPH for NETUID={self.netuid}: {e}")
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
        wallet_path = os.environ.get("WALLET_PATH")
        
        if wallet_path:
            logger.info(f"Instantiating wallet with name: {wallet_name}, hotkey: {hotkey}, path: {wallet_path}")
            self._wallet = bt.wallet(name=wallet_name, hotkey=hotkey, path=wallet_path)
        else:
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
        if self._external_ip is None:
            self._external_ip = os.environ.get("EXTERNAL_IP", None) or utils.try_get_external_ip()
        return self._external_ip

    @property
    def subtensor_network(self) -> str | None:
        return os.environ.get("SUBTENSOR_NETWORK", None)

    @property
    def base_dir(self) -> str:
        return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    @property
    def env_file(self) -> str | None:
        return self._env_file

    def miners(self) -> Tuple[List[int], List[str]]:
        uids = []
        meta: bt.Metagraph = self.metagraph
        logger.debug(f"METAGRAPH UIDs: {meta.uids}")

        # miners = ( placeholder1 is miner or (placeholder is not validator && uid != 0 ) )
        for uid in meta.uids:
            a = meta.axons[uid]
            if a.placeholder1 == RoleFlag.MINER.value:
                uids.append(int(uid))
            elif a.placeholder1 != RoleFlag.VALIDATOR.value and int(uid) != 0:
                 uids.append(int(uid))
            logger.debug(f"UID: {uid}, is_serving: {meta.axons[uid].is_serving}, permit: {meta.validator_permit[uid]}, stake: {meta.S[uid]}")
            # if not meta.axons[uid].is_serving:
            #   continue
            # if meta.validator_permit[uid] and meta.S[uid] > 1024:
            #     continue
        hotkeys = [meta.hotkeys[u] for u in uids]
        return uids, hotkeys

    def reread(self):
        if self._env_file and os.path.exists(self._env_file):
            try:
                with open(self._env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('THEGRAPH_API_TOKEN='):
                            key, value = line.split('=', 1)
                            old_value = os.environ.get(key)
                            if old_value != value:
                                os.environ[key] = value
                                logger.info(f"Reloaded THEGRAPH_API_TOKEN from {self._env_file}")
                            break
            except Exception as e:
                logger.error(f"Failed to reread THEGRAPH_API_TOKEN from {self._env_file}: {e}")

    def inspect(self):
        uids = self.metagraph.uids
        logger.info(f"Inspecting METAGRAPH UIDs: {uids}")

settings = Settings()