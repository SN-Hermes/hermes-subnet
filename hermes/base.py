from abc import ABC, abstractmethod
import sys
from loguru import logger
from bittensor.core.extrinsics.serving import serve_extrinsic

from common.enums import RoleFlag
from common.settings import Settings, settings
from common.utils import try_get_external_ip


class BaseNeuron(ABC):
    settings: Settings
    uid: int
    
    @property
    @abstractmethod
    def role(self) -> str:
        '''
        Returns the role of the neuron.
        '''


    def __init__(self):
        self.settings = settings

        self.uid = self.settings.metagraph.hotkeys.index(
            self.settings.wallet.hotkey.ss58_address
        )

    def start(self, flag: RoleFlag):
        self.check_registered()

        external_ip = self.settings.external_ip or try_get_external_ip()
        serve_success = serve_extrinsic(
          subtensor=self.settings.subtensor,
          wallet=self.settings.wallet,
          ip=external_ip,
          port=self.settings.port,
          protocol=4,
          netuid=self.settings.netuid,
          placeholder1=flag.value,
        )

        msg = f"Serving {self.role} endpoint {external_ip}:{self.settings.port} on network: {self.settings.subtensor.network} with netuid: {self.settings.netuid} uid:{self.uid} {serve_success}"
        if not serve_success:
            logger.error(msg)
            sys.exit(1)

        logger.info(msg)

    def check_registered(self):
        if not self.settings.subtensor.is_hotkey_registered(
            netuid=self.settings.netuid,
            hotkey_ss58=self.settings.wallet.hotkey.ss58_address,
        ):
            logger.error(
                f"Wallet: {self.settings.wallet} is not registered on netuid {self.settings.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit(1)