from abc import ABC, abstractmethod
import logging
import os
import sys
from typing import Any

from loguru import logger
from bittensor.core.extrinsics.serving import serve_extrinsic
from common.settings import Settings
from common.utils import try_get_external_ip
import agent.graphql_agent as subAgent


class BaseNeuron(ABC):
    serverAgent: Any
    exampleAgent: Any
    settings: Settings
    should_exit: bool
    uid: int
    
    @property
    @abstractmethod
    def role(self) -> str:
        '''
        Returns the role of the neuron.
        '''

    def __init__(self):
        Settings.load_env_file(self.role)
        self.settings = Settings()
        self.should_exit = False
        self.uid = self.settings.metagraph.hotkeys.index(
            self.settings.wallet.hotkey.ss58_address
        )
        self.serverAgent = subAgent.initServerAgent()
        self.exampleAgent = subAgent.initExampleAgent()

    def start(self):
        external_ip = self.settings.external_ip or try_get_external_ip()
        serve_success = serve_extrinsic(
          subtensor=self.settings.subtensor,
          wallet=self.settings.wallet,
          ip=external_ip,
          port=self.settings.port,
          protocol=4,
          netuid=self.settings.netuid,
        )

        msg = f"Serving {self.role} endpoint {external_ip}:{self.settings.port} on network: {self.settings.subtensor.network} with netuid: {self.settings.netuid} uid:{self.uid} {serve_success}"
        if not serve_success:
            logger.error(msg)
            sys.exit(1)

        logger.info(msg)
