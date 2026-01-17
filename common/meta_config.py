import os
from typing import (
    Any,
)


import aiohttp
from loguru import logger

from common.protocol import MetaConfigResponse

class MetaConfig:
    def __init__(self, meta_config: dict[str, Any] = None):
        if meta_config is None:
            self.meta_config = {}
        else:
            self.meta_config = meta_config

    async def pull(self):
        """pull projects from board service."""
        headers = {
            "accept": "application/json",
        }
        board_url = os.environ.get('BOARD_SERVICE')
        if not board_url:
            return
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{board_url}/config/meta", headers=headers) as resp:
                response_data = await resp.json()
                logger.debug(f"Meta config response: {response_data}")
        
        parsed = MetaConfigResponse(**response_data)
        return parsed
