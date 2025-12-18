import sys
from pathlib import Path

import asyncio
import os
import traceback
from uuid import uuid4
from langchain_openai import ChatOpenAI
from loguru import logger
from common import utils
from common.project_manager import ProjectManager
from common.settings import Settings


# https://thegraph.com/explorer/subgraphs/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1?view=Query&chain=arbitrum-one
cid = 'QmeB7YfNvLbM9AnSVeh5JvsfUwm1KVCtUDwaDLh5oxupGh'

# subgraph id
endpoint = 'https://gateway.thegraph.com/api/subgraphs/id/HMuAwufqZ1YCRmzL2SfHTVkzZovC9VL2UAKhjvRqKiR1'


# python -m scripts.pull_project_from_cid
if __name__ == "__main__":
    settings = Settings()
    settings.load_env_file("validator")

    synthetic_model_name = os.getenv("LLM_MODEL")
    logger.info(f"Using LLM_MODEL: {synthetic_model_name}")
    if not synthetic_model_name:
        logger.error("LLM_MODEL not set in environment variables. Exiting.")
        exit(1)
    
    llm_synthetic = ChatOpenAI(
            model=synthetic_model_name,
            temperature=1
    )

    if not settings.env_file:
        logger.error("Failed to load .env.validator file. Exiting.")
        exit(1)

    project_manager = ProjectManager(
        llm=llm_synthetic,
        target_dir=Path(__file__).parent.parent / "projects" / "validator",
    )

    combined = f"{cid}{endpoint}"
    hash_value = utils.hash256(combined)[:8]
    cid_hash = f"{cid}_{hash_value}"

    asyncio.run(
        project_manager.register_project(
            cid_hash=cid_hash,
            endpoint=endpoint,
        )
    )


    