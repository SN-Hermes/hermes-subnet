import asyncio
import json
import os

import dotenv
from langchain_openai import ChatOpenAI
from loguru import logger
from common.project_manager import ProjectManager
from hermes.validator.question_generator import QuestionGenerator
import agent.graphql_agent as subAgent
dotenv.load_dotenv('.env')


SUBQL_CID = 'QmfUNJC1Qz8m3F67sQmxrwjuSAu4WaCR1iBdPPdzBruQ7P'
model_name = os.getenv("LLM_MODEL", "gpt-5")


# python -m scripts.synthetic_generate
if __name__ == "__main__":
    projectManager = ProjectManager('./projects')

    # asyncio.run(projectManager.pull())

    asyncio.run(projectManager.register_project(SUBQL_CID, 'https://index-api.onfinality.io/sq/subquery/subquery-mainnet'))
    server_agent = subAgent.initServerAgentWithConfig(projectManager.get_project(SUBQL_CID))

    count = 10
    question_generator = QuestionGenerator(max_history=count)

    for i in range(10):
        question = asyncio.run(question_generator.generate_question_with_agent(SUBQL_CID, projectManager.get_project(SUBQL_CID).schema_content, server_agent))
        logger.info(f"Generated question {i+1}/{count}: {question}")

        response = asyncio.run(server_agent.query_no_stream(question))
        ground_truth = response.get('messages', [])[-1].content
        logger.info(f"   ground_truth: {ground_truth}")
        
        logger.info(f"\n")
