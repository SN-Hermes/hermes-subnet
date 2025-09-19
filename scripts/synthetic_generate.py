import asyncio
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

    manifest = asyncio.run(projectManager.pull_manifest(SUBQL_CID))
    entity_schema = asyncio.run(projectManager.pull_schema(manifest))

    llm = ChatOpenAI(
        model=model_name,
        temperature=1
    )

    asyncio.run(projectManager.register_project(SUBQL_CID, 'https://index-api.onfinality.io/sq/subquery/subquery-mainnet'))
    server_agent = subAgent.initServerAgentWithConfig(projectManager.get_project(SUBQL_CID))


    count = 10
    question_generator = QuestionGenerator(max_history=count)

    logger.info(f"model_name: {model_name}")
    logger.info(f"entity_schema: ({len(entity_schema)} chars)")

    for i in range(10):
        question = asyncio.run(question_generator.generate_question(SUBQL_CID, entity_schema, llm))
        logger.info(f"Generated question {i+1}/{count}: {question}")

        # response = asyncio.run(server_agent.query_no_stream(question))
        # ground_truth = response.get('messages', [])[-1].content
        # logger.info(f"   ground_truth: {ground_truth}")

        logger.info(f"\n")



