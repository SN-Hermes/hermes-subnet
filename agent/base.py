
import os

from langchain_openai import ChatOpenAI
from loguru import logger
from langgraph.prebuilt import create_react_agent

from agent.subquery_graphql_agent.base import GraphQLSource
from agent.subquery_graphql_agent.project import LocalProjectBase, LocalProjectCodex, LocalProjectCovalent, LocalProjectTheGraph, LocalProjectSubquery
from agent.subquery_graphql_agent.tools import GraphQLQueryValidatorAndExecutedTool, GraphQLSchemaInfoTool, GraphQLTypeDetailTool



class BaseAgent:
    project: LocalProjectSubquery | LocalProjectTheGraph | LocalProjectCodex | LocalProjectCovalent
    recursion_limit: int = 12

    def __init__(self, project: LocalProjectSubquery | LocalProjectTheGraph | LocalProjectCodex | LocalProjectCovalent):

        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
        logger.info(f"Initializing GraphQLAgent with model: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=300,
            max_retries=3,
            # extra_body={"thinking": {"type": "disabled"}},
        )
        self.project = project

    def tools(self):
        graphql_source = GraphQLSource(
            endpoint=self.project.endpoint, 
            entity_schema=self.project.schema_content, 
            full_schema=self.project.full_schema_content if isinstance(self.project, LocalProjectCodex) else None,
            headers=None,
            node_type=self.project.node_type,
            manifest=self.project.manifest
        )

        tools = [
            GraphQLSchemaInfoTool(
                graphql_source=graphql_source,
                node_type=graphql_source.node_type
            ),
            GraphQLTypeDetailTool(graphql_source=graphql_source, node_type=graphql_source.node_type),
            GraphQLQueryValidatorAndExecutedTool(graphql_source=graphql_source, node_type=graphql_source.node_type),
        ]
        return tools

    async def query(
        self,
        question: str,
        prompt_cache_key: str = '',
        block_height: int = 0
    ) -> dict[str, any]:
        """Execute a non-streaming query.

        Args:
            question: The query question
            block_height: The block height for time-travel queries
        """

        # Create appropriate system prompt based on query type
        prompt = self.project.create_system_prompt()

        # Create a temporary agent with the appropriate prompt
        temp_executor = create_react_agent(
            model=self.llm,
            tools=self.tools(),
            prompt=prompt,
        )

        messages = self.project.create_middle_instructions_messages(block_height=block_height)

        messages.extend([
            {"role": "user", "content": question}
        ])

        response = await temp_executor.ainvoke(
            {
                "messages": messages
            },
            config={
                "configurable": {
                    "recursion_limit": self.recursion_limit,
                    "block_height": block_height,
                }
            },
            prompt_cache_key=prompt_cache_key
        )
        return response

    async def ainvoke(self, input: dict):
        temp_executor = create_react_agent(
            model=self.llm,
            tools=self.tools(),
            prompt=None,
        )
        response = await temp_executor.ainvoke(input)
        return response
