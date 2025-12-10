"""Base GraphQL Toolkit implementation."""

from dataclasses import dataclass
import os
import logging
from typing import List, Optional, Dict, Any

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import ConfigDict

from common.prompt_template import get_block_rule_prompt

from .tools import (
    GraphQLQueryValidatorAndExecutedTool,
    GraphQLSchemaInfoTool,
    GraphQLTypeDetailTool,
)

from .node_types import GraphqlProvider, detect_node_type
from .tools import create_system_prompt


# Create logger
logger = logging.getLogger(__name__)

# Set log level from environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if hasattr(logging, log_level):
    logger.setLevel(getattr(logging, log_level))
    logging.getLogger().setLevel(getattr(logging, log_level))

class GraphQLSource:
    """
    GraphQL database connection wrapper.
    Similar to langchain's SQLDatabase but for GraphQL endpoints.
    """
    
    def __init__(
        self,
        endpoint: str,
        entity_schema: str,
        headers: Optional[Dict[str, str]] = None,
        schema_cache_ttl: int = 3600,
        node_type: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GraphQL database connection.
        
        Args:
            endpoint: GraphQL endpoint URL
            entity_schema: Raw schema content for entity definitions
            headers: Optional HTTP headers
            schema_cache_ttl: Schema cache time-to-live in seconds
            node_type: Optional explicit node type (SubQL, The Graph, etc.)
            manifest: Optional project manifest for node type detection
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.schema_cache_ttl = schema_cache_ttl
        self.entity_schema = entity_schema
        self.node_type = node_type
        self.manifest = manifest or {}
        self._schema_cache: Optional[Dict] = None
        self._schema_timestamp = 0
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get GraphQL schema with caching."""
        import time
        from .graphql import fetch_graphql_schema
        
        current_time = time.time()
        if (self._schema_cache is None or 
            current_time - self._schema_timestamp > self.schema_cache_ttl):
            
            headers = {**self.headers}
            if self.node_type == GraphqlProvider.THE_GRAPH:
                thegraph_token = os.getenv("THEGRAPH_API_TOKEN")
                if thegraph_token and "Authorization" not in headers:
                    headers["Authorization"] = f"Bearer {thegraph_token}"
                    logger.info("Added THEGRAPH_API_TOKEN to Authorization header to fetch schema")
            introspection_result = await fetch_graphql_schema(
                self.endpoint, 
                include_arg_descriptions=True,
                headers=headers
            )
            self._schema_cache = introspection_result
            self._schema_timestamp = current_time
        
        return self._schema_cache
    
    async def get_schema_data(self) -> Dict[str, Any]:
        """Get just the __schema part for compatibility with existing code."""
        introspection_result = await self.get_schema()
        return introspection_result.get("data", {}).get("__schema", {})
    
    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        import aiohttp
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        # Prepare headers
        headers = {**self.headers, "Content-Type": "application/json"}
        
        # For The Graph projects, add API token from environment if available
        if self.node_type == GraphqlProvider.THE_GRAPH:
            thegraph_token = os.getenv("THEGRAPH_API_TOKEN")
            if thegraph_token and "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {thegraph_token}"
                logger.info("Added THEGRAPH_API_TOKEN to Authorization header to execute query")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"GraphQL query failed: {response.status}")
    
    def get_endpoint(self) -> str:
        """Get the GraphQL endpoint URL."""
        return self.endpoint


class GraphQLToolkit(BaseToolkit):
    """
    GraphQL Agent Toolkit.
    
    Provides tools for LLM agents to interact with GraphQL APIs,
    similar to LangChain's SQLDatabaseToolkit.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    graphql_source: GraphQLSource
    llm: Optional[BaseLanguageModel] = None
    
    def __init__(
        self,
        graphql_source: GraphQLSource,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs
    ):
        """
        Initialize the GraphQL toolkit.
        
        Args:
            graphql_source: GraphQL source connection with schema access
            llm: Optional language model for advanced tools
        """
        super().__init__(graphql_source=graphql_source, llm=llm, **kwargs)
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all available GraphQL tools.
        
        Returns:
            List of GraphQL tools
        """
        tools = [
            GraphQLSchemaInfoTool(
                graphql_source=self.graphql_source,
                node_type=self.graphql_source.node_type
            ),
            GraphQLTypeDetailTool(graphql_source=self.graphql_source),
            GraphQLQueryValidatorAndExecutedTool(graphql_source=self.graphql_source),
            # GraphQLQueryValidatorTool(graphql_source=self.graphql_source),
            # GraphQLExecuteTool(graphql_source=self.graphql_source)
        ]
        
        return tools
    
    @property
    def dialect(self) -> str:
        """Get the dialect name."""
        return "graphql"
@dataclass
class ProjectConfig:
    """Configuration for a SubQuery or The Graph project."""
    cid: str
    endpoint: str
    schema_content: str
    cid_hash: Optional[str] = None
    node_type: str = GraphqlProvider.UNKNOWN
    manifest: Dict[str, Any] = None
    domain_name: str = "GraphQL Project"
    domain_capabilities: List[str] = None
    decline_message: str = "I'm specialized in this project's data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about this project's data instead."
    suggested_questions: List[str] = None
    authorization: Optional[str] = None

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.domain_capabilities is None:
            self.domain_capabilities = [
                "Blockchain data indexed by this project",
                "Entity relationships and queries",
                "Project-specific metrics and analytics"
            ]
        if self.suggested_questions is None:
            self.suggested_questions = [
                "What types of data can I query from this project?",
                "Show me a sample GraphQL query",
                "What entities are available in this schema?",
                "How can I filter the data?"
            ]


class GraphQLAgent:
    """GraphQL agent for a specific SubQuery project."""

    def __init__(self, config: ProjectConfig):
        """Initialize the agent with project configuration."""
        self.config = config

        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        logger.info(f"Initializing GraphQLAgent with model: {model_name}")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )

        # Create tools with node type information and authorization header
        headers = {}
        if config.authorization:
            headers["Authorization"] = config.authorization

        toolkit = create_graphql_toolkit(
            config.endpoint,
            config.schema_content,
            headers=headers if headers else None,
            node_type=config.node_type,
            manifest=config.manifest
        )
        self.tools = toolkit.get_tools()

        # Setup agent
        self._setup_agent()

    def _setup_agent(self):
        # Create system prompt for langgraph
        prompt = create_system_prompt(
            domain_name=self.config.domain_name,
            domain_capabilities=self.config.domain_capabilities,
            decline_message=self.config.decline_message
        )

        # Create agent with system message
        self.executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt
        )

    async def query_no_stream(self, question: str, prompt_cache_key: str = '', is_synthetic: bool = False, block_height: int = 0):
        """Execute a non-streaming query.

        Args:
            question: The query question
            is_synthetic: Whether this is a synthetic challenge (affects domain filtering behavior)
            block_height: The block height for time-travel queries
        """
        block_rule = get_block_rule_prompt(block_height, self.config.node_type)

        # Create appropriate system prompt based on query type
        prompt = create_system_prompt(
            domain_name=self.config.domain_name,
            domain_capabilities=self.config.domain_capabilities,
            decline_message=self.config.decline_message,
            is_synthetic=is_synthetic,
        )

        # Create a temporary agent with the appropriate prompt
        temp_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        response = await temp_executor.ainvoke(
            {
                "messages": [
                    {"role": "system", "content": block_rule},
                    {"role": "user", "content": question}
                ],
            },
            config={
                "configurable": {
                    "recursion_limit": 12,
                    "block_height": block_height,
                }
            },
            prompt_cache_key=prompt_cache_key
        )
        return response, prompt, block_rule

    async def query(self, messages: list, include_think: bool = False):
        """Streaming query using langgraph agent with conversation history support."""
        logger.info(f"GraphQLAgent.query called with include_think={include_think}")
        think_started = False
        chunk_size = 60

        # Convert message format if needed
        if isinstance(messages, str):
            # Backward compatibility: if a string is passed, treat as single user message
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages:
            # Convert ChatCompletionMessage objects to dict format if needed
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # Pydantic model - convert to dict
                    formatted_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    # Already in correct format
                    formatted_messages.append(msg)
                else:
                    # Fallback
                    formatted_messages.append({"role": "user", "content": str(msg)})
            messages = formatted_messages

        # Get last user message for logging
        last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

        try:
            logger.info(f"Processing query for {self.config.cid} with {len(messages)} messages: {last_user_msg[:100]}...")
            async for event in self.executor.astream(
                {"messages": messages},
                config={"recursion_limit": 12}
            ):
                logger.debug(f"Event keys: {list(event.keys())}")
                logger.debug(f"Event: {event}")

                # Handle langgraph events - they contain node names as keys
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        # Agent node - contains tool calls or final message
                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            for message in messages:
                                # Handle tool calls in agent messages
                                if hasattr(message, 'tool_calls') and message.tool_calls and include_think:
                                    if not think_started:
                                        yield "<think>\n"
                                        think_started = True
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get('name', 'unknown')
                                        yield f"[Tool: {tool_name}]\n"

                                # Handle regular message content
                                elif hasattr(message, 'content') and message.content:
                                    if think_started:
                                        yield "</think>\n"
                                        think_started = False

                                    content = str(message.content).strip()
                                    idx = 0
                                    while idx < len(content):
                                        chunk = content[idx:idx+chunk_size]
                                        yield chunk
                                        idx += chunk_size

                    elif node_name == "tools" and include_think:
                        # Tools node - contains tool execution results
                        if not think_started:
                            yield "<think>\n"
                            think_started = True

                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            for message in messages:
                                if hasattr(message, 'content') and message.content:
                                    yield "\n[Tool Output]:\n"

                                    observation = str(message.content)
                                    # Truncate schema info output
                                    if 'graphql_schema_info' in str(message.name if hasattr(message, 'name') else ''):
                                        max_length = 2000
                                        if len(observation) > max_length:
                                            observation = observation[:max_length] + f"\n\n... [Output truncated after {max_length} characters to save tokens.]"

                                    idx = 0
                                    while idx < len(observation):
                                        chunk = observation[idx:idx+chunk_size]
                                        yield chunk
                                        idx += chunk_size
                                    yield "\n\n"

            # Close any remaining think block
            if think_started:
                yield "</think>\n"

        except Exception as e:
            logger.error(f"Query failed for {self.config.cid}: {str(e)}")
            yield f"I encountered an issue processing your query. Error: {str(e)}"
            return


# Factory function for creating GraphQL toolkit
def create_graphql_toolkit(
    endpoint: str,
    entity_schema: str,
    headers: Optional[Dict[str, str]] = None,
    llm: Optional[BaseLanguageModel] = None,
    node_type: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> GraphQLToolkit:
    """
    Create a GraphQL toolkit instance with automatic node type detection.
    
    Args:
        endpoint: GraphQL endpoint URL
        entity_schema: Raw schema content for entity definitions
        headers: Optional HTTP headers for authentication
        llm: Optional language model
        node_type: Optional explicit node type (if known)
        manifest: Optional project manifest for node type detection
        
    Returns:
        GraphQL toolkit instance
    """
    graphql_source = GraphQLSource(
        endpoint=endpoint, 
        entity_schema=entity_schema, 
        headers=headers,
        node_type=node_type,
        manifest=manifest
    )
    return GraphQLToolkit(graphql_source=graphql_source, llm=llm)