import os
import sys

sys.path.append(os.path.abspath("/Users/demon/Desktop/work/onf/subql-graphql-agent/examples"))
from server import GraphQLAgent as ServerGraphQLAgent, ProjectConfig, stream_chat_completion, non_stream_chat_completion
from working_example import GraphQLAgent as ExampleGraphQLAgent

stream_chat_completion = stream_chat_completion
non_stream_chat_completion = non_stream_chat_completion

def initServerAgent() -> ServerGraphQLAgent:
    schema_file_path = "/Users/demon/Desktop/work/onf/subql-graphql-agent/examples/schema.graphql"
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        entity_schema = f.read()
    project_config = ProjectConfig(
        cid='xx',
        endpoint="https://index-api.onfinality.io/sq/subquery/subquery-mainnet",
        schema_content=entity_schema
    )
    agent = ServerGraphQLAgent(project_config)
    return agent

def initServerAgentWithConfig(project_config: ProjectConfig) -> ServerGraphQLAgent:
    agent = ServerGraphQLAgent(project_config)
    return agent

# def initExampleAgent() -> ExampleGraphQLAgent:
#     agent = ExampleGraphQLAgent("https://index-api.onfinality.io/sq/subquery/subquery-mainnet")
#     return agent