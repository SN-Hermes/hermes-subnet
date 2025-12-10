import importlib
import json
from pathlib import Path
import time
from typing import Literal
import pkgutil
import sys
import os
from loguru import logger
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import json
from typing import (
    Any,
    Literal,
    Union,
)
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
)

from langgraph.graph import StateGraph, START, END
from agent.stats import ToolCountHandler
from agent.subquery_graphql_agent.base import GraphQLAgent
from common.project_manager import ProjectConfig, ProjectManager
from common.prompt_template import get_block_rule_prompt
from common.protocol import ExtendedMessagesState
import common.utils as utils


class AgentManager:
    project_manager: ProjectManager
    graphql_agent: dict[str, GraphQLAgent]
    '''
    { 
        [cid]: {
            tools: {},
            miner_agent: agent,
            server_agent: agent,
            agent_graph: graph,
            counter: counter,
        }
    }
    '''
    miner_agent: dict[str, any]
    save_project_dir: str
    llm_synthetic: ChatOpenAI

    def __init__(self, save_project_dir: str, llm_synthetic: ChatOpenAI, ipc_common_config: dict = None):
        self.save_project_dir = save_project_dir
        self.graphql_agent = {}
        self.miner_agent = {}
        self.llm_synthetic = llm_synthetic
        self.project_manager = ProjectManager(self.llm_synthetic, self.save_project_dir)
        self.ipc_common_config = ipc_common_config

    async def start(self, pull=True, role: Literal["", "validator", "miner"] = "", silent: bool = False):
        if pull:
            await self.project_manager.pull(silent=silent)
        else:
            self.project_manager.load()

        if self.ipc_common_config is not None:
            projects_config = self.project_manager.get_projects()
            for cid_hash, config in projects_config.items():
                self.ipc_common_config.update({
                    cid_hash: {
                        "node_type": config.node_type,
                        "endpoint": config.endpoint,
                    }
                })

        if role == "miner":
            self._init_miner_agents()
        elif role == "validator":
            self._init_agents()

    def _init_agents(self):
        new_agents = []
        for cid_hash, project_config in self.get_projects().items():
            if cid_hash not in self.graphql_agent:
                new_agents.append(cid_hash)
                self.graphql_agent[cid_hash] =  GraphQLAgent(project_config)
        if new_agents:
            logger.info(f"[AgentManager] Initialized graphql_agents for projects: {new_agents}")

    def _init_miner_agents(self):
        enable_fallback = os.getenv("ENABLE_FALL_BACK_GRAPHQL_AGENT", "false").lower() == "true"
        logger.info(f"[AgentManager] ENABLE_FALL_BACK_GRAPHQL_AGENT: {enable_fallback}")

        base_path = Path(self.save_project_dir)
        for project_dir in base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            cid_hash = project_dir.name
            if cid_hash == "__pycache__":
                continue

            if cid_hash in self.miner_agent:
                continue

            project = self.miner_agent.get(cid_hash)
            prev_tools = self.miner_agent.get(cid_hash, {}).get('tools', {})
            current_tools = {}
            
            relative_module_parts = project_dir.relative_to(Path(__file__).parent.parent / "projects").parts
            package_prefix = ".".join(["projects"] + list(relative_module_parts))

            for module_info in pkgutil.iter_modules([str(project_dir)]):
                module_name = module_info.name
                # full_module = f"projects.{project_name}.{module_name}"
                full_module = f"{package_prefix}.{module_name}"

                if full_module in sys.modules:
                    mod = importlib.reload(sys.modules[full_module])
                else:
                    mod = importlib.import_module(full_module)

                # module_tools = {t.name: t for t in getattr(mod, "tools", []) if isinstance(t, BaseTool)}
                module_tools = {utils.get_func_name(t): t for t in getattr(mod, "tools", [])}
                current_tools.update(module_tools)

            miner_tools = []
            created, updated, deleted = [], [], []
            for name, _tool in current_tools.items():
                version = getattr(type(_tool), "__version__", "0.0.0")
                prev_version = prev_tools.get(name)
                if not prev_version:
                    created.append(_tool)
                elif prev_version != version:
                    updated.append(_tool)
                miner_tools.append(_tool)

                deleted = [name for name in prev_tools.keys() if name not in current_tools]
            
            if (not project) or (created or updated or deleted):
                config_path = project_dir / 'config.json'

                suc = False
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                            suc = True
                    except Exception as e:
                        logger.error(f"Failed to read {config_path}: {e}")
                        config = {}
                else:
                    logger.warning(f"Config file not found: {config_path}")
                    config = {}

                if not suc:
                    continue

                graphql_agent = GraphQLAgent(ProjectConfig(**config))

                def graphql_agent_tool():
                    """
                    This tool is automatically invoked when other specialized tools are unable to answer the user's query, ensuring that the langgraph can still provide a meaningful response.
                    If you decide to use this tool, just return an empty AIMessage Content.
                    """
                    pass

                def make_call_graphql_agent(agent: GraphQLAgent):
                    async def call_graphql_agent(state: ExtendedMessagesState) -> dict:
                        messages = state["messages"]
                        human_messages = [m for m in messages if m.type == 'human']

                        enable_log = os.getenv("ENABLE_LOG_GRAPHQL_AGENT", "false").lower() == "true"
                        if enable_log:
                            logger.info(f" call_graphql_agent - human_messages: {human_messages} ")
                        if not human_messages:
                            return {"messages": [AIMessage(content="")]}

                        block_height = state.get("block_height", 0) if isinstance(state, dict) else getattr(state, "block_height", 0)

                        msgs = [
                            SystemMessage(content=get_block_rule_prompt(block_height, agent.config.node_type))
                        ] + human_messages

                        input_token_usage, input_cache_read_token_usage, output_token_usage = 0, 0, 0
                        tool_calls = []
                        error = None
                        try:
                            response = await agent.executor.ainvoke(
                                {"messages": msgs},
                                config={
                                        "recursion_limit": 12,
                                        "configurable": {
                                        "block_height": block_height,
                                    }
                                },
                                prompt_cache_key=f"{agent.config.cid_hash}_{time.perf_counter()}"
                            )
                            if enable_log:
                                logger.info(f" --------call_graphql_agent------ response: {response} ")
                        
                            last = response['messages'][-1]
                            input_token_usage, input_cache_read_token_usage, output_token_usage = utils.extract_token_usage(response['messages'][0: -1])

                            tool_calls = utils.extract_tool_calls(response['messages'])

                            if not last.content:
                                error_msg = utils.try_get_invalid_tool_messages(last)
                                if error_msg:
                                    last.content = error_msg

                        except Exception as e:
                            error = str(e)
                            logger.error(f" call_graphql_agent - error: {e} ")
                            last = AIMessage(content=f"Error invoking GraphQL Agent: {e}")

                        return {
                                "messages": [last], 
                                "intermediate_graphql_agent_input_token_usage": input_token_usage,
                                "intermediate_graphql_agent_input_cache_read_token_usage": input_cache_read_token_usage,
                                "intermediate_graphql_agent_output_token_usage": output_token_usage,
                                "graphql_agent_hit": True,
                                "tool_calls": tool_calls,
                                "error": error
                            }
                    return call_graphql_agent

                def tool_condition(
                    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
                    messages_key: str = "messages",
                ):
                    if getattr(state, "error", None) is not None:
                        return "final"
            
                    if isinstance(state, list):
                        ai_message = state[-1]
                    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
                        ai_message = messages[-1]
                    elif messages := getattr(state, messages_key, []):
                        ai_message = messages[-1]
                    else:
                        raise ValueError(f"No messages found in input state to tool_edge: {state}")

                    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                        # Check if any tool call is graphql_agent_tool
                        for tool_call in ai_message.tool_calls:
                            if tool_call['name'] == "graphql_agent_tool":
                                return "call_graphql_agent"
                        return "tools"
                    return "final"

                # logger.info(f"[AgentManager] Project {cid_hash} - Detected changes in tools. Created: {[utils.get_func_name(t) for t in created]}, Updated: {[utils.get_func_name(t) for t in updated]}, Deleted: {deleted}")
                
                llm_with_tools = self.llm_synthetic.bind_tools(miner_tools + [graphql_agent_tool] if enable_fallback else [])

                def make_call_model(llm: ChatOpenAI):
                    async def call_model_func(state: ExtendedMessagesState) -> int:
                        messages = state["messages"]
                        error = None
                        response_messages = None
                        # logger.info(f" call_model - messages: {messages} ")
                        try:
                            response_messages = await llm.ainvoke(messages)
                            # logger.info(f" call_model - response: {response_messages} ")
                        except Exception as e:
                            logger.error(f" call_model - error: {e} ")
                            # response_messages = AIMessage(content=f"Error invoking LLM: {e}")
                            error = str(e)
                        return {"messages": [response_messages] if error is None else messages, "error": error}
                    return call_model_func

                def make_final_node():
                    async def final_func(state: ExtendedMessagesState) -> int:
                        # messages = state["messages"]
                        # logger.info(f" final - state: {state} ")
                        return state
                    return final_func


                builder = StateGraph(ExtendedMessagesState)
                builder.add_node("call_model", make_call_model(llm_with_tools))
                builder.add_node("tools", ToolNode(miner_tools))
                builder.add_node("call_graphql_agent", make_call_graphql_agent(graphql_agent))
                builder.add_node("final", make_final_node())
                builder.add_edge("call_graphql_agent", "final")
                builder.add_conditional_edges(
                    "call_model",
                    tool_condition,
                )
                builder.add_edge("final", END)

                builder.add_edge("tools", "call_model")
                builder.add_edge(START, "call_model")
                graph = builder.compile()

                logger.info(f"[AgentManager] load agent, Project {cid_hash} using model {self.llm_synthetic.model_name} - tools: {[utils.get_func_name(t) for t in miner_tools]}, Created: {[utils.get_func_name(t) for t in created]}, Updated: {[utils.get_func_name(t) for t in updated]}, Deleted: {deleted}, with prompt: {suc}")

                self.miner_agent[cid_hash] = {
                    "tools": [],
                    "graphql_agent": graphql_agent,
                    "agent_graph": graph,
                    "counter": ToolCountHandler()
                }
            else:
                logger.info(f"[AgentManager] Project {cid_hash} - No changes in tools.")

        return self.miner_agent

    def get_projects(self):
        return self.project_manager.get_projects()

    def get_graphql_agent(self, cid_hash: str) -> GraphQLAgent:
        return self.graphql_agent[cid_hash]

    def get_miner_agent(self, cid_hash: str | None = None) -> dict | tuple[StateGraph, GraphQLAgent]:
        if cid_hash:
            config = self.miner_agent.get(cid_hash, {})
            return (
                config.get('agent_graph', None),
                config.get('graphql_agent', None)
            )
        return self.miner_agent