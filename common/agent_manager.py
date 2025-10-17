import importlib
import json
from pathlib import Path
from typing import Literal
import pkgutil
import sys
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.schema import AIMessage
from loguru import logger
from langgraph.graph import StateGraph, MessagesState, START, END
from agent.stats import ToolCountHandler
from agent.subquery_graphql_agent.base import GraphQLAgent
from common.project_manager import ProjectConfig, ProjectManager
from common.utils import create_system_prompt
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

    def __init__(self, save_project_dir: str, llm_synthetic: ChatOpenAI):
        self.save_project_dir = save_project_dir
        self.graphql_agent = {}
        self.miner_agent = {}
        self.llm_synthetic = llm_synthetic
        self.project_manager = ProjectManager(self.llm_synthetic, self.save_project_dir)


    async def start(self, pull=True, role: Literal["", "validator", "miner"] = ""):
        if pull:
            await self.project_manager.pull()
        else:
            self.project_manager.load()

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
        def miner_router(state):
            messages = state["messages"]
            for m in messages:
                # TODO: check tool call has real output
                if m.type == 'ai' and len(m.tool_calls) > 0:
                    return "final_filter"
            # TODO: trim
            # first_message = messages[0:1]
            # state['messages'] = first_message
            return "graphql_agent"
        
        def last_message_filter(state: MessagesState):
            last = state['messages'][-1]
            if not last.content:
                error_msg = utils.try_get_invalid_tool_messages(last)
                if error_msg:
                    last.content = error_msg
            # logger.info(f"====================== Entering last_message_filter ====================== {state['messages']}  last:{last}\n")
            return {"messages": [last]}

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

                module_tools = {t.name: t for t in getattr(mod, "tools", []) if isinstance(t, BaseTool)}
                current_tools.update(module_tools)

            tools = []
            created, updated, deleted = [], [], []
            for name, tool in current_tools.items():
                version = getattr(type(tool), "__version__", "0.0.0")
                prev_version = prev_tools.get(name)
                if not prev_version:
                    created.append(tool)
                elif prev_version != version:
                    updated.append(tool)
                tools.append(tool)

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

                # prompt = create_system_prompt(
                #     domain_name=config.get("domain_name", ""),
                #     domain_capabilities=config.get("domain_capabilities", []),
                #     decline_message=config.get("decline_message", "")
                # ) if suc else f"You are the agent for project {cid}."

                # reconstruct agent
                # model = os.environ.get("MINER_LLM_MODEL", "gpt-4o-mini")

                graphql_agent = GraphQLAgent(ProjectConfig(**config))
                miner_agent = create_react_agent(
                    # model="openai:" + model,
                    model=self.llm_synthetic,
                    tools=tools,
                    prompt= f"You are the agent for project {cid_hash}."
                )
                logger.info(f"[AgentManager] load agent, Project {cid_hash} using model {self.llm_synthetic.model_name} - tools: {[t.name for t in tools]}, Created: {[t.name for t in created]}, Updated: {[t.name for t in updated]}, Deleted: {deleted}, with prompt: {suc}")

                async def fallback_graphql_agent(state: MessagesState):
                    # logger.info(f"====================== Entering fallback_graphql_agent ====================== {state["messages"]}")
                    # logger.info(f"Passing to fallback_graphql_agent")
                    messages = state["messages"][0:1]
                    user_input = messages[-1].content if len(messages) > 0 else ""
                    # logger.info(f"Passing to graphql_agent with messages: {messages}  000  {user_input}")
                    response = await graphql_agent.executor.ainvoke({"messages": [{"role": "user", "content": user_input}]})
                    # logger.info(f"============================================= sss {response}")
                    # messages = response['messages']
                    # for msg in messages:
                    #     logger.info(f"    ddddddddddddddddddddd: {msg.type} {msg}\n")

                    last = response['messages'][-1]
                    if not last.content:
                        error_msg = utils.try_get_invalid_tool_messages(last)
                        if error_msg:
                            last.content = error_msg

                    # logger.info(f"++++++++++++++++++++++++++++++++++++++++++++: {last}")
                    return {"messages": [last]}

                builder = StateGraph(MessagesState)
                builder.add_node("miner_agent", miner_agent)
                # builder.add_node("graphql_agent", graphql_agent.executor)
                builder.add_node("graphql_agent", fallback_graphql_agent)
                builder.add_node("final_filter", last_message_filter)

                builder.add_conditional_edges(
                    "miner_agent",
                    miner_router
                )

                builder.add_edge("graphql_agent", "final_filter")
                builder.add_edge("final_filter", END)

                builder.add_edge(START, "miner_agent")
                multi_agent_graph = builder.compile()

                self.miner_agent[cid_hash] = {
                    "tools": {t.name: getattr(type(t), "__version__", "0.0.0") for t in tools},
                    "miner_agent": miner_agent,
                    "graphql_agent": graphql_agent,
                    "agent_graph": multi_agent_graph,
                    "counter": ToolCountHandler()
                }
            else:
                logger.info(f"[AgentManager] Project {cid_hash} - No changes in tools.")

        return self.miner_agent

    def get_projects(self):
        return self.project_manager.get_projects()

    def get_graphql_agent(self, cid_hash: str) -> GraphQLAgent:
        return self.graphql_agent[cid_hash]

    def get_miner_agent(self, cid_hash: str | None = None) -> dict | tuple[StateGraph, StateGraph, GraphQLAgent]:
        if cid_hash:
            config = self.miner_agent.get(cid_hash, {})
            return (
                config.get('agent_graph', None),
                config.get('miner_agent', None),
                config.get('graphql_agent', None)
            )
        return self.miner_agent