import importlib
import json
from pathlib import Path
from typing import Literal
import pkgutil
import sys
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger
from langgraph.graph import StateGraph, MessagesState, START, END
from agent.stats import ToolCountHandler
from agent.subquery_graphql_agent.base import GraphQLAgent
from common.project_manager import ProjectConfig, ProjectManager
from common.utils import create_system_prompt


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

    async def start(self, pull=True, role: Literal["", "validator", "miner"] = ""):
        self.project_manager = ProjectManager(self.llm_synthetic, self.save_project_dir)

        if pull:
            await self.project_manager.pull()
        else:
            self.project_manager.load()

        if role == "miner":
            self.miner_agent = {}
            self._init_miner_agents()
        elif role == "validator":
            self.graphql_agent = {}
            self._init_agents()
        

    def _init_agents(self):
        for cid, project_config in self.get_projects().items():
            self.graphql_agent[cid] =  GraphQLAgent(project_config)
        logger.info(f"[AgentManager] Initialized graphql_agents for projects: {list(self.graphql_agent.keys())}")

    def _init_miner_agents(self):
        def miner_router(state):
            messages = state["messages"]
            for m in messages:
                # TODO: check tool call has real output
                if m.type == 'ai' and len(m.tool_calls) > 0:
                    return END
            # TODO: trim
            # first_message = messages[0:1]
            # state['messages'] = first_message
            return "graphql_agent"
        
        base_path = Path(self.save_project_dir)
        for project_dir in base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            cid = project_dir.name
            if cid == "__pycache__":
                continue

            project = self.miner_agent.get(cid)
            prev_tools = self.miner_agent.get(cid, {}).get('tools', {})
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
                    prompt= f"You are the agent for project {cid}."
                )
                logger.info(f"[AgentManager] load agent, Project {cid} using model {self.llm_synthetic.model_name} - tools: {[t.name for t in tools]}, Created: {[t.name for t in created]}, Updated: {[t.name for t in updated]}, Deleted: {deleted}, with prompt: {suc}")


                builder = StateGraph(MessagesState)
                builder.add_node("miner_agent", miner_agent)
                builder.add_node("graphql_agent", graphql_agent.executor)
                builder.add_conditional_edges(
                    "miner_agent",
                    miner_router
                )
                builder.add_edge(START, "miner_agent")
                multi_agent_graph = builder.compile()

                self.miner_agent[cid] = {
                    "tools": {t.name: getattr(type(t), "__version__", "0.0.0") for t in tools},
                    "miner_agent": miner_agent,
                    "graphql_agent": graphql_agent,
                    "agent_graph": multi_agent_graph,
                    "counter": ToolCountHandler()
                }
            else:
                logger.info(f"[AgentManager] Project {cid} - No changes in tools.")

        return self.miner_agent

    def get_projects(self):
        return self.project_manager.get_projects()

    def get_graphql_agent(self, cid) -> GraphQLAgent:
        return self.graphql_agent[cid]
    
    def get_miner_agent(self, cid: str | None = None) -> dict | tuple[StateGraph, StateGraph, GraphQLAgent]:
        if cid:
            config = self.miner_agent.get(cid, {})
            return (
                config.get('agent_graph', None),
                config.get('miner_agent', None),
                config.get('graphql_agent', None)
            )
        return self.miner_agent