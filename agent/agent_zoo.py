import json
import os
from pathlib import Path
import pkgutil
import sys
from typing import ClassVar, Dict
import importlib
from loguru import logger
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, MessagesState, START, END

from agent.stats import ToolCountHandler
from common.project_manager import ProjectConfig
from common.utils import create_system_prompt
import agent.graphql_agent as subAgent


class AgentZoo:
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
    agent_configs: ClassVar[Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph | ToolCountHandler]]] = {}
    counter_configs: ClassVar[Dict[str, ToolCountHandler]] = {}
  
    @classmethod
    def load_agents(cls, projects_dir: Path) -> Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph]]:
        
        def miner_router(state):
            messages = state["messages"]
            for m in messages:
                # TODO: check tool call has real output
                if m.type == 'ai' and len(m.tool_calls) > 0:
                    return END
            # TODO: trim
            first_message = messages[0:1]
            # return {"next": "server_agent", "state_update": {"messages": first_message}}
            return "server_agent"
        
        base_path = Path(projects_dir)
        for project_dir in base_path.iterdir():
            if not project_dir.is_dir():
                continue
            
            cid = project_dir.name
            if cid == "__pycache__":
                continue

            project = cls.agent_configs.get(cid)
            prev_tools = cls.agent_configs.get(cid, {}).get('tools', {})
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

                prompt = create_system_prompt(
                    domain_name=config.get("domain_name", ""),
                    domain_capabilities=config.get("domain_capabilities", []),
                    decline_message=config.get("decline_message", "")
                ) if suc else f"You are the agent for project {cid}."

                # reconstruct agent
                model = os.environ.get("MINER_LLM_MODEL", "gpt-4o-mini")

                server_agent = subAgent.initServerAgentWithConfig(ProjectConfig(**config))
                miner_agent = create_react_agent(
                    model="openai:" + model,
                    tools=tools,
                    prompt= f"You are the agent for project {cid}."
                )
                logger.info(f"[AGENT] load agent, Project {cid} using model {model} - tools: {[t.name for t in tools]}, Created: {[t.name for t in created]}, Updated: {[t.name for t in updated]}, Deleted: {deleted}, with prompt: {suc}")


                builder = StateGraph(MessagesState)
                builder.add_node("miner_agent", miner_agent)
                builder.add_node("server_agent", server_agent.executor)
                builder.add_conditional_edges(
                    "miner_agent",
                    miner_router
                )
                builder.add_edge(START, "miner_agent")
                multi_agent_graph = builder.compile()

                cls.agent_configs[cid] = {
                    "tools": {t.name: getattr(type(t), "__version__", "0.0.0") for t in tools},
                    "miner_agent": miner_agent,
                    "server_agent": server_agent,
                    "agent_graph": multi_agent_graph,
                    "counter": ToolCountHandler()
                }
            else:
                logger.info(f"[AGENT] Project {cid} - No changes in tools.")
    
        return cls.agent_configs

    @classmethod
    def get_agent(cls, cid: str) -> CompiledStateGraph | None:
        return cls.agent_configs.get(cid, {}).get('agent')

    @classmethod
    def get_counter(cls, cid: str) -> Dict[str, int] | None:
        return cls.counter_configs.get(cid)

    @classmethod
    def remove_agent(cls, cid: str):
        if cid in cls.agent_configs:
            del cls.agent_configs[cid]
            del cls.counter_configs[cid]
            logger.info(f"[AGENT] Removed agent for {cid}")


# python -m agent.agent_zoo
if __name__ == "__main__":
    agents = AgentZoo.load_agents()
    for project, agent in agents.items():
        print(f"Loaded agent for {project}: {agent}")

    count = ToolCountHandler()
    logger.info(f" init count: {count.stats()}")


    logger.info(agents.get('Qmww').get('agent').invoke(
        {"messages": [{"role": "user", "content": "what is 11 + 49?"}]},
        config={"callbacks": [count]}
    ))
    logger.info(f"\n\n one count: {count.stats()}")

    logger.info(agents.get('Qmww').get('agent').invoke(
        {"messages": [{"role": "user", "content": "what is 11 + 50?"}]},
        config={"callbacks": [count]}
    ))
    logger.info(f"\n\n second count: {count.stats()}")




