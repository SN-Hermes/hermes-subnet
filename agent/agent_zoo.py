import os
from pathlib import Path
import pkgutil
import sys
from typing import ClassVar, Dict
import importlib
import inspect
from loguru import logger
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph
from agent.stats import ToolCountHandler


class AgentZoo:
    '''
    { 
        [project_name]: {
            tools: {},
            agent: agent,
            counter: counter,
        }
    }
    '''
    agent_configs: ClassVar[Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph | ToolCountHandler]]] = {}
    counter_configs: ClassVar[Dict[str, ToolCountHandler]] = {}
  
    @classmethod
    def load_agents(cls, projects_dir: str = "projects") -> Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph]]:
        base_path = Path(projects_dir)
        for project_dir in base_path.iterdir():
            if not project_dir.is_dir():
                continue
            project_name = project_dir.name

            project = cls.agent_configs.get(project_name)
            prev_tools = cls.agent_configs.get(project_name, {}).get('tools', {})
            current_tools = {}
            
            for module_info in pkgutil.iter_modules([str(project_dir)]):
                module_name = module_info.name
                full_module = f"projects.{project_name}.{module_name}"

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
                # reconstruct agent
                agent = create_react_agent(
                    model="openai:gpt-4.1-mini",
                    tools=tools,
                    prompt=f"You are the agent for project {project_name}."
                )
                logger.info(f"[AGENT] load agent, Project {project_name} - tools: {[t.name for t in tools]}, Created: {[t.name for t in created]}, Updated: {[t.name for t in updated]}, Deleted: {deleted}")


                cls.agent_configs[project_name] = {
                    "tools": {t.name: getattr(type(t), "__version__", "0.0.0") for t in tools},
                    "agent": agent,
                    "counter": ToolCountHandler()
                }
            else:
                logger.info(f"[AGENT] Project {project_name} - No changes in tools.")
    
        return cls.agent_configs


    @classmethod
    def get_agent(cls, project_name: str) -> CompiledStateGraph | None:
        return cls.agent_configs.get(project_name, {}).get('agent')

    @classmethod
    def get_counter(cls, project_name: str) -> Dict[str, int] | None:
        return cls.counter_configs.get(project_name)

    @classmethod
    def remove_agent(cls, project_name: str):
        if project_name in cls.agent_configs:
            del cls.agent_configs[project_name]
            del cls.counter_configs[project_name]
            logger.info(f"[AGENT] Removed agent for {project_name}")


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




