from typing import Literal
from loguru import logger
from langchain_openai import ChatOpenAI
from typing import (
    Literal,
)
from agent.base import BaseAgent
from common.project_manager import ProjectManager


class AgentManager:
    project_manager: ProjectManager
    agent: dict[str, BaseAgent]
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
        self.agent = {}
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
            local_projects = self.project_manager.get_local_projects()
            for cid_hash, config in local_projects.items():
                self.ipc_common_config.update({
                    cid_hash: {
                        "node_type": config.node_type,
                        "endpoint": config.endpoint,
                    }
                })
        
        self._init_agents()

    def _init_agents(self):
        removed_agents = []
        new_agents = []
        for cid_hash, p in self.get_local_projects().items():
            enabled = self.is_project_enabled(cid_hash)
            if not enabled and cid_hash in self.agent:
                removed_agents.append(cid_hash)
                del self.agent[cid_hash]
                continue

            if enabled and cid_hash not in self.agent:
                new_agents.append(cid_hash)
                self.agent[cid_hash] = p.create_agent()
        
        if new_agents:
            logger.info(f"[AgentManager] Created agents for projects: {new_agents}")
        
        if removed_agents:
            logger.info(f"[AgentManager] Removed agents for projects: {removed_agents}")

    def get_local_projects(self):
        return self.project_manager.get_local_projects()

    def get_agent(self, cid_hash: str) -> BaseAgent | None:
        return self.agent.get(cid_hash, None)
    
    def get_agents(self):
        return self.agent

    def is_project_enabled(self, cid_hash: str) -> bool:
        return self.project_manager.is_project_enabled(cid_hash)

    def get_project_phase(self, cid_hash: str) -> int:
        return self.project_manager.get_project_phase(cid_hash)
