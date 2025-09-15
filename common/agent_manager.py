from langchain_openai import ChatOpenAI
from loguru import logger
from agent.subquery_graphql_agent.base import GraphQLAgent
from common.project_manager import ProjectManager


class AgentManager:
    project_manager: ProjectManager
    graphql_agent: dict[str, GraphQLAgent]
    save_project_dir: str
    llm_synthetic: ChatOpenAI
    llm_score: ChatOpenAI

    def __init__(self, save_project_dir: str, llm_synthetic: ChatOpenAI, llm_score: ChatOpenAI):
        self.save_project_dir = save_project_dir
        self.graphql_agent = {}
        self.llm_synthetic = llm_synthetic
        self.llm_score = llm_score

    async def start(self, pull=True):
        self.project_manager = ProjectManager(self.llm_synthetic, self.save_project_dir)

        if pull:
            await self.project_manager.pull()
        else:
            self.project_manager.load()

        self._init_agents()

    def _init_agents(self):
        for cid, project_config in self.get_projects().items():
            self.graphql_agent[cid] =  GraphQLAgent(project_config)
        logger.info(f"[AgentManager] Initialized graphql_agents for projects: {list(self.graphql_agent.keys())}")

    def get_projects(self):
        return self.project_manager.get_projects()

    def get_agent(self, cid) -> GraphQLAgent:
        return self.graphql_agent[cid]