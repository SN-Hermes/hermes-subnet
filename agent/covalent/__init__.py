import os
from agent.covalent.context import CovalentContext
from agent.covalent.tools import create_covalent_tools
from agent.covalent.types import CovalentConfig
from agent.subquery_graphql_agent.project import LocalProjectCovalent
from ..base import BaseAgent

class CovalentAgent(BaseAgent):
    def __init__(self, project: LocalProjectCovalent):
        super().__init__(project)
        self.recursion_limit = 30

    def tools(self):
        config = CovalentConfig(
            base_url=self.project.endpoint,
            authorization=os.getenv("COVALENT_API_TOKEN")
        )
        context = CovalentContext()
        tools = create_covalent_tools(config, context)

        return tools
