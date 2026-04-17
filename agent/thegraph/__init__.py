from agent.subquery_graphql_agent.project import LocalProjectTheGraph
from ..base import BaseAgent


class TheGraphAgent(BaseAgent):
    def __init__(self, project: LocalProjectTheGraph):
        super().__init__(project)
