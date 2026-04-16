from agent.subquery_graphql_agent.project import LocalProjectSubquery
from ..base import BaseAgent


class SubqueryAgent(BaseAgent):
    def __init__(self, project: LocalProjectSubquery):
        super().__init__(project)
