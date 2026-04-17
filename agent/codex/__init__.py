from agent.subquery_graphql_agent.project import LocalProjectCodex
from ..base import BaseAgent


class CodexAgent(BaseAgent):
    def __init__(self, project: LocalProjectCodex):
        super().__init__(project)
