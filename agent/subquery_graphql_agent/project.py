
import json
from pathlib import Path
from loguru import logger
from langchain_core.prompts import PromptTemplate

from .base import ProjectConfig
from .node_types import GraphqlProvider
import common.prompt_template as prompt_template


class LocalProjectBase:
    cid: str
    endpoint: str
    schema_content: str
    cid_hash: str
    node_type: str = GraphqlProvider.UNKNOWN
    manifest: dict[str, any] = None
    domain_name: str = "GraphQL Project"
    domain_capabilities: list[str] = None
    decline_message: str = None
    local_dir: Path = None

    challenge_prompt: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions"],
        template=prompt_template.synthetic_challenge_template_V4
    )

    challenge_prompt_tools: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions", "postgraphile_rules"],
        template=prompt_template.synthetic_challenge_template_tools
    )

    @property
    def save_data(self) -> dict:
        return {
            "cid": self.cid,
            "endpoint": self.endpoint,
            "schema_content": self.schema_content,
            "cid_hash": self.cid_hash,
            "node_type": self.node_type,
            "manifest": self.manifest,
            "domain_name": self.domain_name,
            "domain_capabilities": self.domain_capabilities,
            "decline_message": self.decline_message,
        }

    def prompt_for_challenge(self, recent_questions: str) -> str:
        return self.challenge_prompt.format(
            entity_schema=self.schema_content,
            recent_questions=recent_questions
        )

    def prompt_for_challenge_with_tools(self, recent_questions: str, postgraphile_rules: str) -> str:
        return self.challenge_prompt_tools.format(
            entity_schema=self.schema_content,
            recent_questions=recent_questions,
            postgraphile_rules=postgraphile_rules
        )

    def save(self):
        self.local_dir.mkdir(parents=True, exist_ok=True)
        with open(self.local_dir / "config.json", "w") as f:
            json.dump(self.save_data, f, indent=2)

    
    def to_project_config(self) -> ProjectConfig:
        return ProjectConfig(
            cid=self.cid,
            endpoint=self.endpoint,
            schema_content=self.schema_content,
            cid_hash=self.cid_hash,
            node_type=self.node_type,
            manifest=self.manifest,
            domain_name=self.domain_name,
            domain_capabilities=self.domain_capabilities,
            decline_message=self.decline_message,
        )

class LocalProjectSubgraph(LocalProjectBase):
    challenge_prompt: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions"],
        template=prompt_template.synthetic_challenge_template_subgraph
    )
    challenge_prompt_tools: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions", "postgraphile_rules"],
        template=prompt_template.synthetic_challenge_template_subgraph_tools
    )


class LocalProjectCodex(LocalProjectBase):
    full_schema_content: str

    def save(self):
        config_data = self.save_data
        with open(self.local_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        with open(self.local_dir / "query.graphql", "w") as f:
            f.write(self.schema_content)
        
        with open(self.local_dir / "full_schema.graphql", "w") as f:
            f.write(self.full_schema_content)


    def to_project_config(self) -> ProjectConfig:
        return ProjectConfig(
            cid=self.cid,
            endpoint=self.endpoint,
            schema_content=self.schema_content,
            full_schema_content=self.full_schema_content,
            cid_hash=self.cid_hash,
            node_type=self.node_type,
            manifest=self.manifest,
            domain_name=self.domain_name,
            domain_capabilities=self.domain_capabilities,
            decline_message=self.decline_message,
        )



def project_factory(config: dict = None, **kwargs) -> LocalProjectBase:
    if config is None:
        config = {}
    config = {**config, **kwargs}
    
    node_type = config.get('node_type', GraphqlProvider.UNKNOWN)
     
    if node_type == GraphqlProvider.CODEX:
        project = LocalProjectCodex()
    elif node_type == GraphqlProvider.THE_GRAPH:
        project = LocalProjectSubgraph()
    else:
        project = LocalProjectBase()

    project.cid = config.get('cid')
    project.endpoint = config.get('endpoint')
    project.schema_content = config.get('schema_content')
    project.cid_hash = config.get('cid_hash')
    project.node_type = node_type
    project.manifest = config.get('manifest', {})
    project.domain_name = config.get('domain_name', 'GraphQL Project')
    project.domain_capabilities = config.get('domain_capabilities', [])
    project.decline_message = config.get('decline_message', None)
    project.local_dir = config.get('local_dir', None)


    if isinstance(project, LocalProjectCodex):
        if not project.schema_content:
            query_file = project.local_dir / "query.graphql"
            with open(query_file) as f:
                schema_content = f.read()
                project.schema_content = schema_content

        full_schema_content = config.get('full_schema_content', '')
        if not full_schema_content:
            full_schema_file = project.local_dir / "full_schema.graphql"
            with open(full_schema_file) as f:
                project.full_schema_content = f.read()

    return project


def from_file(path: Path) -> LocalProjectBase | None:
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

        # Validate that the loaded config has all required fields
        required_fields = ['cid_hash', 'endpoint', 'schema_content', 'domain_name', 'domain_capabilities', 'node_type']
        for field in required_fields:
            if field not in data:
                logger.warning(f"[ProjectManager] Existing project {path} missing required field: {field}")
                return None

        return project_factory(config=data, local_dir=path.parent)