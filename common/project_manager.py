from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Dict, List
import aiohttp
import httpx
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel
import yaml
from agent.subquery_graphql_agent.base import ProjectConfig
from agent.subquery_graphql_agent.node_types import detect_node_type
import common.utils as utils


class Metadata(BaseModel):
    cid: str
    endpoint: str

class Project(BaseModel):
    enabled: bool
    description: str
    name: str
    metadata: Metadata

class ProjectData(BaseModel):
    data: List[Project]
    total: int
    page: int
    pageSize: int
    totalPages: int

class ProjectListResponse(BaseModel):
    code: int
    message: str
    data: ProjectData

ALLOWED_CID = []

class ProjectManager:
    projects: Dict[str, Project] = {}
    projects_config: Dict[str, ProjectConfig] = {}
    target_dir: Path | None = None
    llm: ChatOpenAI

    def __init__(self, llm: ChatOpenAI, target_dir: Path | None = None):
        self.llm = llm
        if target_dir is not None:
            self.target_dir = Path(target_dir)

    async def pull(self, silent: bool = False):
        """pull projects from board service with pagination."""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        
        board_url = os.environ.get('BOARD_SERVICE')
        if not board_url:
            logger.error("[ProjectManager] BOARD_SERVICE environment variable is not set.")
            sys.exit(1)
        
        page_size = 50
        offset = 0
        total_fetched = 0
        all_projects: list[Project] = []
        
        async with aiohttp.ClientSession() as session:
            while True:
                data = {
                    "enabled": True,
                    "limit": page_size,
                    "offset": offset,
                }

                if not silent:
                    logger.info(f"[ProjectManager] Fetching projects: offset={offset}, limit={page_size}")
                async with session.post(f"{board_url}/project/list", headers=headers, json=data) as resp:
                    response_data = await resp.json()
                
                parsed = ProjectListResponse(**response_data)
                current_batch = parsed.data.data
                all_projects.extend(current_batch)
                total_fetched += len(current_batch)

                if not silent:
                    logger.info(f"[ProjectManager] Fetched {len(current_batch)} projects, total: {total_fetched}/{parsed.data.total}")

                if total_fetched >= parsed.data.total or len(current_batch) == 0:
                    break
                
                offset += page_size

        if not silent:
            logger.info(f"[ProjectManager] Total projects fetched: {len(all_projects)}")

        # Process all fetched projects
        for project in all_projects:
            cid = project.metadata.cid
            combined = f"{cid}{project.metadata.endpoint}"
            hash_value = utils.hash256(combined)[:8]
            key = f"{cid}_{hash_value}"
            self.projects.update({key: project})

        for cid_hash, project in self.projects.items():
            if ALLOWED_CID and project.metadata.cid not in ALLOWED_CID:
                logger.warning(f"[ProjectManager] Project {project.metadata.cid} is not in the allowed list.")
                continue

            existing_config = self._load_existing_project(cid_hash)
            if existing_config:
                if cid_hash not in self.projects_config:
                    if not silent:
                        logger.info(f"[ProjectManager] Loading existing project: {existing_config.domain_name} ({cid_hash})")
                    self.projects_config[cid_hash] = existing_config
            else:
                # Register projects
                await self.register_project(cid_hash, project.metadata.endpoint)
        
    def load(self):
        projects = {}
        for project_dir in self.target_dir.iterdir():
            if not project_dir.is_dir():
                continue
            cid_hash = project_dir.name
            if cid_hash == "__pycache__":
                continue
            config_file = project_dir / "config.json"
            if not config_file.exists():
                continue
            with open(config_file) as f:
                config = json.load(f)
            projects[cid_hash] = ProjectConfig(**config)

        self.projects_config.update({cid_hash: config for cid_hash, config in projects.items()})
        return self.projects_config

    def get_project(self, cid: str) -> ProjectConfig:
        return self.projects_config.get(cid)

    def get_projects(self) -> Dict[str, ProjectConfig]:
        return self.projects_config

    async def pull_manifest(self, cid: str) -> Dict:
        try:
            logger.info(f"[ProjectManager] Fetching manifest for CID: {cid}")
            manifest_content = await utils.fetch_from_ipfs(cid)
            try:
                manifest = yaml.safe_load(manifest_content)
            except yaml.YAMLError:
                manifest = json.loads(manifest_content)
            return manifest
        except Exception as e:
            raise RuntimeError(f"[ProjectManager] Failed to pull manifest {cid}: {str(e)}")

    async def pull_schema(self, cid: str, manifest: Dict) -> str:
        try:
            # Handle different schema path formats
            schema_info = manifest.get('schema', {})
            if isinstance(schema_info, dict):
                # The Graph format: schema: { file: { "/": "/ipfs/QmXXX" } }
                if 'file' in schema_info and isinstance(schema_info['file'], dict) and '/' in schema_info['file']:
                    schema_path = schema_info['file']['/']
                    if schema_path.startswith('/ipfs/'):
                        # Extract CID from The Graph format: /ipfs/QmXXX
                        schema_cid = schema_path.replace('/ipfs/', '')
                        logger.debug(f"[ProjectManager] Fetching The Graph schema from IPFS CID: {schema_cid}")
                        schema_content = await utils.fetch_from_ipfs(schema_cid)
                    else:
                        logger.debug(f"[ProjectManager] Fetching schema file: {schema_path}")
                        schema_content = await utils.fetch_from_ipfs(cid, schema_path)
                else:
                    # SubQL format: schema: { file: "schema.graphql" }
                    schema_path = schema_info.get('file', 'schema.graphql')
                    if schema_path.startswith('http'):
                        logger.debug(f"[ProjectManager] Fetching schema from external URL: {schema_path}")
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            schema_response = await client.get(schema_path)
                            schema_response.raise_for_status()
                            schema_content = schema_response.text
                    elif schema_path.startswith('ipfs://'):
                        schema_cid = schema_path.replace('ipfs://', '')
                        logger.debug(f"[ProjectManager] Fetching SubQL schema from IPFS CID: {schema_cid}")
                        schema_content = await utils.fetch_from_ipfs(schema_cid)
                    else:
                        logger.debug(f"[ProjectManager] Fetching schema file: {schema_path}")
                        schema_content = await utils.fetch_from_ipfs(cid, schema_path)
            else:
                # Fallback for simple string format
                schema_path = str(schema_info) if schema_info else 'schema.graphql'
                logger.debug(f"[ProjectManager] Fetching schema file: {schema_path}")
                schema_content = await utils.fetch_from_ipfs(cid, schema_path)

            return schema_content
        except Exception as e:
            raise RuntimeError(f"[ProjectManager] Failed to pull schema: {str(e)}")


    async def register_project(self, cid_hash: str, endpoint: str) -> ProjectConfig:
        try:
            # Project doesn't exist locally, need to analyze with LLM
            logger.info(f"[ProjectManager] Analyzing new project: {cid_hash} at {endpoint}")
            cid = cid_hash.split('_')[0]
            manifest = await self.pull_manifest(cid)
            schema_content = await self.pull_schema(cid, manifest)

            detected_node_type = detect_node_type(manifest)
            logger.info(f"Detected node type: {detected_node_type}")
            
            llm_analysis = await self.analyze_project_with_llm(manifest, schema_content, llm=self.llm)

            config = ProjectConfig(
                cid=cid,
                endpoint=endpoint,
                cid_hash=cid_hash,
                schema_content=schema_content,
                node_type=detected_node_type,
                manifest=manifest,
                domain_name=llm_analysis["domain_name"],
                domain_capabilities=llm_analysis["domain_capabilities"],
                decline_message=llm_analysis["decline_message"],
                suggested_questions=llm_analysis.get("suggested_questions", []),
            )
            # Save to disk and memory
            self._save_project(config)

            self.projects_config[cid_hash] = config
            logger.info(f"[ProjectManager] Registered new project: {llm_analysis['domain_name']} ({cid_hash}) with endpoint {endpoint}")
            return config
        except Exception as e:
            raise RuntimeError(f"[ProjectManager] Failed to register project {cid_hash} with endpoint {endpoint}: {str(e)}")


    async def analyze_project_with_llm(self, manifest: dict, schema_content: str, llm=None) -> dict:
        """
        Use LLM to analyze project manifest and schema to generate appropriate prompts.
        Args:
            manifest: Project manifest data
            schema_content: GraphQL schema content
        
        Returns:
            dict: Generated domain_name, domain_capabilities, and decline_message
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
        
            # Use provided LLM or create one with same config as GraphQLAgent
            # TODO: improve. can't change temperature dynamiclly
            if llm is None:
                model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=0  # Same as GraphQLAgent
                )
            # Prepare schema content for LLM (truncate if too long)
            schema_preview = schema_content[:3000] if len(schema_content) > 3000 else schema_content
        
            # Get project basics
            project_name = manifest.get('name', 'Unknown Project')
            project_description = manifest.get('description', '')
        
            # Get network/chain info
            network_info = ""
            if 'network' in manifest:
                network = manifest['network']
                if isinstance(network, dict):
                    chain_id = network.get('chainId', network.get('endpoint', ''))
                    network_info = f"Network: {chain_id}"
        
            # Get datasource info
            datasources_info = ""
            if 'dataSources' in manifest:
                ds_kinds = [ds.get('kind', 'unknown') for ds in manifest['dataSources']]
                datasources_info = f"Data sources: {', '.join(set(ds_kinds))}"
        
            # Create focused analysis prompt
            analysis_prompt = f"""Analyze this SubQuery indexing project and generate specific agent configuration:

PROJECT INFO:
- Name: {project_name}
- Description: {project_description}
- {network_info}
- {datasources_info}

GRAPHQL SCHEMA:
```graphql
{schema_content}
```

Based on the project info and GraphQL schema entities, generate:

1. A clear domain_name that describes what this project indexes
2. Specific domain_capabilities based on the actual GraphQL entities and what queries users can make
3. A decline_message that mentions the specific domain
4. Suggested questions that users can ask to explore the data

IMPORTANT: Look at the GraphQL types to understand what this project tracks.

Respond ONLY with valid JSON in this exact format (no markdown code blocks):
{{
  "domain_name": "Specific Project Name",
  "domain_capabilities": [
    "Query [specific entity] data and relationships",
    "Analyze [specific metrics] and trends", 
    "Track [specific events/transactions]",
    "Monitor [specific blockchain activities]"
  ],
  "decline_message": "I'm specialized in {project_name} data queries. I can help you with [specific data types], but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
  "suggested_questions": [
    "Show me recent [specific entity type] transactions",
    "What are the top [entity] by [field]?",
    "How many [events] happened in the last day?",
    "Can you show me a sample GraphQL query for [entity]?"
  ]
}}

Make each capability very specific to the entities found in the schema."""

            logger.info(f"[ProjectManager] Project info - Name: {project_name}, Description: {project_description[:100]}...")
            logger.info("[ProjectManager] Analyzing project with LLM...")
            logger.debug(f"[ProjectManager] Network: {network_info}")
            logger.debug(f"[ProjectManager] Data sources: {datasources_info}")
            logger.debug(f"[ProjectManager] Schema length: {len(schema_content)} chars (preview: {len(schema_preview)} chars)")
            logger.debug(f"[ProjectManager] Sending prompt to LLM (length: {len(analysis_prompt)} chars)")
            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            logger.debug(f"[ProjectManager] LLM Raw Response: {response.content}")
        
            # Parse JSON response - handle markdown code blocks
            try:
                content = response.content.strip()
            
                # Remove markdown code blocks if present
                if content.startswith('```json'):
                 content = content[7:]  # Remove ```json
                if content.startswith('```'):
                    content = content[3:]   # Remove ```
                if content.endswith('```'):
                    content = content[:-3]  # Remove closing ```
            
                content = content.strip()
                result = json.loads(content)
            
                # Ensure all required fields are present
                if 'suggested_questions' not in result:
                    logger.warning("LLM response missing suggested_questions, adding defaults")
                    result['suggested_questions'] = [
                        "What types of data can I query from this project?",
                        "Show me a sample GraphQL query",
                        "What entities are available in this schema?",
                        "How can I filter the data?"
                    ]
            
                logger.info(f"[ProjectManager] LLM analysis completed: {result['domain_name']}")
                logger.info(f"[ProjectManager] Generated capabilities: {len(result['domain_capabilities'])} items")
                logger.info(f"[ProjectManager] Generated questions: {len(result['suggested_questions'])} items")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"[ProjectManager] LLM response was not valid JSON: {e}")
                logger.debug(f"[ProjectManager] Full raw response: {response.content}")
                logger.debug(f"[ProjectManager] Cleaned content: {content}")
                raise ValueError("Invalid JSON response from LLM")
            
        except Exception as e:
            logger.warning(f"[ProjectManager] LLM analysis failed: {e}, using enhanced fallback")
        
            # Enhanced fallback analysis
            project_name = manifest.get('name', 'SubQuery Project')
            project_description = manifest.get('description', '')
        
            # Generate better domain name
            if project_description and len(project_description) > 10:
                domain_name = f"{project_name} - {project_description[:50]}..."
            else:
                domain_name = project_name
            
            # Generate basic capabilities
            capabilities = [
                "Query blockchain data indexed by this project",
                "Analyze transaction patterns and trends", 
                "Track historical blockchain activities",
                "Monitor smart contract events and state changes"
            ]
            
            return {
                "domain_name": domain_name,
                "domain_capabilities": capabilities,
                "decline_message": f"I'm specialized in {project_name} data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
                "suggested_questions": [
                    "What types of data can I query from this project?",
                    "Show me a sample GraphQL query",
                    "What entities are available in this schema?",
                    "How can I filter the data?"
                ]
            }

    
    def _load_existing_project(self, cid_hash: str) -> ProjectConfig | None:
        """Load existing project configuration from local disk if it exists."""
        if self.target_dir is None:
            return None

        config_file = self.target_dir / cid_hash / "config.json"
        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Validate that the loaded config has all required fields
            required_fields = ['cid', 'endpoint', 'schema_content', 'domain_name', 'domain_capabilities']
            for field in required_fields:
                if field not in config_data:
                    logger.warning(f"[ProjectManager] Existing project {cid_hash} missing required field: {field}")
                    return None

            return ProjectConfig(**config_data)
        except Exception as e:
            logger.error(f"[ProjectManager] Failed to load existing project {cid_hash}: {e}")
            return None

    def _save_project(self, config: ProjectConfig):
        # current_dir = Path(__file__).parent
        # PROJECTS_DIR = current_dir.parent / "projects"
        dir = self.target_dir / config.cid_hash
        dir.mkdir(parents=True, exist_ok=True)

        file = dir / "config.json"
        with open(file, "w") as f:
            json.dump(asdict(config), f, indent=2)
