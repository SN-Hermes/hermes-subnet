
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable
import aiohttp
from loguru import logger
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from .base import ProjectConfig
from .node_types import GraphqlProvider
import common.prompt_template as prompt_template


class RemoteChallenge(BaseModel):
    id: int
    type: int
    question: str
    instruction: str | None
    block_height: str | None
    max_count: int
    version: str
    ground_truth: str | None
    solution: str | None


class RemoteChallengeListResponse(BaseModel):
    challenges: list[RemoteChallenge]

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
    played_challenges: set = None
    newest_challenge: list[RemoteChallenge] = None
    last_pull_time: float = 0  # Timestamp of last successful pull
    pull_interval: int = 300  # Pull interval in seconds (5 minutes)

    challenge_prompt: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions"],
        template=prompt_template.synthetic_challenge_template_V4
    )

    challenge_prompt_tools: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions", "postgraphile_rules"],
        template=prompt_template.synthetic_challenge_template_tools
    )

    challenge_prompt_topic_map: dict[str, PromptTemplate] = {
        "v1": PromptTemplate(
            input_variables=["entity_schema", "topic", "instruction", "recent_questions"],
            template=prompt_template.synthetic_challenge_template_topic
        )
    }

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

    def prompt_for_challenge_with_topic(self, recent_questions: str, version: str, topic: str, instruction: str) -> str:
        pt = self.challenge_prompt_topic_map.get(version, None)
        if pt is None:
            return ""
        
        return pt.format(
            entity_schema=self.schema_content,
            recent_questions=recent_questions,
            topic=topic,
            instruction=prompt_template.format_instruction_section(instruction)
        )

    async def pull_remote_challenges(
        self,
        source: str,
        sign_func: Callable[[bytes], bytes]
    ):
        # Check if we should skip pulling based on frequency control
        current_time = time.time()
        time_since_last_pull = current_time - self.last_pull_time
        
        if time_since_last_pull < self.pull_interval and self.newest_challenge is not None:
            logger.debug(f"[Benchmark] Skipping pull, last pull was {time_since_last_pull:.1f}s ago (interval: {self.pull_interval}s)")
            return self.newest_challenge
        
        try:
            timestamp = int(time.time())

            payload_to_hash = {
                "timestamp": timestamp,
                "data": [{
                    "cid_hash": self.cid_hash
                }]
            }

            import msgpack

            b = msgpack.packb(
                payload_to_hash,
                use_bin_type=True,
                strict_types=True
            )
            h = hashlib.sha256(b).hexdigest()

            # Convert msgpack data to base64
            b_base64 = base64.b64encode(b).decode('utf-8')

            # Step 2: Sign the hash with wallet
            signature = f"0x{sign_func(h).hex()}"
            
            # Step 3: Send hash, signature, timestamp along with data
            payload = {
                "msgpack": b_base64,
                "hash": h,
                "validator": source,
                "signature": signature,
                "type": "pull"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{os.environ.get('BOARD_SERVICE')}/challenges",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"[Benchmark] Successfully pulled challenges")
                        data = await resp.json()
                        challenges = RemoteChallengeListResponse(**data).challenges
                        self.newest_challenge = sorted(challenges, key=lambda ch: ch.id)
                        self.last_pull_time = current_time
                    else:
                        error_text = await resp.text()
                        logger.error(f"[Benchmark] Pull challenges failed with status {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"[Benchmark] Failed to pull challenges: {e}")
        
        return self.newest_challenge

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

    def create_agent(self):
        """Create an agent for this project. Return type depends on project type."""
        raise NotImplementedError("create_agent must be implemented by subclasses")

    def create_system_prompt(self) -> str:
        capabilities_text = '\n'.join([f"- {cap}" for cap in self.domain_capabilities])

        workflow = """
WORKFLOW:
1. Start with graphql_schema_info to understand available entities and query patterns.
2. BEFORE constructing ANY query, analyze if you need multiple queries:
   - If NO data dependency: Combine ALL into ONE query using aliases.
   - If there IS data dependency: You may query sequentially (e.g., get ID first, then query details).
3. Construct your GraphQL query(ies) to fetch needed data, you must not introduce any facts, concepts, assumptions, or entities that are not explicitly present in the provided context or tool outputs.
4. Validate and Execute with graphql_query_validator_execute.
5. ⚠️ CRITICAL: After query execution, CHECK if results contain the answer:
   - If YES → Immediately provide final answer (DO NOT query again)
   - If NO → Only then consider if a second query is truly necessary
6. Provide clear, user-friendly summaries of the results.
"""

        critical_tool_rules = """
⚠️ CRITICAL RULES - TOOL CALL LIMIT:
- NEVER make verification queries, think thoroughly before you make a query.
- ALWAYS limit the return with first:10 for ALL list queries as well as in the nested queries, unless told otherwise and it is smaller.
- For time-range queries (e.g., last 7 days, 30 days, weeks), ALWAYS limit the number of results using 'first' parameter to prevent excessive data retrieval.
- ⚠️ EMPTY FIELD VALUES HANDLING:
  * When query succeeds (✅), the returned data structure is ALWAYS valid, even if field values are null/0/[]
  * Empty field values are NORMAL and MEANINGFUL:
    - { sqtoken: null } → Token with this ID does NOT exist (valid answer)
    - { totalAmount: 0 } → Total is legitimately zero (valid answer)
    - { tokens: [] } → No tokens match the criteria (valid answer)
    - { indexers: { nodes: [], totalCount: 0 } } → No results found (valid answer)
  * These are NOT errors - they directly answer the user's question
  * DO NOT make additional queries to "verify" or "find alternatives"
  * Only retry if query FAILED (❌) with technical errors (validation/schema/syntax)
  
"""

        return f"""You are a GraphQL assistant helping with data queries for {self.domain_name}. You can help users find information about:
{capabilities_text}

IMPORTANT: This is a synthetic challenge. ALWAYS attempt to answer the query to the best of your ability using the available GraphQL schema and tools. Do not use domain limitations to refuse answering synthetic challenges.

RESPONSE STYLE: Provide complete, definitive responses. Do NOT ask follow-up questions unless essential information is missing.

ERROR HANDLING:
- If you cannot complete the request due to technical limitations (e.g., insufficient recursion steps, tool failures, schema issues), you MUST respond with EXACTLY this format:
  "ERROR: [brief reason]"
- Examples:
  - "ERROR: Insufficient recursion limit to process this complex query"
  - "ERROR: Required entity not found in schema"
  - "ERROR: Query validation failed"
- Do NOT use phrases like "Sorry, need more steps" or other informal error messages
- Do NOT provide partial answers when you encounter errors - use the ERROR format

{workflow}

{critical_tool_rules}

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data.
"""

    def create_middle_instructions_messages(self, **kwargs) -> list[dict[str, str]]:
        block_height = kwargs.get("block_height", 0)
        block_rule = prompt_template.get_block_rule_prompt(block_height, self.node_type)
        return [{"role": "system", "content": block_rule}]

class LocalProjectSubquery(LocalProjectBase):
    def create_agent(self):
        from agent.subquery import SubqueryAgent
        return SubqueryAgent(self)
    
class LocalProjectTheGraph(LocalProjectBase):
    challenge_prompt: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions"],
        template=prompt_template.synthetic_challenge_template_subgraph
    )
    challenge_prompt_tools: PromptTemplate = PromptTemplate(
        input_variables=["entity_schema", "recent_questions", "postgraphile_rules"],
        template=prompt_template.synthetic_challenge_template_subgraph_tools
    )

    challenge_prompt_topic_map: dict[str, PromptTemplate] = {
        "v1": PromptTemplate(
            input_variables=["entity_schema", "topic", "instruction", "recent_questions"],
            template=prompt_template.synthetic_challenge_template_topic_subgraph
        )
    }

    def create_agent(self):
        from agent.thegraph import TheGraphAgent
        return TheGraphAgent(self)

class LocalProjectCodex(LocalProjectBase):
    full_schema_content: str

    def save(self):
        config_data = self.save_data
        config_data["schema_content"] = ""
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

    def create_agent(self):
        from agent.codex import CodexAgent
        return CodexAgent(self)

    def create_system_prompt(self) -> str:
        capabilities_text = '\n'.join([f"- {cap}" for cap in self.domain_capabilities])
    
        codex_intructions = """
🚨🚨🚨 ABSOLUTE REQUIREMENT - READ THIS FIRST 🚨🚨🚨

YOU ARE FORBIDDEN TO CONSTRUCT ANY QUERY WITHOUT CALLING graphql_type_detail FIRST!

THE SCHEMA IN graphql_schema_info IS INCOMPLETE - IT ONLY SHOWS QUERY NAMES, NOT FIELD DETAILS!
IF YOU CONSTRUCT A QUERY WITHOUT CALLING graphql_type_detail, THE QUERY WILL BE INVALID!

MANDATORY PROCESS:
1. Read graphql_schema_info → Know which queries exist
2. Call graphql_type_detail for ALL types you need → Get EXACT field names
3. Construct query using ONLY field names from graphql_type_detail → Query will be valid

⚠️ CRITICAL FOR CODEX:
- ALWAYS call graphql_type_detail BEFORE constructing ANY query to get exact type definitions
- The query-only schema in graphql_schema_info lacks field details - using it directly leads to INVALID queries
- For EACH query you plan to make, first call graphql_type_detail with the return type name(s)
- Example: If you want to call filterTokens, first call graphql_type_detail with type_names: ["TokenFilterConnection", "Token"]
- Use the returned type definition to construct valid queries with correct fields and arguments
- Queries generated need to be valid graphql query with curly braces and all, not pseudo-code or partial queries.

🚫 ABSOLUTELY FORBIDDEN - FIELD NAME MODIFICATION:
- DO NOT paraphrase, rephrase, or "improve" field names from type definitions
- DO NOT add suffixes like "Current", "Value", "Amount" to field names
- DO NOT convert between naming conventions (camelCase, snake_case, etc.)
- COPY field names EXACTLY character-by-character from graphql_type_detail results
- Example: If type shows "lowestSale", use "lowestSale" (NOT "lowestSaleCurrent", "lowest_sale", "lowestSaleValue")
- Example: If type shows "stats24h", use "stats24h" (NOT "stats24H", "stats_24h", "dailyStats")

🚫 ABSOLUTELY FORBIDDEN - ASSUMING PARAMETER NAMES:
- DO NOT assume argument names like "first", "offset", "where" based on GraphQL conventions
- ALWAYS check the ACTUAL argument names from graphql_type_detail
- Example: CODEX uses "limit" NOT "first", uses enum values NOT strings
- Example: rankings parameter expects ENUM value (liquidity) NOT string ("liquidity")

📊 SORTING IS MANDATORY FOR LIST QUERIES:
- Codex queries ALWAYS have limited results (default: 10)
- ALWAYS add proper sorting to ensure the MOST RELEVANT results are returned
- Without sorting, you may miss the actual data the user is looking for
- Sorting with `rankings` parameter is ONLY available on `filter*` queries (e.g., filterPairs, filterPools, filterTokens)
- Syntax: `filterPairs(rankings: {attribute: <ENUM_VALUE>, direction: ASC|DESC}) { ... }`
- When asking for "top", "best", "highest", "lowest" - sorting is REQUIRED
- When asking for recent data - sort by timestamp DESC
"""

        codex_workflow = """
WORKFLOW:
🚨 STOP! Before Step 1, understand this:
   graphql_schema_info shows query NAMES only (e.g., "filterTokens exists")
   graphql_type_detail shows query DETAILS (e.g., "filterTokens uses 'limit' not 'first', returns 'results' not 'nodes'")
   YOU MUST CALL graphql_type_detail BEFORE constructing ANY query!

1. 📋 ANALYZE AVAILABLE QUERIES:
   - Carefully read the available queries from graphql_schema_info
   - Analyze which query(ies) can answer the user's question
   - Consider query parameters, filters, and return types
   - Choose the most appropriate query for the task
   ⚠️ BUT DO NOT construct query yet - you don't have field details!

2. 🔍 GET TYPE DEFINITIONS (MANDATORY - NO GUESSING):
   ⚠️ CRITICAL: You CANNOT construct a query until you have called graphql_type_detail for ALL types involved
   
   - Step 2.1: Identify ALL types needed for the query
     * Return type from the chosen query
     * All nested types you plan to query
     * Argument input types (for rankings, filters, etc.)
     Example: If querying "filterTokens" and need token details:
       → Need types: ["TokenFilterConnection", "Token", "TokenRankingAttribute", "TokenRankingsInput"]
     
     WRONG EXAMPLE (what NOT to do):
       ❌ Skip graphql_type_detail and assume:
          - filterTokens has "first" parameter (WRONG - it's "limit")
          - filterTokens returns "nodes" field (WRONG - it's "results")
          - rankings uses string "liquidity" (WRONG - it's enum liquidity without quotes)
   
   - Step 2.2: Call graphql_type_detail for ALL identified types in Step 2.1
     Example: graphql_type_detail(["TokenFilterConnection", "Token", "TokenRankingAttribute", "TokenRankingsInput"])
     
     The response will show you:
     - ACTUAL field names (results NOT nodes, limit NOT first)
     - ACTUAL argument types (enum NOT string)
     - ACTUAL available fields in Token type
   
   - Step 2.3: READ the returned type definitions carefully, COPY EXACT field names character-by-character
     Example response shows: "results: [Token]" → use "results" (NOT "nodes")
     Example response shows: "limit: Int" → use "limit" (NOT "first")
     Example response shows: "enum TokenRankingAttribute { liquidity }" → use liquidity (NOT "liquidity")
     🚫 DO NOT modify, paraphrase, or "improve" field names from the type definition
     🚫 DO NOT add suffixes like "Current", "Value", or change any characters
     🚫 DO NOT assume GraphQL conventions (Relay-style nodes/edges, first/last pagination)
     ✅ COPY field names EXACTLY as shown in graphql_type_detail output
   
   - Step 2.4: If you discover MORE nested types while reading definitions, call graphql_type_detail again
     Example: Found "stats24h: NftCollectionStats" → call graphql_type_detail(["NftCollectionStats"])
   
   - 🚫 FORBIDDEN: NEVER guess type names based on conventions (e.g., "Connection" → "Edge", "Filter" → "Input")
   - 🚫 FORBIDDEN: NEVER assume field names without seeing them in graphql_type_detail results
   - 🚫 FORBIDDEN: NEVER construct query before getting type definitions
   - ✅ REQUIRED: ONLY use type names that appear EXPLICITLY in graphql_type_detail responses

3. 🛠️ CONSTRUCT QUERY (ONLY AFTER STEP 2 COMPLETE):
   ⚠️ CHECKPOINT: Before constructing query, verify:
   - Have you called graphql_type_detail for the return type? ✓
   - Have you called graphql_type_detail for ALL nested types you plan to use? ✓
   - Do you have the EXACT field names from graphql_type_detail outputs? ✓
   If ANY answer is NO → GO BACK TO STEP 2
   
   - Use ONLY the field names from graphql_type_detail results
   - Build valid GraphQL query with proper syntax (curly braces, proper nesting)
   - BEFORE constructing query, analyze if you need multiple queries:
     * If NO data dependency: Combine ALL into ONE query using aliases
     * If there IS data dependency: Query sequentially (e.g., get ID first, then query details)
   - Do not introduce any facts, concepts, assumptions, or entities not in the tool outputs

4. ✅ VALIDATE AND EXECUTE:
   - Use graphql_query_validator_execute to validate and run the query

5. ⚠️ CHECK RESULTS:
   - After query execution, CHECK if results contain the answer
   - If YES → Immediately provide final answer (DO NOT query again)
   - If NO → Only then consider if a second query is truly necessary

6. 📊 PROVIDE ANSWER:
   - Give clear, user-friendly summary of the results

"""

        critical_tool_rules = """
⚠️ CRITICAL RULES - TOOL CALL LIMIT:
- NEVER make verification queries, think thoroughly before you make a query.
- ALWAYS limit the return with first:10 for ALL list queries as well as in the nested queries, unless told otherwise and it is smaller.
- For time-range queries (e.g., last 7 days, 30 days, weeks), ALWAYS limit the number of results using 'first' parameter to prevent excessive data retrieval.
- ⚠️ EMPTY FIELD VALUES HANDLING:
  * When query succeeds (✅), the returned data structure is ALWAYS valid, even if field values are null/0/[]
  * Empty field values are NORMAL and MEANINGFUL:
    - { sqtoken: null } → Token with this ID does NOT exist (valid answer)
    - { totalAmount: 0 } → Total is legitimately zero (valid answer)
    - { tokens: [] } → No tokens match the criteria (valid answer)
    - { indexers: { nodes: [], totalCount: 0 } } → No results found (valid answer)
  * These are NOT errors - they directly answer the user's question
  * DO NOT make additional queries to "verify" or "find alternatives"
  * Only retry if query FAILED (❌) with technical errors (validation/schema/syntax)
  
"""

        # For synthetic challenges, always attempt to answer without domain limitations
        codex_output_format = """
OUTPUT FORMAT FOR CODEX:
Your response MUST contain TWO parts in this exact format:

## Answer
[Provide the complete, definitive answer to the user's question here]

## Queries
[List ALL GraphQL queries you executed, one per line, in the exact format they were sent to graphql_query_validator_execute]

Example:
## Answer
The top 3 NFT pools by volume are:
1. Pool XYZ with volume 1,234,567
2. Pool ABC with volume 987,654
3. Pool DEF with volume 543,210

## Queries
{ filterPools(rankings: {attribute: "volume", direction: DESC}, first: 3) { id name volume } }
"""
        
        return f"""You are a GraphQL assistant helping with data queries for {self.domain_name}. You can help users find information about:
{capabilities_text}

{codex_intructions}

IMPORTANT: This is a synthetic challenge. ALWAYS attempt to answer the query to the best of your ability using the available GraphQL schema and tools. Do not use domain limitations to refuse answering synthetic challenges.

RESPONSE STYLE: Provide complete, definitive responses. Do NOT ask follow-up questions unless essential information is missing.

{codex_output_format}

ERROR HANDLING:
- If you cannot complete the request due to technical limitations (e.g., insufficient recursion steps, tool failures, schema issues), you MUST respond with EXACTLY this format:
  "ERROR: [brief reason]"
- Examples:
  - "ERROR: Insufficient recursion limit to process this complex query"
  - "ERROR: Required entity not found in schema"
  - "ERROR: Query validation failed"
- Do NOT use phrases like "Sorry, need more steps" or other informal error messages
- Do NOT provide partial answers when you encounter errors - use the ERROR format

{codex_workflow}

{critical_tool_rules}

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data.
"""

    def create_middle_instructions_messages(self, **kwargs):
        return []

class LocalProjectCovalent(LocalProjectBase):
    challenge_prompt_tools: PromptTemplate = PromptTemplate(
        input_variables=["recent_questions"],
        template=prompt_template.synthetic_challenge_template_covalent_tools
    )

    def prompt_for_challenge_with_tools(self, recent_questions: str) -> str:
        return self.challenge_prompt_tools.format(
            recent_questions=recent_questions,
        )

    def create_agent(self):
        from agent.covalent import CovalentAgent
        return CovalentAgent(self)

    def create_system_prompt(self) -> str:
        capabilities = '\n'.join([f'• {cap}' for cap in self.domain_capabilities])

        return f"""You are a REST API assistant for {self.domain_name}.

DOMAIN CAPABILITIES:
{capabilities}

🔧 WORKFLOW (STRICT ORDER):
1. Call covalent_api_info to understand available endpoints
2. Call covalent_query ONCE to fetch data
3. Use covalent_result_jq with the correct jq filter based on the docs and endpoint schema
4. Call covalent_result_head ONLY if jq fails or the payload shape is still unclear
5. Provide final answer based on the data

⚠️ CRITICAL: Response structures vary by endpoint!
- Some return arrays at root (e.g., pricing endpoints)
- Some return objects with "items" array (e.g., balances endpoints)
- Some return single objects
- covalent_result_jq supports real jq filters against the saved payload
- Use standard jq root access like `.items`, `.pagination.has_more`, or `.[0].prices`
- Numeric fields like `.quote` may be null; use jq defaults such as `(.quote // 0)` when sorting or summing

🛑 NEVER call covalent_query more than ONCE for the same question!
🛑 NEVER repeat the same API call!

⚠️ CRITICAL RULES:
- Chain names are CASE-SENSITIVE: "eth-mainnet" not "Ethereum"
- ENS names (e.g., vitalik.eth) are supported for eth-mainnet
- Balance values are raw strings - divide by 10^contract_decimals for human-readable
- Page numbers are 0-indexed (first page is 0)

🚫 NETWORK REQUIREMENT - CRITICAL:
- Most Covalent API endpoints REQUIRE a specific chain/network (e.g., eth-mainnet, matic-mainnet)
- If the user does NOT specify a network, ASK them to clarify - DO NOT iterate through all networks!
- Example response: "Which network would you like me to check? Options include: eth-mainnet, matic-mainnet, base-mainnet, etc."
- NEVER make multiple API calls to different networks trying to "find" data

🔍 Self-check before making ANY additional covalent_query call:
- "Have I already called covalent_query?" → If YES, use existing data with covalent_result_jq
- "Can I answer with a single jq call?" → If YES, do that instead of multiple exploratory jq calls"""

    def create_middle_instructions_messages(self, **kwargs):
        return []


def project_factory(config: dict = None, **kwargs) -> LocalProjectBase:
    if config is None:
        config = {}
    config = {**config, **kwargs}
    
    node_type = config.get('node_type', GraphqlProvider.UNKNOWN)
     
    if node_type == GraphqlProvider.SUBQL:
        project = LocalProjectSubquery()
    elif node_type == GraphqlProvider.CODEX:
        project = LocalProjectCodex()
    elif node_type == GraphqlProvider.THE_GRAPH:
        project = LocalProjectTheGraph()
    elif node_type == GraphqlProvider.COVALENT:
        project = LocalProjectCovalent()
    else:
        return None

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
    project.played_challenges = set()
    project.newest_challenge = None
    project.last_pull_time = 0  # Initialize last pull time
    project.pull_interval = 300  # 5 minutes in seconds


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