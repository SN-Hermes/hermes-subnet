"""GraphQL Tools for LLM agents."""

import json
import asyncio
from typing import Optional, Type, Dict, Any, Annotated
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger
import graphql
from graphql import build_client_schema, validate
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from .graphql import process_graphql_schema
from .node_types import GraphqlProvider
from .thegraph_tools import create_thegraph_schema_info_content


class GraphQLSchemaInfoInput(BaseModel):
    """Input for GraphQL schema info tool."""
    # No input needed for schema overview
    pass


class GraphQLSchemaInfoTool(BaseTool):
    """
    Tool to get comprehensive GraphQL schema information with automatic node type detection.
    Supports both SubQL (PostGraphile v4) and The Graph protocol patterns.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "graphql_schema_info"
    description: str = """
    Get the raw GraphQL entity schema with automatic node type detection and appropriate query patterns.
    
    Use this tool ONCE at the start, then use the raw schema to:
    1. Identify @entity types and infer their query patterns
    2. See all fields and their types to determine relationships
    3. Apply node-specific patterns (SubQL/PostGraphile or The Graph) to construct valid queries
    
    DO NOT call this tool multiple times. The raw schema contains everything needed.
    """
    args_schema: Type[BaseModel] = GraphQLSchemaInfoInput
    
    def __init__(self, graphql_source, node_type: str):
        super().__init__()
        self._graphql_source = graphql_source
        self._node_type = node_type
    
    @property
    def graphql_source(self):
        return self._graphql_source
    
    def _run(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> str:
        """Get GraphQL schema info synchronously."""
        return asyncio.run(self._arun(config=config, run_manager=run_manager))
    
    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        config: Annotated[RunnableConfig, InjectedToolArg] = None,
    ) -> str:
        """Get raw GraphQL schema with automatic node type detection and appropriate guidance."""
        try:
            # Extract block_height from config
            block_height = 0
            if config and "configurable" in config:
                block_height = config["configurable"].get("block_height", 0)
            # logger.info(f"[GraphQLSchemaInfoTool._arun] block_height from config: {block_height}")

            # Get raw schema from GraphQL source
            schema_content = self.graphql_source.entity_schema
            
            # Generate appropriate schema info based on node type
            if self._node_type == GraphqlProvider.THE_GRAPH:
                return create_thegraph_schema_info_content(schema_content, block_height)
            elif self._node_type == GraphqlProvider.SUBQL:
                return self._generate_subql_info(schema_content, block_height)
            
        except Exception as e:
            return f"Error reading schema info: {str(e)}"

    def _generate_subql_info(self, schema_content: str, block_height: int) -> str:
        """Generate SubQL-specific schema information."""
        return f"""📖 SUBQL (POSTGRAPHILE v4) SCHEMA & RULES:

🔍 RAW ENTITY SCHEMA:
{schema_content}

📋 POSTGRAPHILE v4 INFERENCE RULES:
- Each @entity type → database table with 2 queries: singular(id) & plural(filter/pagination)
- Fields with @derivedFrom → relationship fields, need subfield selection
- Foreign key fields ending in 'Id' → direct ID access
- System tables (_pois, _metadatas, _metadata) → ignore these

📖 POSTGRAPHILE v4 QUERY PATTERNS:
1. 📊 ENTITY QUERIES:
   - Single query: entityName(id: ID!) → EntityType
   - Collection query: entityNames(first: Int, filter: EntityFilter, orderBy: [EntityOrderBy!]) → EntityConnection

2. 🔗 RELATIONSHIP QUERIES:
   - Foreign key ID: fieldNameId (returns ID directly)
   - Single entity: fieldName {{ id, otherFields }}
   - Collection relationships: fieldName {{ nodes {{ id, otherFields }}, pageInfo {{ hasNextPage, endCursor }}, totalCount }}
   - With filters: fieldName(filter: {{ ... }}) {{ nodes {{ ... }}, totalCount }}

3. 📝 FILTER PATTERNS (PostGraphile Format):
   
   STRING FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - in: [String!], notIn: [String!]
   - lessThan, lessThanOrEqualTo, greaterThan, greaterThanOrEqualTo
   - Case insensitive: equalToInsensitive, inInsensitive, etc.
   - isNull: Boolean
   
   BIGINT/NUMBER FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - lessThan, lessThanOrEqualTo, greaterThan, greaterThanOrEqualTo
   - in: [BigInt!], notIn: [BigInt!]
   - isNull: Boolean
   
   BOOLEAN FILTERS:
   - equalTo, notEqualTo, distinctFrom, notDistinctFrom
   - in: [Boolean!], notIn: [Boolean!]
   - isNull: Boolean
   
   EXAMPLES:
   - {{ id: {{ equalTo: "0x123" }} }}
   - {{ status: {{ in: ["active", "pending"] }} }}
   - {{ count: {{ greaterThan: 100 }} }}
   - {{ name: {{ equalToInsensitive: "alice" }} }}

4. 📈 ORDER BY PATTERNS:
   - Format: Convert fieldName to UPPER_CASE with underscores, then add _ASC/_DESC
   - Conversion: camelCase → UPPER_SNAKE_CASE
   - Examples: id → ID_ASC, createdAt → CREATED_AT_DESC, projectId → PROJECT_ID_ASC

5. 📄 PAGINATION:
   - Forward: first: 10, after: "cursor"
   - Backward: last: 10, before: "cursor"
   - Offset: offset: 20, first: 10

6. 📊 AGGREGATION (PostGraphile Aggregation Plugin):
   
   GLOBAL AGGREGATES (all data):
   - aggregates {{ sum {{ fieldName }}, distinctCount {{ fieldName }}, min {{ fieldName }}, max {{ fieldName }} }}
   - aggregates {{ average {{ fieldName }}, stddevSample {{ fieldName }}, stddevPopulation {{ fieldName }} }}
   - aggregates {{ varianceSample {{ fieldName }}, variancePopulation {{ fieldName }}, keys }}
   
   GROUPED AGGREGATES (group by):
   - groupedAggregates(groupBy: [FIELD_NAME], having: {{ ... }}) {{ keys, sum {{ fieldName }} }}
   - groupBy: Required, uses UPPER_SNAKE_CASE format (same as orderBy)
   - having: Optional, uses same filter format as main query
   
   EXAMPLES:
   - {{ indexers {{ aggregates {{ sum {{ totalReward }}, distinctCount {{ projectId }} }} }} }}
   - {{ indexers {{ groupedAggregates(groupBy: [PROJECT_ID]) {{ keys, sum {{ totalReward }} }} }} }}

🚨 CRITICAL AGENT RULES:
1. For missing user info ("my rewards"), ASK for wallet/ID - NEVER fabricate data
2. Pass queries to graphql_query_validator_execute as plain text (no backticks/quotes)
3. Only use graphql_type_detail as FALLBACK when validation fails - prefer raw schema

⚠️ CRITICAL FOREIGN KEY RULES:
- Fields with @derivedFrom CANNOT be queried alone - they need subfield selection
- Use: fieldName {{ id, otherField }} NOT just fieldName
- Foreign key fields ending in 'Id' can be queried directly as they return ID values

⚠️ CRITICAL @jsonField RULES:
- Fields marked with @jsonField are stored as JSON and CANNOT be expanded
- Query @jsonField fields directly without subfield selection
- Example: metadata @jsonField → Use metadata NOT metadata {{ subfields }}
- @jsonField fields return raw JSON data, treat as scalar values

🔍 FOREIGN KEY IDENTIFICATION:
- Look at field TYPE, not field name, to determine relationship
- If field type is @entity → it's a foreign key relationship
  - Physical storage: <fieldName>Id exists and can be used in filters
  - Query usage: fieldName {{ subfields }} for object, fieldNameId for ID
  - Entity lookup: Use the TYPE name to find the @entity definition
- If field type is basic type/enum/@jsonField → NOT a foreign key
  - Query directly: fieldName (no subfield selection needed)
  - For @jsonField: Query as scalar, DO NOT expand subfields

⚠️ CRITICAL: Field type determines entity, NOT field name
- Field: project: Project → Look for @entity Project (not @entity project)
- Field: owner: Account → Look for @entity Account (not @entity owner)

📝 TYPE MAPPING EXAMPLES:
- project: Project → Find @entity Project, query project {{ id, owner }} or projectId
- owner: Account → Find @entity Account, query owner {{ id, address }} or ownerId
- delegator: Delegator → Find @entity Delegator, query delegator {{ id, amount }}
- status: String → Basic type: use status directly
- metadata: JSON @jsonField → Query metadata directly (NOT metadata {{ subfields }})
- type: IndexerType → Enum: use type directly

🎯 REMEMBER: Field name ≠ Entity name. Use TYPE to find the @entity definition!

📋 RELATIONSHIP QUERY EXAMPLES:
✅ {{ indexer(id: "0x123") {{ id, project {{ id, owner }} }} }}
✅ {{ project(id: "0x456") {{ id, indexers {{ nodes {{ id, status }}, totalCount }} }} }}
✅ {{ indexers {{ nodes {{ id, projectId, project {{ id, owner }} }} }} }}
❌ {{ project {{ indexers {{ id, status }} }} }} (missing nodes wrapper)

📋 @jsonField QUERY EXAMPLES:
✅ {{ project(id: "0x123") {{ id, metadata, config }} }} (query @jsonField directly)
✅ {{ indexers {{ nodes {{ id, metadata, settings }} }} }} (@jsonField as scalar)
❌ {{ project {{ metadata {{ name, description }} }} }} (@jsonField cannot be expanded)
❌ {{ indexer {{ config {{ threshold, timeout }} }} }} (@jsonField cannot have subfields)

📊 AGGREGATION QUERY EXAMPLES:
✅ {{ indexers {{ aggregates {{ sum {{ totalReward }}, distinctCount {{ projectId }} }} }} }}
✅ {{ projects {{ aggregates {{ average {{ totalBoost }}, max {{ totalReward }} }} }} }}
✅ {{ indexers {{ groupedAggregates(groupBy: [PROJECT_ID]) {{ keys, sum {{ totalReward }}, distinctCount {{ id }} }} }} }}
✅ {{ rewards {{ groupedAggregates(groupBy: [ERA, INDEXER_ID], having: {{ era: {{ greaterThan: 100 }} }}) {{ keys, sum {{ amount }} }} }} }}


When Constructing GraphQL queries, You Must follow these strict rules:

🚫 GraphQL Query Pagination Rules (STRICT)
1. Every list field MUST have a `first:` parameter.
2. The value of first must always be ≤ 5, unless the user explicitly asks for a larger number.
3. This rule applies recursively, including all nested connection fields such as: nodes, edges, items, delegations, deploymentBoosterSummaries, etc.
4. You must never use first: 100 or first: 50 or any value > 5 without explicit user request.
5. If you are unsure whether a field supports pagination, assume it does and apply first: 5.
6. ⚠️ CRITICAL: NEVER re-query with different pagination values (first: 5 → first: 50). Use your first result!

❌ Bad Pagination Examples — These are strictly forbidden:
# Missing pagination
{{ indexers {{ nodes {{ id }}  }} }}
{{ indexers {{ delegations {{ nodes {{ id }} }}   }} }}
{{ indexer(id: xxx) {{ delegations {{ nodes {{ id }} }} }} }}

# Overly large pagination (FORBIDDEN)
{{ indexers(first: 50) {{ nodes {{ id }}  }} }}
{{ indexers(first: 100) {{ nodes {{ id }}  }} }}
{{ deployments(first: 50, orderBy: DESC) {{ nodes {{ id }} }} }}  ← NEVER use first: 50!

# Re-querying with different pagination (CRITICAL VIOLATION)
First query:  {{ deployments(first: 5) {{ nodes {{ id }} }} }}
Second query: {{ deployments(first: 50) {{ nodes {{ id }} }} }}  ← FORBIDDEN! Use first result!

✅ Correct Pagination Examples — You must always follow this pattern:
{{ indexers(first: 5) {{ delegations(first: 5) {{ nodes {{ id }} }}   }} }}
{{ indexer(id: xxx) {{ delegations(first: 5) {{ nodes {{ id }}  }} }} }}
{{ deployments(first: 5, orderBy: AMOUNT_DESC) {{ nodes {{ id amount }} }} }}

🚫 Query Consolidation Rules (MANDATORY - Minimize Queries)
⚠️ CRITICAL: Minimize the number of GraphQL queries. Combine when possible!

Rules:
1. If queries are INDEPENDENT (no data dependency) → MUST combine into ONE query using aliases
2. If second query DEPENDS on first query result (e.g., needs an ID) → Sequential queries are ALLOWED
3. NEVER query the same entity multiple times for fields that could be fetched together
4. Always prefer fewer queries over more queries

✅ ALLOWED: Sequential queries with data dependency
Example: "Find highest stake indexer, then get its delegations"
First query: {{ indexers(first: 1, orderBy: TOTAL_STAKE_DESC) {{ nodes {{ id }} }} }}
Second query: {{ indexer(id: "result_from_first") {{ id totalStake delegations(first: 5) {{ nodes {{ id }} }} }} }}
→ This is OK because second query needs the ID from first query
    
❌ FORBIDDEN: Multiple queries without data dependency
# Bad - these could be combined (WRONG)
First query: {{ indexers(first: 5) {{ nodes {{ id }} }} }}
Second query: {{ delegations(first: 5) {{ nodes {{ id }} }} }}  ← These are independent!

# Bad - querying same entity for different fields (WRONG)
First query: {{ indexer(id: "0x123") {{ id totalStake }} }}
Second query: {{ indexer(id: "0x123") {{ delegations(first: 5) {{ nodes {{ id }} }} }} }}  ← Should combine!

# Bad - querying same collection separately for nodes and aggregates (WRONG)
First query: {{ deploymentBoosterSummaries(first: 5, orderBy: DESC) {{ nodes {{ id }} }} }}
Second query: {{ deploymentBoosterSummaries(first: 5, orderBy: DESC) {{ groupedAggregates {{ ... }} }} }}
← Should combine! You can query nodes AND aggregates in ONE query!

✅ CORRECT: Combine independent queries
{{
    indexers(first: 5) {{ nodes {{ id totalStake }} }}
    delegations(first: 5) {{ nodes {{ id amount }} }}
}}

✅ CORRECT: Combine all fields for same entity
{{
    indexer(id: "0x123") {{ 
        id 
        totalStake 
        delegations(first: 5) {{ nodes {{ id amount }} }}
    }}
}}

✅ CORRECT: Query nodes AND aggregates together
{{
    deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{
        nodes {{ id totalAmount consumer }}
        groupedAggregates(groupBy: [DEPLOYMENT_ID]) {{
            keys
            sum {{ totalAdded totalRemoved totalAmount }}
        }}
    }}
}}
← Both nodes and aggregates in ONE query!

🚫 Minimal Query Rules (CRITICAL - ONLY QUERY ESSENTIAL FIELDS)
1. Must Query only the fields that are directly relevant to answering the user's question.
2. Must Avoid fetching extra metadata, nested relationships, or unrelated entities.
3. Must Do not expand the query “just in case” — every field must serve a clear purpose.
4. Never query for summary, statistics, or unrelated datasets unless explicitly requested.
5. Each query must form a minimal and direct data path:
  - user question → relevant field(s) → answer.
6. ⚠️ CRITICAL: Apply minimal field selection to ALL levels including nested relationships!
   - Only query fields you actually need at EVERY level of the query
   - Do NOT query extra fields in nested objects "just in case"
   - Example: If you only need project.id, query "project {{ id }}" NOT "project {{ id owner metadata }}"

❌ Bad Minimal Examples — These are strictly forbidden:

❌ Too broad - querying unrelated fields:
{{ 
    indexers(first: 5) {{
        nodes {{
            id
            era
            totalStake
            delegations(first: 5) {{ 
                nodes {{ id amountEraValueAfter delegator {{ id }} }}
            }}
            rewards(first: 5) {{ nodes {{ id }} }}
        }}
    }}
}}

❌ Multiple Queries for same entity:
{{
    indexer(id: 0x123) {{ id }}
}}
{{
    indexer(id: 0x123) {{ id }}
}}

❌ CRITICAL: Querying extra fields in nested relationships (VERY COMMON MISTAKE):
# BAD - When you only need deployment.project.id:
{{
    deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{
        nodes {{
            id
            deployment {{
                id
                project {{
                    id
                    owner      # ← UNNECESSARY! Question doesn't need owner
                    metadata   # ← UNNECESSARY! Question doesn't need metadata
                }}
            }}
            totalAdded      # ← UNNECESSARY! Question doesn't need this
            totalRemoved    # ← UNNECESSARY! Question doesn't need this
            totalAmount
            consumer        # ← UNNECESSARY! Question doesn't need consumer
            createAt        # ← UNNECESSARY! Question doesn't need timestamps
            updateAt        # ← UNNECESSARY! Question doesn't need timestamps
        }}
    }}
}}

❌ Querying @jsonField when not needed:
{{
    projects(first: 5) {{
        nodes {{
            id
            metadata     # ← Only query if question asks about metadata
            config       # ← Only query if question asks about config
        }}
    }}
}}

✅ Correct Minimal Examples — You must always follow this pattern:

✅ Only query fields actually needed:
{{
    indexers(first: 5) {{ nodes {{ id totalStake }} }}
}}

✅ CORRECT: Minimal nested fields (only what's needed):
# If question only needs deployment.id and deployment.project.id and totalAmount:
{{
    deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{
        nodes {{
            id
            deployment {{
                id
                project {{ id }}      # ← Only query project.id, nothing else!
            }}
            totalAmount              # ← Only totalAmount, no other amount fields
        }}
    }}
}}

✅ If you need deployment.metadata (which is @jsonField):
{{
    deploymentBoosterSummaries(first: 5) {{
        nodes {{
            id
            deployment {{ 
                id 
                metadata     # ← OK to query @jsonField if needed
                project {{ id }}
            }}
            totalAmount
        }}
    }}
}}

🔍 Before querying ANY field, ask yourself:
- "Does the user's question explicitly need this field?" → If NO, don't query it
- "Am I querying createAt/updateAt?" → Remove unless question asks about time
- "Am I querying owner/consumer?" → Remove unless question asks about ownership
- "Am I querying metadata/config (@jsonField)?" → Remove unless question asks about it
- "Am I querying totalAdded/totalRemoved?" → Remove unless question asks about individual amounts
- "Do I really need ALL fields in this nested object?" → If NO, only query what you need!

🚫 Stop Condition Rules (CRITICAL - CHECK RESULTS FIRST)
⚠️ MANDATORY: ALWAYS check if current query results already contain the answer BEFORE making another query!

1. After EVERY query execution, IMMEDIATELY analyze if the returned data is sufficient to answer the question
2. If the data contains the answer → STOP and provide the final answer. DO NOT query again.
3. Only make a second query if the first result is genuinely missing required data
4. NEVER re-query the same entity just to "verify" or "get more details" if the first result already has the data
5. NEVER re-query the same entity with different pagination (first: 5 vs first: 50) - use what you got first!

❌ ABSOLUTELY FORBIDDEN Patterns:

❌ Pattern 1: Re-querying with data already available
# First query returns full data:
{{ indexers(first: 1) {{ nodes {{ id: "0xABC", totalStake: 1000, delegations: [...] }} }} }}
# Then WRONGLY queries again:
{{ indexer(id: "0xABC") {{ totalStake delegations {{ ... }} }} }}  ← FORBIDDEN! Data already available!

❌ Pattern 2: Re-querying same entity with different pagination (CRITICAL VIOLATION)
# First query:
{{ deployments(first: 5, orderBy: AMOUNT_DESC) {{ nodes {{ id amount }} }} }}
# Then WRONGLY queries with larger first value:
{{ deployments(first: 50, orderBy: AMOUNT_DESC) {{ nodes {{ id amount }} }} }}  ← FORBIDDEN! Use first result!
# Then WRONGLY queries again with first: 1:
{{ deployments(first: 1, orderBy: AMOUNT_DESC) {{ nodes {{ id amount }} }} }}  ← FORBIDDEN! Already got it!

❌ Pattern 3: "Trying different parameters to get better results"
This is FORBIDDEN. Your first query should be correct. Do not experiment with queries.

❌ Pattern 4: Incrementally adjusting filter conditions (CRITICAL VIOLATION)
# First query returns empty or insufficient results:
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 85 }} }}) {{ nodes {{ id }} }} }}
# Then WRONGLY tries slightly different filter:
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 84 }} }}) {{ nodes {{ id }} }} }}
# Then tries AGAIN with another adjustment:
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 83 }} }}) {{ nodes {{ id }} }} }}
← FORBIDDEN! Do not "trial and error" with filters. Use a broader range from the start.

# Even WORSE - Random illogical attempts:
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: -10 }} }}) {{ nodes {{ id }} }} }}  ← WTH?
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 75 }} }}) {{ nodes {{ id }} }} }}
{{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 70 }} }}) {{ nodes {{ id }} }} }}
← ABSOLUTELY FORBIDDEN! This is random guessing with no logic!

⚠️ If first query returns empty → STOP and THINK:
1. "What is the typical range for this field?" (e.g., era numbers are usually 0-200)
2. "What filter would logically capture the data I need?"
3. Make ONE query with a reasonable, broader range (e.g., >= 50 or >= 0)
4. DO NOT randomly try different values hoping something works!

✅ CORRECT: Think before querying
If looking for recent indexers by commission era:
- First understand: "Current era is ~100, so >= 80 should cover recent ones"
- Query ONCE with: {{ indexers(first: 100, filter: {{ commissionEra: {{ greaterThanOrEqualTo: 80 }} }}) }}
- If empty, adjust ONCE to broader: >= 50 or remove filter entirely

❌ Pattern 5: Querying same collection separately for nodes and aggregates (CRITICAL VIOLATION)
# First query gets nodes:
{{ deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{ nodes {{ id totalAmount }} }} }}
# Then WRONGLY queries SAME collection for aggregates:
{{ deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{ groupedAggregates(groupBy: [DEPLOYMENT_ID]) {{ keys sum {{ totalAmount }} }} }} }}
← FORBIDDEN! You can query nodes AND aggregates in ONE query!

✅ CORRECT: Query nodes and aggregates together
{{ deploymentBoosterSummaries(first: 5, orderBy: TOTAL_AMOUNT_DESC) {{
    nodes {{ id totalAmount consumer }}
    groupedAggregates(groupBy: [DEPLOYMENT_ID]) {{ keys sum {{ totalAmount }} }}
}} }}
← Both in ONE query! No need for second query!

✅ CORRECT Pattern - Use the data you already have:
# First query returns sufficient data:
{{ deployments(first: 5, orderBy: AMOUNT_DESC) {{ nodes {{ id: "0x1", amount: 1000 }} }} }}

# Question: "Which deployment has highest amount?"
# Answer: The first node is the answer (orderBy: DESC means first is highest)
# Then IMMEDIATELY provide answer:
"The deployment with highest amount is 0x1 with 1000"  ← No second query needed!

🔍 Self-check before making ANY additional query:
- "Does the first query result already contain this data?" → If YES, STOP
- "Am I re-querying the same entity with different pagination?" → If YES, FORBIDDEN
- "Am I trying to 'get more results' when first result already answers the question?" → If YES, STOP
- "Did I use orderBy correctly so the first result is already the answer?" → If YES, use it!
- "Can I query nodes AND aggregates together in ONE query?" → If YES, combine them!

⚠️ Critical reminders:
- Query results contain FULL object data, not just IDs
- If you used orderBy: DESC, the FIRST result is the highest/maximum
- first: 5 is enough for most questions - don't re-query with first: 50
- Use what you already have!

🚨 CRITICAL BLOCK HEIGHT RULE:
- CURRENT BLOCK HEIGHT: ##{block_height}##
- IF CURRENT BLOCK HEIGHT is NOT 0 (non-zero value):
  * ALL GraphQL queries MUST include the blockHeight parameter
  * Set blockHeight to the CURRENT BLOCK HEIGHT value
  * This ensures queries return data at the specified block state
  
  ✅ CORRECT (when CURRENT BLOCK HEIGHT = 5460865):
  {{
    indexers(first: 5, blockHeight: "5460865") {{ nodes {{ id totalStake }} }}
  }}
  
  {{
    indexer(id: "0x123", blockHeight: "5460865") {{ id totalStake }}
  }}
  
  {{
    deployments(first: 5, orderBy: AMOUNT_DESC, blockHeight: "5460865") {{ nodes {{ id amount }} }}
  }}
  
  ❌ WRONG (missing blockHeight when CURRENT BLOCK HEIGHT is non-zero):
  {{
    indexers(first: 5) {{ nodes {{ id totalStake }} }}
  }}
  
- IF CURRENT BLOCK HEIGHT is 0:
  * Do NOT add blockHeight parameter to queries
  * Query normally without blockHeight

  
💡 NOW USE THE RAW SCHEMA ABOVE TO:
1. Find @entity types (e.g., Project, Indexer) 
2. Infer queries: project(id), projects(filter/pagination)
3. Identify field types to determine foreign key relationships
4. Construct your GraphQL query using the patterns above
5. Validate the query, then execute it

DO NOT call graphql_schema_info again - everything needed is above."""
    
    def _generate_thegraph_info(self, schema_content: str) -> str:
        """Generate The Graph-specific schema information."""
        return create_thegraph_schema_info_content(schema_content)

def create_system_prompt(
    domain_name: str,
    domain_capabilities: list,
    decline_message: str,
    is_synthetic: bool = False,
) -> str:
    """
    Create a system prompt for langgraph GraphQL agent.

    Args:
        domain_name: Name of the domain/project (e.g., "SubQuery Network", "DeFi Protocol")
        domain_capabilities: List of capabilities/data types the agent can help with
        decline_message: Custom message when declining out-of-scope requests
        is_synthetic: Whether this is a synthetic challenge (affects domain filtering behavior)

    Returns:
        str: System prompt for langgraph agent
    """
    capabilities_text = '\n'.join([f"- {cap}" for cap in domain_capabilities])
    
    if is_synthetic:
        # For synthetic challenges, always attempt to answer without domain limitations
        return f"""You are a GraphQL assistant helping with data queries for {domain_name}. You can help users find information about:
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

WORKFLOW:
1. Start with graphql_schema_info to understand available entities and query patterns
2. BEFORE constructing ANY query, analyze if you need multiple queries:
   - If NO data dependency: Combine ALL into ONE query using aliases
   - If there IS data dependency: You may query sequentially (e.g., get ID first, then query details)
3. Construct your GraphQL query(ies) to fetch needed data
4. Validate and Execute with graphql_query_validator_execute
5. ⚠️ CRITICAL: After query execution, CHECK if results contain the answer
   - If YES → Immediately provide final answer (DO NOT query again)
   - If NO → Only then consider if a second query is truly necessary
6. Provide clear, user-friendly summaries of the results

⚠️ CRITICAL RULES - TOOL CALL LIMIT:
- You can call graphql_query_validator_execute AT MOST 2 times per question
- Call Counter: Count how many times you've called this tool already
- Before 2nd call: Ask yourself "Is this really necessary? Does my first result already answer the question?"
- If you used orderBy DESC: The FIRST result is the maximum/highest - DO NOT query again with first:1
- NEVER change pagination (first:5 → first:1) to "verify" - that's a waste!
- NEVER incrementally adjust filter conditions (>= 85 → >= 84 → >= 83) - use broader range from start!
- NEVER randomly try different filter values (>= -10 → >= 75 → >= 70) - think about logic first!
- If first query returns empty/insufficient → Analyze WHY, then make ONE logical adjusted query
- Think: "What filter range would logically cover the data I need?" before querying
- If queries have NO data dependency → MUST combine into ONE query
- If second query needs result from first → You MAY query twice (but minimize this)
- NEVER query the same entity twice for different fields that could be fetched together
- ALWAYS check if first query results already answer the question before querying again

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data."""
    else:
        # For organic queries, use domain-based filtering
        return f"""You are a GraphQL assistant specialized in {domain_name} data queries. You can help users find information about:
{capabilities_text}

RESPONSE STYLE: Provide complete, definitive responses. Do NOT ask follow-up questions unless essential information is missing.

WORKFLOW:

IF NOT RELATED to {domain_name}:
- Politely decline with: "{decline_message}"

IF RELATED to {domain_name} data:
1. Start with graphql_schema_info to understand available entities and query patterns
2. Construct proper GraphQL queries based on the schema
3. Execute queries with graphql_query_validator_execute
4. Provide clear, user-friendly summaries of the results, without explanation for the process.

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data."""


a = '''
The GraphQL query should only include fields that are directly relevant to the user's question. Unrelated fields Must not be queried.
'''
class GraphQLTypeDetailInput(BaseModel):
    """Input for GraphQL type detail tool."""
    type_name: str = Field(description="Name of the GraphQL type to examine")


class GraphQLTypeDetailTool(BaseTool):
    """
    Tool to get type definition for a specific GraphQL type.
    Use only as fallback when validation fails - prefer using raw schema from graphql_schema_info.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "graphql_type_detail"
    description: str = """
    Get type definition for a specific GraphQL type (depth=0 only to minimize tokens).
    
    IMPORTANT: Only use this tool as a FALLBACK when query validation fails and you need
    to check specific type definitions. Prefer using the raw schema from graphql_schema_info.
    
    Input: type_name (string) - exact type name to examine
    Example: "IndexerConnection", "Project", "EraRewardFilter"
    """
    args_schema: Type[BaseModel] = GraphQLTypeDetailInput
    
    def __init__(self, graphql_source):
        super().__init__()
        self._graphql_source = graphql_source
        self._schema_data_cache = None
    
    @property
    def graphql_source(self):
        return self._graphql_source
    
    def _run(
        self,
        type_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get GraphQL type detail synchronously."""
        return asyncio.run(self._arun(type_name))
    
    async def _arun(
        self,
        type_name: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Get type definition for a specific GraphQL type."""
        try:
            # Use cached schema_data if available
            if self._schema_data_cache is None:
                self._schema_data_cache = await self.graphql_source.get_schema_data()
            schema_data = self._schema_data_cache
            
            # Use depth=0 to minimize token consumption
            result = process_graphql_schema(schema_data, filter=type_name, depth=0)
            
            if "not found" in result.lower():
                return f"Type '{type_name}' not found in schema. Check type name spelling or use graphql_schema_info to see available types."
            
            return f"""Type definition for '{type_name}' (depth=0 for minimal tokens):

{result}

💡 This is a fallback tool - prefer using raw schema from graphql_schema_info for better context."""
            
        except Exception as e:
            return f"Error getting type detail: {str(e)}"


class GraphQLQueryValidatorInput(BaseModel):
    """Input for GraphQL query validator tool."""
    query: str = Field(description="GraphQL query string to validate")


class GraphQLQueryValidatorTool(BaseTool):
    """
    Tool to validate GraphQL query syntax and structure.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "graphql_query_validator"
    description: str = """
    Validate a GraphQL query string for syntax and basic structure.
    Input: Pass the GraphQL query as plain text without any formatting.
    
    CORRECT: { indexers(first: 1) { nodes { id } } }
    WRONG: `{ indexers(first: 1) { nodes { id } } }`
    WRONG: ```{ indexers(first: 1) { nodes { id } } }```
    
    The tool will automatically clean code blocks, backticks, and quotes.
    """
    args_schema: Type[BaseModel] = GraphQLQueryValidatorInput
    
    def __init__(self, graphql_source):
        super().__init__()
        self._graphql_source = graphql_source
    
    @property
    def graphql_source(self):
        return self._graphql_source
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Validate GraphQL query synchronously."""
        return asyncio.run(self._arun(query))
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Validate GraphQL query against schema."""
        try:
            # Clean up common formatting issues first
            query = query.strip()
            
            # Remove code block markers (```...```)
            if query.startswith('```') and query.endswith('```'):
                query = query[3:-3].strip()
                # Also remove language identifier if present (e.g., ```graphql)
                lines = query.split('\n')
                if lines and lines[0].strip() and not lines[0].strip().startswith('{'):
                    query = '\n'.join(lines[1:]).strip()
            
            # Remove single backticks if present
            if query.startswith('`') and query.endswith('`'):
                query = query[1:-1].strip()
            
            # Remove quotes if present
            if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
                query = query[1:-1].strip()
            
            # Basic syntax validation
            validation_errors = []
            
            # Check for basic GraphQL structure
            if not query:
                return "❌ Validation failed: Empty query"
            
            # Check for balanced braces
            open_braces = query.count('{')
            close_braces = query.count('}')
            if open_braces != close_braces:
                validation_errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
            
            # Check for balanced parentheses
            open_parens = query.count('(')
            close_parens = query.count(')')
            if open_parens != close_parens:
                validation_errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
            
            # Early return if basic syntax errors found
            if validation_errors:
                return f"❌ Basic syntax validation failed:\n" + "\n".join([f"- {error}" for error in validation_errors])
            
            # Advanced validation with GraphQL parser and schema
            try:
                # Parse the query
                document = graphql.parse(query)
                
                # Get complete introspection result for proper validation
                introspection_result = await self.graphql_source.get_schema()
                
                # Build GraphQL schema from introspection data (use data part only)
                schema_data = introspection_result.get('data', None)
                if not schema_data:
                    return "❌ Schema validation failed: No data in introspection result"
                
                schema = build_client_schema(schema_data)
                
                # Use graphql-core's built-in validation
                validation_errors = validate(schema, document)

                from loguru import logger

                if validation_errors:
                    logger.info(f"============================validation error for query: {query}, {validation_errors}")

                    error_messages = [error.message for error in validation_errors]
                    return f"❌ Schema validation failed:\n" + "\n".join([f"- {error}" for error in error_messages])
                else:
                    return f"✅ Query is valid and matches schema:\n\n{query}"
                    
            except Exception as parse_error:
                return f"❌ Query parsing failed: {str(parse_error)}"
            
        except Exception as e:
            return f"Error validating query: {str(e)}"

class GraphQLQueryValidatorAndExecutedTool(BaseTool):
    """
    Tool to validate GraphQL query syntax and structure and execute GraphQL queries.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "graphql_query_validator_execute"
    description: str = """
    Validate a GraphQL query string for syntax and basic structure and execute it if valid.
    Input: Pass the GraphQL query as plain text without any formatting.

    🛑 CRITICAL RULES BEFORE USING THIS TOOL:
    1. After calling this tool, CHECK the result immediately
    2. If the result contains the answer to the question → DO NOT call this tool again
    3. If you used orderBy with DESC → The FIRST result is the maximum/highest value
    4. DO NOT call this tool again with different 'first' values (first: 5 → first: 1) - use what you got!
    5. DO NOT call this tool to "verify" or "double-check" - trust your first query result
    6. DO NOT incrementally adjust filters (>= 85 → >= 84 → >= 83) - use broader range from start!
    7. DO NOT randomly try filter values (>= -10 → >= 75 → >= 70) - THINK about logic first!
    8. If first query is empty → STOP, THINK about reasonable filter range, then query ONCE more
    9. DO NOT query same collection separately for nodes and aggregates - combine in ONE query!

    Example of CORRECT usage:
    Query: "Which deployment has highest amount?"
    1. Call tool with: deployments(first: 5, orderBy: AMOUNT_DESC) { nodes { id amount } }
    2. Result: [{ id: "0x1", amount: 1000 }, ...]
    3. Answer immediately: "Deployment 0x1 has the highest amount of 1000"
    4. DO NOT call tool again! ← Important!

    Example of WRONG usage (DO NOT DO THIS):
    1. Call tool with: deployments(first: 5, orderBy: DESC) → Get result
    2. Call tool again with: deployments(first: 1, orderBy: DESC) ← WRONG! Already got answer!
    
    3. Or incrementally adjust filters (FORBIDDEN):
       - First: indexers(filter: { commissionEra >= 85 }) → Empty
       - Then: indexers(filter: { commissionEra >= 84 }) ← WRONG!
       - Then: indexers(filter: { commissionEra >= 83 }) ← WRONG!
       
    4. Or randomly try values (ABSOLUTELY FORBIDDEN):
       - First: indexers(filter: { commissionEra >= -10 }) → Too broad/nonsensical
       - Then: indexers(filter: { commissionEra >= 75 }) ← Random guess!
       - Then: indexers(filter: { commissionEra >= 70 }) ← Still guessing!
       
    5. Or query same collection separately for nodes and aggregates (FORBIDDEN):
       - First: deploymentBoosterSummaries(first: 5) { nodes { id } }
       - Then: deploymentBoosterSummaries(first: 5) { groupedAggregates { ... } }
       ← WRONG! Query nodes AND aggregates together in ONE query!
       
    ✅ CORRECT: Think first, then query with reasonable range:
       indexers(filter: { commissionEra >= 50 }) or remove filter if unsure
    
    ✅ CORRECT: Query nodes and aggregates together:
       deploymentBoosterSummaries(first: 5) { nodes { id } groupedAggregates { ... } }

    CORRECT: { indexers(first: 1) { nodes { id } } }
    WRONG: `{ indexers(first: 1) { nodes { id } } }`
    WRONG: ```{ indexers(first: 1) { nodes { id } } }```
    
    The tool will automatically clean code blocks, backticks, and quotes.
    """
    args_schema: Type[BaseModel] = GraphQLQueryValidatorInput
    
    def __init__(self, graphql_source):
        super().__init__()
        self._graphql_source = graphql_source
    
    @property
    def graphql_source(self):
        return self._graphql_source
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Validate GraphQL query synchronously."""
        return asyncio.run(self._arun(query))
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Validate GraphQL query against schema."""
        try:
            # Clean up common formatting issues first
            query = query.strip()
            
            # Remove code block markers (```...```)
            if query.startswith('```') and query.endswith('```'):
                query = query[3:-3].strip()
                # Also remove language identifier if present (e.g., ```graphql)
                lines = query.split('\n')
                if lines and lines[0].strip() and not lines[0].strip().startswith('{'):
                    query = '\n'.join(lines[1:]).strip()
            
            # Remove single backticks if present
            if query.startswith('`') and query.endswith('`'):
                query = query[1:-1].strip()
            
            # Remove quotes if present
            if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
                query = query[1:-1].strip()
            
            # Basic syntax validation
            validation_errors = []
            
            # Check for basic GraphQL structure
            if not query:
                return "❌ Validation failed: Empty query"
            
            # Check for balanced braces
            open_braces = query.count('{')
            close_braces = query.count('}')
            if open_braces != close_braces:
                validation_errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
            
            # Check for balanced parentheses
            open_parens = query.count('(')
            close_parens = query.count(')')
            if open_parens != close_parens:
                validation_errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
            
            # Early return if basic syntax errors found
            if validation_errors:
                return f"❌ Basic syntax validation failed:\n" + "\n".join([f"- {error}" for error in validation_errors])
            
            # Advanced validation with GraphQL parser and schema
            try:
                # Parse the query
                document = graphql.parse(query)
                
                # Get complete introspection result for proper validation
                introspection_result = await self.graphql_source.get_schema()
                
                # Build GraphQL schema from introspection data (use data part only)
                schema_data = introspection_result.get('data', None)
                if not schema_data:
                    return "❌ Schema validation failed: No data in introspection result"
                
                schema = build_client_schema(schema_data)
                
                # Use graphql-core's built-in validation
                validation_errors = validate(schema, document)

                if validation_errors:
                    logger.info(f"============================validation error for query: {query}, {validation_errors}")
                    error_messages = [error.message for error in validation_errors]
                    return f"❌ Schema validation failed:\n" + "\n".join([f"- {error}" for error in error_messages])
                else:
                    return await self._execute(query)
                    
            except Exception as parse_error:
                return f"❌ Query parsing failed: {str(parse_error)}"
            
        except Exception as e:
            return f"Error validating query: {str(e)}"
    
    async def _execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> str:
        try:
            result = await self.graphql_source.execute_query(query, variables)
            if "errors" in result:
                errors = result["errors"]
                error_messages = [error.get("message", str(error)) for error in errors]
                return f"❌ Query execution failed:\n" + "\n".join([f"- {msg}" for msg in error_messages])
            
            if "data" in result:
                data = result["data"]
                formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
                return f"✅ Query executed successfully:\n\n{formatted_data}"
            
            return f"⚠️ Unexpected response format:\n{json.dumps(result, indent=2)}"
            
        except Exception as e:
            return f"Error executing query: {str(e)}"


class GraphQLExecuteInput(BaseModel):
    """Input for GraphQL execute tool."""
    query: str = Field(description="GraphQL query string to execute")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Optional query variables as JSON object")


class GraphQLExecuteTool(BaseTool):
    """
    Tool to execute GraphQL queries.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = "graphql_execute"
    description: str = """
    Execute a GraphQL query against the API endpoint.
    Input: GraphQL query as plain text without any formatting markers.
    
    CORRECT: { indexers(first: 2) { nodes { id } } }
    WRONG: 
    - `{ indexers... }` (with backticks)
    - ```{ indexers... }``` (with code blocks)
    - "{ indexers... }" (with quotes)  
    - {"query": "{ indexers... }"} (JSON wrapped)
    
    The tool will automatically clean formatting issues.
    """
    args_schema: Type[BaseModel] = GraphQLExecuteInput
    
    def __init__(self, graphql_source):
        super().__init__()
        self._graphql_source = graphql_source
    
    @property
    def graphql_source(self):
        return self._graphql_source
    
    def _run(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute GraphQL query synchronously."""
        return asyncio.run(self._arun(query, variables))
    
    async def _arun(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute GraphQL query."""
        try:
            # Clean up common formatting issues
            query = query.strip()
            
            # Remove code block markers (```...```)
            if query.startswith('```') and query.endswith('```'):
                query = query[3:-3].strip()
                # Also remove language identifier if present (e.g., ```graphql)
                lines = query.split('\n')
                if lines and lines[0].strip() and not lines[0].strip().startswith('{'):
                    query = '\n'.join(lines[1:]).strip()
            
            # Remove single backticks if present
            if query.startswith('`') and query.endswith('`'):
                query = query[1:-1].strip()
            
            # Remove quotes if present
            if (query.startswith('"') and query.endswith('"')) or (query.startswith("'") and query.endswith("'")):
                query = query[1:-1].strip()
            
            result = await self.graphql_source.execute_query(query, variables)
            
            if "errors" in result:
                errors = result["errors"]
                error_messages = [error.get("message", str(error)) for error in errors]
                return f"❌ Query execution failed:\n" + "\n".join([f"- {msg}" for msg in error_messages])
            
            if "data" in result:
                data = result["data"]
                formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
                return f"✅ Query executed successfully:\n\n{formatted_data}"
            
            return f"⚠️ Unexpected response format:\n{json.dumps(result, indent=2)}"
            
        except Exception as e:
            return f"Error executing query: {str(e)}"