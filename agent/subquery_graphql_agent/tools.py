"""GraphQL Tools for LLM agents."""

import json
import asyncio
from typing import Optional, Type, Dict, Any, Annotated, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger
import graphql
from graphql import build_client_schema, validate
from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

if TYPE_CHECKING:
    from agent.subquery_graphql_agent.base import GraphQLSource

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
    
    @property
    def postgraphile_rules(self) -> str:
        if self._node_type == GraphqlProvider.THE_GRAPH:
            return """
📋 SUBGRAPH INFERENCE RULES:
- Each @entity type → database table with 2 queries: singular(id) & plural(filter/pagination)
- Fields with @derivedFrom → relationship fields, need subfield selection
- Foreign key fields is not accessible directly, must use relationship field
- System tables (_meta) → ignore these

📖 SUBGRAPH QUERY PATTERNS:
1. 📊 ENTITY QUERIES:
   - Single query: entityName(id: ID!,subgraphError: _SubgraphErrorPolicy_! = deny) → EntityType
   - Collection query: entityNames(skip: Int, first: Int, where: EntityFilter, orderBy: EntityOrderBy, orderDirection: OrderDirection, subgraphError: _SubgraphErrorPolicy_! = deny) → [EntityType]
   - ⚠️ PLURAL NAMING: If entity ends with 's' (e.g., Series), plural adds 'es' (e.g., serieses). Follow standard English pluralization rules.
   - Multiple queries: You can send multiple independent queries in a single GraphQL request if they have no data dependencies between them

2. 🔗 RELATIONSHIP QUERIES:
   - Direct field access: entity { field { id, otherFields } }
   - Direct array access for one-to-many relationships

3. 📝 FILTER PATTERNS (SubGraph Format - <field>_<op>):
   
   ID FILTERS:
   - Direct field comparisons: id: "0x123"
   - id_not: String! - not equal to
   - id_gt, id_lt, id_gte, id_lte: String! - comparison operators
   - id_in: [ID!] - match any value in list
   - id_not_in: [ID!] - not match any value in list
   
   STRING FILTERS:
   - Direct field comparisons: name: "alice"
   - name_contains, name_contains_nocase: String! - substring matching
   - name_not_contains, name_not_contains_nocase: String! - not contains substring
   - name_starts_with, name_starts_with_nocase: String! - prefix matching
   - name_not_starts_with, name_not_starts_with_nocase: String! - not starts with
   - name_ends_with, name_ends_with_nocase: String! - suffix matching
   - name_not_ends_with, name_not_ends_with_nocase: String! - not ends with
   - name_gt, name_lt, name_gte, name_lte: String! - lexicographic comparison
   - name_in: [String!] - match any value in list
   - name_not_in: [String!] - not match any value in list
   - name_not: String! - not equal to
   
   BYTES FILTERS (Ethereum addresses, hashes, etc.):
   - Direct field comparisons: name: "0x1234..." (full hex string with 0x prefix)
   - name_not: Bytes! - not equal to
   - name_gt, name_lt, name_gte, name_lte: Bytes! - byte-order comparison
   - name_in: [Bytes!] - match any value in list
   - name_not_in: [Bytes!] - not match any value in list
   - name_contains: Bytes! - contains byte sequence (hex substring)
   - name_not_contains: Bytes! - does not contain byte sequence
   
   NUMBER FILTERS (Int, BigInt, BigDecimal):
   - Direct field comparisons: amount: "100"
   - amount_gt, amount_gte, amount_lt, amount_lte: String! - numeric comparisons (values as strings)
   - amount_in: [String!] - match any value in list (BigInt/BigDecimal as strings)
   - amount_not_in: [String!] - not match any value in list
   - amount_not: String! - not equal to
   
   BOOLEAN FILTERS:
   - Direct field comparisons: active: true
   - active_not: Boolean! - not equal to
   - active_in: [Boolean!] - match any value in list
   - active_not_in: [Boolean!] - not match any value in list
   
   NESTED FILTERS (AND/OR Logic):
   - and: [EntityFilter!] - all conditions must be true
   - or: [EntityFilter!] - at least one condition must be true
   - Can be nested arbitrarily deep for complex logic
   
   EXAMPLES:
   - { id: "0x123" } - direct ID match
   - { id_in: ["0x123", "0x456"] } - ID in list
   - { status_in: ["active", "pending"] } - string in list
   - { amount_gt: "100" } - BigInt greater than
   - { name_contains_nocase: "alice" } - case-insensitive substring
   - { symbol_starts_with: "UNI" } - prefix matching
   - { balance_gte: "1000000000000000000" } - BigInt >= 1 ETH
   - { and: [{ active: true }, { balance_gt: "0" }] } - AND logic
   - { or: [{ symbol: "ETH" }, { symbol: "BTC" }] } - OR logic

4. 📈 ORDER BY PATTERNS:
   - orderBy: field_name (camelCase field names)
   - orderDirection: asc | desc
   - Examples: orderBy: id, orderBy: createdAt, orderBy: amount

5. 📄 PAGINATION:
   - first: Int (limit results)
   - skip: Int (offset results)  
   - No cursor-based pagination (unlike SubQL)

🚨 CRITICAL AGENT RULES:
1. ALWAYS validate queries with graphql_query_validator before executing
2. For missing user info ("my tokens", "my positions"), ASK for address - NEVER fabricate data
3. Pass queries to graphql_execute as plain text (no backticks/quotes)

⚠️ CRITICAL THE GRAPH ENTITY RULES:
- Entity fields are accessed directly without @derivedFrom complexity
- No "nodes" wrapper for collections (unlike SubQL)
- Use direct field access: entity { relatedField { id, otherField } }
- Collections return arrays directly: entities { field }

⚠️ CRITICAL SCALAR RULES:
- ID fields are strings, not integers: "0x123abc"
- Int fields are regular integers: 42
- BigInt fields stored as strings: "12345678901234567890"
- BigDecimal fields stored as strings for precise decimals: "123.456789"
- Bytes for hex-encoded byte arrays: "0x1234abcd"
- All number comparisons in filters use string values for BigInt/BigDecimal

🔍 ENTITY IDENTIFICATION:
- Look at @entity directive to identify entities
- Field types determine relationships - no @derivedFrom needed
- Direct field references indicate relationships
- Example: user: User → Look for @entity User, query user { id, address }

📝 TYPE MAPPING EXAMPLES (The Graph):
- user: User → Find @entity User, query user { id, address }
- token: Token → Find @entity Token, query token { id, symbol, decimals }
- id: ID → Query as string: "0x123abc"
- count: Int → Query as integer: 42
- amount: BigInt → Query as string: "1000000000000000000" (1 ETH in wei)
- price: BigDecimal → Query as string: "1234.567890123456789"
- timestamp: BigInt → Query as string: "1640995200"  
- data: Bytes → Query as hex string: "0x1234abcd"
- active: Boolean → Query as boolean: true/false

📋 RELATIONSHIP QUERY EXAMPLES:
✅ { user(id: "0x123") { id, tokens { id, symbol, balance } } }
✅ { tokens { id, symbol, holder { id, address } } }
✅ { transfers(first: 10) { id, from { address }, to { address }, amount } }
❌ { tokens { nodes { id, symbol } } } (no "nodes" wrapper needed)

📊 FILTERING QUERY EXAMPLES:
✅ { users(where: { balance_gt: "1000" }) { id, address, balance } }
✅ { transfers(where: { amount_gte: "100", token: "0x123" }) { id, amount } }
✅ { tokens(where: { symbol_in: ["ETH", "BTC"] }) { id, symbol } }
✅ { tokens(where: { name_contains_nocase: "uniswap" }) { id, name, symbol } }
✅ { users(where: { id_not_in: ["0x123", "0x456"] }) { id, address } }
✅ { pairs(where: { and: [{ token0: "0x123" }, { reserve0_gt: "1000" }] }) { id, token0, token1 } }
✅ { swaps(where: { or: [{ amount0_gt: "100" }, { amount1_gt: "100" }] }) { id, amount0, amount1 } }
✅ { tokens(where: { symbol_starts_with_nocase: "uni" }) { id, symbol, name } }
✅ { positions(where: { owner_not: "0x0000", liquidity_gt: "0" }) { id, owner, liquidity } }
"""
        elif self._node_type == GraphqlProvider.SUBQL:
            return """
📋 POSTGRAPHILE v4 INFERENCE RULES:

- Each @entity type → database table with 2 queries: singular(id) & plural(filter/pagination)
- Fields with @derivedFrom → relationship fields, need subfield selection
- Foreign key fields ending in 'Id' → direct ID access
- System tables (_pois, _metadatas, _metadata) → ignore these

📖 POSTGRAPHILE v4 QUERY PATTERNS:
1. 📊 ENTITY QUERIES:
   - Single query: entityName(id: ID!) → EntityType
   - Collection query: entityNames(first: Int, filter: EntityFilter, orderBy: [EntityOrderBy!]) → EntityConnection
   - ⚠️ PLURAL NAMING: If entity ends with 's' (e.g., Series), plural adds 'es' (e.g., serieses). Follow standard English pluralization rules.
   - Multiple queries: You can send multiple independent queries in a single GraphQL request if they have no data dependencies between them

2. 🔗 RELATIONSHIP QUERIES:
   - Foreign key ID: fieldNameId (returns ID directly)
   - Single entity: fieldName { id, otherFields }
   - Collection relationships: fieldName { nodes { id, otherFields }, pageInfo { hasNextPage, endCursor }, totalCount }
   - With filters: fieldName(filter: { ... }) { nodes { ... }, totalCount }

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
   - { id: { equalTo: "0x123" } }
   - { status: { in: ["active", "pending"] } }
   - { count: { greaterThan: 100 } }
   - { name: { equalToInsensitive: "alice" } }

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
   - aggregates { sum { fieldName }, distinctCount { fieldName }, min { fieldName }, max { fieldName } }
   - aggregates { average { fieldName }, stddevSample { fieldName }, stddevPopulation { fieldName } }
   - aggregates { varianceSample { fieldName }, variancePopulation { fieldName }, keys }

   GROUPED AGGREGATES (group by):
   - groupedAggregates(groupBy: [FIELD_NAME], having: { ... }) { keys, sum { fieldName } }
   - groupBy: Required, uses UPPER_SNAKE_CASE format (same as orderBy)
   - having: Optional, uses same filter format as main query
   
   EXAMPLES:
   - { indexers { aggregates { sum { totalReward }, distinctCount { projectId } } } }
   - { indexers { groupedAggregates(groupBy: [PROJECT_ID]) { keys, sum { totalReward } } } }

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
- Example: metadata @jsonField → Use metadata NOT metadata { subfields }
- @jsonField fields return raw JSON data, treat as scalar values

🔍 FOREIGN KEY IDENTIFICATION:
- Look at field TYPE, not field name, to determine relationship
- If field type is @entity → it's a foreign key relationship
  - Physical storage: <fieldName>Id exists and can be used in filters
  - Query usage: fieldName { subfields } for object, fieldNameId for ID
  - Entity lookup: Use the TYPE name to find the @entity definition
- If field type is basic type/enum/@jsonField → NOT a foreign key
  - Query directly: fieldName (no subfield selection needed)
  - For @jsonField: Query as scalar, DO NOT expand subfields

⚠️ CRITICAL: Field type determines entity, NOT field name
- Field: project: Project → Look for @entity Project (not @entity project)
- Field: owner: Account → Look for @entity Account (not @entity owner)

📝 TYPE MAPPING EXAMPLES:
- project: Project → Find @entity Project, query project { id, owner } or projectId
- owner: Account → Find @entity Account, query owner { id, address } or ownerId
- delegator: Delegator → Find @entity Delegator, query delegator { id, amount }
- status: String → Basic type: use status directly
- metadata: JSON @jsonField → Query metadata directly (NOT metadata { subfields })
- type: IndexerType → Enum: use type directly

🎯 REMEMBER: Field name ≠ Entity name. Use TYPE to find the @entity definition!

📋 RELATIONSHIP QUERY EXAMPLES:
✅ { indexer(id: "0x123") { id, project { id, owner } } }
✅ { project(id: "0x456") { id, indexers { nodes { id, status }, totalCount } } }
✅ { indexers { nodes { id, projectId, project { id, owner } } } }
❌ { project { indexers { id, status } } } (missing nodes wrapper)

📋 @jsonField QUERY EXAMPLES:
✅ { project(id: "0x123") { id, metadata, config } } (query @jsonField directly)
✅ { indexers { nodes { id, metadata, settings } } } (@jsonField as scalar)
❌ { project { metadata { name, description } } } (@jsonField cannot be expanded)
❌ { indexer { config { threshold, timeout } } } (@jsonField cannot have subfields)

📊 AGGREGATION QUERY EXAMPLES:
✅ { indexers { aggregates { sum { totalReward }, distinctCount { projectId } } } }
✅ { projects { aggregates { average { totalBoost }, max { totalReward } } } }
✅ { indexers { groupedAggregates(groupBy: [PROJECT_ID]) { keys, sum { totalReward }, distinctCount { id } } } }
✅ { rewards { groupedAggregates(groupBy: [ERA, INDEXER_ID], having: { era: { greaterThan: 100 } }) { keys, sum { amount } } } }
"""

        raise NotImplementedError("PostGraphile rules not implemented for this node type.")

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

            # Get raw schema from GraphQL source
            schema_content = self.graphql_source.entity_schema
            
            # Generate appropriate schema info based on node type
            if self._node_type == GraphqlProvider.THE_GRAPH:
                return self._generate_thegraph_info(schema_content)
            elif self._node_type == GraphqlProvider.SUBQL:
                return self._generate_subql_info(schema_content)
            
        except Exception as e:
            return f"Error reading schema info: {str(e)}"

    def _generate_subql_info(self, schema_content: str) -> str:
        """Generate SubQL-specific schema information."""
        return f"""📖 SUBQL (POSTGRAPHILE v4) SCHEMA & RULES:

🔍 RAW ENTITY SCHEMA:
{schema_content}

{self.postgraphile_rules}

🔍 Before querying ANY field, ask yourself:
- "Does the user's question explicitly need this field?" → If NO, don't query it
- "Am I querying createAt/updateAt?" → Remove unless question asks about time
- "Am I querying owner/consumer?" → Remove unless question asks about ownership
- "Am I querying metadata/config (@jsonField)?" → Remove unless question asks about it
- "Am I querying totalAdded/totalRemoved?" → Remove unless question asks about individual amounts
- "Do I really need ALL fields in this nested object?" → If NO, only query what you need!

⚠️ If first query returns empty → STOP and THINK:
1. "What is the typical range for this field?"
2. "What filter would logically capture the data I need?"

🔍 Self-check before making ANY additional query:
- "Does the first query result already contain this data?" → If YES, STOP
- "Am I re-querying the same entity with different pagination?" → If YES, FORBIDDEN
- "Am I trying to 'get more results' when first result already answers the question?" → If YES, STOP
- "Did I use orderBy correctly so the first result is already the answer?" → If YES, use it!
- "Can I query nodes AND aggregates together in ONE query?" → If YES, combine them!

DO NOT call graphql_schema_info again - everything needed is above."""

    def _generate_thegraph_info(self, schema_content: str) -> str:
        """Generate The Graph-specific schema information."""
        return create_thegraph_schema_info_content(schema_content, self.postgraphile_rules)

def create_system_prompt(
    domain_name: str,
    domain_capabilities: list,
    decline_message: str,
    is_synthetic: bool = False,
    extra_instructions: str | None = None,
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
    critical_rules = """
⚠️ CRITICAL RULES - TOOL CALL LIMIT:
- NEVER make verification queries, think thoroughly before you make a query.
- ALWAYS limit the return with first:10 for ALL list queries as well as in the nested queries, unless told otherwise and it is smaller.
- For time-range queries (e.g., last 7 days, 30 days, weeks), ALWAYS limit the number of results using 'first' parameter to prevent excessive data retrieval.
- If first query returns empty/insufficient → Analyze WHY, then make ONE logical adjusted query.
"""

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

{workflow}

{critical_rules}

{extra_instructions if extra_instructions else ""}

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

{critical_rules}

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data.
"""


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

    CORRECT: { indexers(first: 1) { nodes { id } } }
    WRONG: `{ indexers(first: 1) { nodes { id } } }`
    WRONG: ```{ indexers(first: 1) { nodes { id } } }```
    
    The tool will automatically clean code blocks, backticks, and quotes.
    """
    args_schema: Type[BaseModel] = GraphQLQueryValidatorInput
    
    def __init__(self, graphql_source: "GraphQLSource"):
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
            logger.error(f"Error executing query: {str(e)}")
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