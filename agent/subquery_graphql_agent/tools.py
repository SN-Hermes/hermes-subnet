"""GraphQL Tools for LLM agents."""

import json
import asyncio
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

import graphql
from graphql import build_client_schema, validate, build_schema

from langchain_core.tools import BaseTool
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get GraphQL schema info synchronously."""
        return asyncio.run(self._arun())
    
    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Get raw GraphQL schema with automatic node type detection and appropriate guidance."""
        try:
            # Get raw schema from GraphQL source
            schema_content = self.graphql_source.entity_schema
            
            # Generate appropriate schema info based on node type
            if self._node_type == GraphqlProvider.THE_GRAPH:
                return create_thegraph_schema_info_content(schema_content)
            elif self._node_type == GraphqlProvider.SUBQL:
                return self._generate_subql_info(schema_content)
            
        except Exception as e:
            return f"Error reading schema info: {str(e)}"
    
    def _generate_subql_info(self, schema_content: str) -> str:
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
1. ALWAYS validate queries with graphql_query_validator before executing
2. For missing user info ("my rewards"), ASK for wallet/ID - NEVER fabricate data
3. Pass queries to graphql_execute as plain text (no backticks/quotes)
4. Only use graphql_type_detail as FALLBACK when validation fails - prefer raw schema

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
    decline_message: str
) -> str:
    """
    Create a system prompt for langgraph GraphQL agent.
    
    Args:
        domain_name: Name of the domain/project (e.g., "SubQuery Network", "DeFi Protocol")
        domain_capabilities: List of capabilities/data types the agent can help with
        decline_message: Custom message when declining out-of-scope requests
        
    Returns:
        str: System prompt for langgraph agent
    """
    capabilities_text = '\n'.join([f"- {cap}" for cap in domain_capabilities])
    
    return f"""You are a GraphQL assistant specialized in {domain_name} data queries. You can help users find information about:
{capabilities_text}

RESPONSE STYLE: Provide complete, definitive responses. Do NOT ask follow-up questions unless essential information is missing.

WORKFLOW:

IF NOT RELATED to {domain_name}:
- Politely decline with: "{decline_message}"

IF RELATED to {domain_name} data:
1. Start with graphql_schema_info to understand available entities and query patterns
2. Construct proper GraphQL queries based on the schema
3. Validate queries with graphql_query_validator before execution
4. Execute queries with graphql_execute
5. Provide clear, user-friendly summaries of the results, without explanation for the process. 

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data."""



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
                
                if validation_errors:
                    error_messages = [error.message for error in validation_errors]
                    return f"❌ Schema validation failed:\n" + "\n".join([f"- {error}" for error in error_messages])
                else:
                    return f"✅ Query is valid and matches schema:\n\n{query}"
                    
            except Exception as parse_error:
                return f"❌ Query parsing failed: {str(parse_error)}"
            
        except Exception as e:
            return f"Error validating query: {str(e)}"




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