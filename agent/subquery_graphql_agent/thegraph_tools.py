"""
The Graph-specific GraphQL tools and schema parsing logic.

Provides specialized tools and prompts for The Graph protocol nodes,
which have different schema patterns compared to SubQL nodes.
"""

def create_thegraph_schema_info_content(schema_content: str) -> str:
    """
    Create The Graph-specific schema information content.
    
    Args:
        schema_content: Raw GraphQL schema string
        block_height: Current block height for time-travel queries
        
    Returns:
        Formatted schema information string for The Graph
    """
    return f"""📖 THE GRAPH PROTOCOL SCHEMA & RULES:

🔍 RAW ENTITY SCHEMA:
{schema_content}

📋 SUBGRAPH INFERENCE RULES:
- Each @entity type → database table with 2 queries: singular(id) & plural(filter/pagination)
- Fields with @derivedFrom → relationship fields, need subfield selection
- Foreign key fields is not accessible directly, must use relationship field
- System tables (_meta) → ignore these

📖 SUBGRAPH QUERY PATTERNS:
1. 📊 ENTITY QUERIES:
   - Single query: entityName(id: ID!,subgraphError: _SubgraphErrorPolicy_! = deny) → EntityType
   - Collection query: entityNames(skip: Int, first: Int, where: EntityFilter, orderBy: EntityOrderBy, orderDirection: OrderDirection, subgraphError: _SubgraphErrorPolicy_! = deny) → [EntityType]
   - Multiple queries: You can send multiple independent queries in a single GraphQL request if they have no data dependencies between them

2. 🔗 RELATIONSHIP QUERIES:
   - Direct field access: entity {{ field {{ id, otherFields }} }}
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
   - {{ id: "0x123" }} - direct ID match
   - {{ id_in: ["0x123", "0x456"] }} - ID in list
   - {{ status_in: ["active", "pending"] }} - string in list
   - {{ amount_gt: "100" }} - BigInt greater than
   - {{ name_contains_nocase: "alice" }} - case-insensitive substring
   - {{ symbol_starts_with: "UNI" }} - prefix matching
   - {{ balance_gte: "1000000000000000000" }} - BigInt >= 1 ETH
   - {{ and: [{{ active: true }}, {{ balance_gt: "0" }}] }} - AND logic
   - {{ or: [{{ symbol: "ETH" }}, {{ symbol: "BTC" }}] }} - OR logic

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
- Use direct field access: entity {{ relatedField {{ id, otherField }} }}
- Collections return arrays directly: entities {{ field }}

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
- Example: user: User → Look for @entity User, query user {{ id, address }}

📝 TYPE MAPPING EXAMPLES (The Graph):
- user: User → Find @entity User, query user {{ id, address }}
- token: Token → Find @entity Token, query token {{ id, symbol, decimals }}
- id: ID → Query as string: "0x123abc"
- count: Int → Query as integer: 42
- amount: BigInt → Query as string: "1000000000000000000" (1 ETH in wei)
- price: BigDecimal → Query as string: "1234.567890123456789"
- timestamp: BigInt → Query as string: "1640995200"  
- data: Bytes → Query as hex string: "0x1234abcd"
- active: Boolean → Query as boolean: true/false

📋 RELATIONSHIP QUERY EXAMPLES:
✅ {{ user(id: "0x123") {{ id, tokens {{ id, symbol, balance }} }} }}
✅ {{ tokens {{ id, symbol, holder {{ id, address }} }} }}
✅ {{ transfers(first: 10) {{ id, from {{ address }}, to {{ address }}, amount }} }}
❌ {{ tokens {{ nodes {{ id, symbol }} }} }} (no "nodes" wrapper needed)

📊 FILTERING QUERY EXAMPLES:  
✅ {{ users(where: {{ balance_gt: "1000" }}) {{ id, address, balance }} }}
✅ {{ transfers(where: {{ amount_gte: "100", token: "0x123" }}) {{ id, amount }} }}
✅ {{ tokens(where: {{ symbol_in: ["ETH", "BTC"] }}) {{ id, symbol }} }}
✅ {{ tokens(where: {{ name_contains_nocase: "uniswap" }}) {{ id, name, symbol }} }}
✅ {{ users(where: {{ id_not_in: ["0x123", "0x456"] }}) {{ id, address }} }}
✅ {{ pairs(where: {{ and: [{{ token0: "0x123" }}, {{ reserve0_gt: "1000" }}] }}) {{ id, token0, token1 }} }}
✅ {{ swaps(where: {{ or: [{{ amount0_gt: "100" }}, {{ amount1_gt: "100" }}] }}) {{ id, amount0, amount1 }} }}
✅ {{ tokens(where: {{ symbol_starts_with_nocase: "uni" }}) {{ id, symbol, name }} }}
✅ {{ positions(where: {{ owner_not: "0x0000", liquidity_gt: "0" }}) {{ id, owner, liquidity }} }}

💡 NOW USE THE RAW SCHEMA ABOVE TO:
1. Find @entity types (e.g., User, Token, Transfer)
2. Construct queries using The Graph patterns
3. Use direct field access for relationships
4. Apply The Graph-specific filtering and pagination
5. Validate the query, then execute it
6. AVOID DUPLICATE QUERIES: Do not generate queries that would retrieve the same data already obtained from previous queries in the same session

DO NOT call graphql_schema_info again - everything needed is above.
"""
