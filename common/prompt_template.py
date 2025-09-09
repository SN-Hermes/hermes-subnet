from langchain.prompts import PromptTemplate


# 问的很泛, 不好. 都是方法论, 没有具体的数值问题
synthetic_challenge_template_V1 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}

CRITICAL CONSTRAINT - MUST AVOID REPETITION:
{recent_questions}

Your task:
1. Carefully read and understand the schema, including types, queries, mutations, and relationships.
2. Generate ONE natural question that a user might ask based on this file.
3. Output only the question, nothing else.
4. Ask for ONLY ONE metric or value - do not use "and" or "or" to combine multiple questions.
5. Do not include explanations, answers, or more than one question.
6. Ask about what CAN be queried, not specific made-up scenarios.
7. Focus on query patterns and schema structure, not fictional data.
8. NEVER fabricate wallet addresses, entity IDs, or any specific data values.
9. ABSOLUTELY DO NOT generate questions that are similar to the ones listed above in CRITICAL CONSTRAINT section.


Output: [Question only, no explanations]
"""


synthetic_challenge_template_V2 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}

CRITICAL CONSTRAINT - MUST AVOID REPETITION:
{recent_questions}

Your task:
1. Ask about a specific numerical value, metric, or calculation.
2. Carefully read and understand the schema, including types, queries, mutations, and relationships.
3. Generate ONE natural question that a user might ask based on this file.
4. Output only the question, nothing else.
5. Ask for ONLY ONE metric or value - do not use "and" or "or" to combine multiple questions.
6. Do not include explanations, answers, or more than one question.
7. Ask about what CAN be queried, not specific made-up scenarios.
8. Focus on query patterns and schema structure, not fictional data.
9. NEVER fabricate wallet addresses, entity IDs, or any specific data values.
10. ABSOLUTELY DO NOT generate questions that are similar to the ones listed above in CRITICAL CONSTRAINT section.


Output: [Question only, no explanations]
"""


# llm / agent 生成问题效果一样, 都是比较级比较多
synthetic_challenge_template_V3 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}


Task: Generate ONE natural question about numerical data from the schema above.

Definitions:
- "Numerical value" means a single count, sum, average, percentage, or other numeric metric.
- Each question must involve exactly ONE metric.
- If the output would be a list, show only the first 3 results.


CRITICAL CONSTRAINT - MUST AVOID REPETITION:
{recent_questions}

Your task:
1. Ask about a specific numerical value, metric, or calculation.
2. Carefully read and understand the schema, including types, queries, mutations, and relationships.
3. Each question must focus on a single data point or calculation
5. Ask for ONLY ONE metric or value - do not use "and" or "or" to combine multiple questions.
6. Do not include explanations, answers, or more than one question.
7. Ask about what CAN be queried, not specific made-up scenarios.
8. Focus on query patterns and schema structure, not fictional data.
9. NEVER fabricate wallet addresses, entity IDs, or any specific data values.
10. ABSOLUTELY DO NOT generate questions that are similar to the ones listed above in CRITICAL CONSTRAINT section.


Output: [Question only, no explanations]
"""

# highest、total 问题比较多
synthetic_challenge_template_V4 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}


Task: Generate ONE natural question about numerical data from the schema above.

Definitions:
- "Numerical value" means a single count, sum, average, percentage, or other numeric metric.
- Each question must involve exactly ONE metric.


CRITICAL CONSTRAINT - MUST AVOID REPETITION:
{recent_questions}

Your task:
1. Ask about a specific numerical value, metric, or calculation.
2. Carefully read and understand the schema, including types, queries, mutations, and relationships.
3. Each question must focus on a single data point or calculation
5. Ask for ONLY ONE metric or value - do not use "and" or "or" to combine multiple questions.
6. Do not include explanations, answers, or more than one question.
7. Ask about what CAN be queried, not specific made-up scenarios.
8. Focus on query patterns and schema structure, not fictional data.
9. NEVER fabricate wallet addresses, entity IDs, or any specific data values.
10. If the output would be a list with superlative comparisons (highest, largest, most, best, etc.), randomly choose with 50% probability: (1) show the first 3 results, or (2) show only the single highest/largest result.
11. ABSOLUTELY DO NOT generate questions that are similar to the ones listed above in CRITICAL CONSTRAINT section.


Output: [Question only, no explanations]
"""

SYNTHETIC_PROMPT = PromptTemplate(
    input_variables=["entity_schema", "recent_questions"],
    template=synthetic_challenge_template_V4
)



# for demo purpose
synthetic_challage_subql_V2 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}

Available Addresses:
- Indexers: 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6, 0xF64476a9A06ABC89da3CE502c6E09b22B676C14E
- Consumer: 0x31E99bdA5939bA2e7528707507b017f43b67F89B

Available Era: 0x30, 0x40, 0x45, 0x48 (hexadecimal)

Task: Generate ONE natural question about numerical data from the schema above.

Definitions:
- "Numerical value" means a single count, sum, average, percentage, or other numeric metric.
- Each question must involve exactly ONE metric.
- If the output would be a list, show only the first 3 results.

CRITICAL CONSTRAINT - MUST AVOID REPETITION:
{recent_questions}

Requirements:
1. Ask about a specific numerical value, metric, or calculation
2. Ensure the question is answerable using the provided schema
3. Focus on indexer/consumer operations or performance
4. Use natural, conversational language
5. You may reference the specific addresses above if relevant
6. The question must specify a single era from the available list: 0x40, 0x48, 0x49, 0x50, 0x51
7. If the answer would be a list, limit results to the first 3 items
8. Ask for ONLY ONE metric or value - do not use "and" or "or" to combine multiple questions
9. Each question must focus on a single data point or calculation
10. Randomly vary between these three main topic categories with equal probability:
    - Indexer rewards (total rewards, reward distribution, etc.)
    - Stake (staking amounts, stake distribution, etc.)
11. ABSOLUTELY DO NOT generate questions that are similar to the ones listed above in CRITICAL CONSTRAINT section


Question Examples:
- "How many blocks did indexer 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6 process in era 0x48?"
- "What is the total gas consumed by all indexers in era 0x49?"
- "How many queries did the consumer submit during era 0x50, showing only the first 3 results?"
- "What percentage of indexing operations completed successfully in era 0x51?"
- "Show me the top 3 highest transaction counts per block in era 0x40"


Output: [Question only, no explanations]
"""


SYNTHETIC_PROMPT_SUBQL = PromptTemplate(
    input_variables=["entity_schema", "recent_questions"],
    template=synthetic_challage_subql_V2
)


score_template = """You are a strict fact-checking evaluator.  
Given a [Reference Answer] and a [Response], evaluate how factually close the Response is to the Reference Answer.  

Rules:  
1. Judge only based on factual correctness, not tone or style.  
2. Provide a single integer score between 0 and 10, where 0 = completely inconsistent, and 10 = perfectly consistent.  
3. You may use at most one decimal place (e.g., 7, 8.5, 10).
4. Output only the score as a number. Do not provide explanations or any extra text.  

[Reference Answer]:  
{ground_truth} 

[Response]:  
{miner_answer}  

Score:
"""

SCORE_PROMPT = PromptTemplate(
    input_variables=["ground_truth", "miner_answer"],
    template=score_template
)