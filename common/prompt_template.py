from langchain.prompts import PromptTemplate

synthetic_template_V1 = """
You are given the following document as background context:
{entity_schema}
Your task:
1. Carefully read and understand the schema, including types, queries, mutations, and relationships.
2. Generate ONE natural question that a user might ask based on this file.
3. The question must be related to a numerical value (e.g., quantity, percentage, date, amount, measurement) that appears in the file.
4. The question must explicitly relate to "indexer".
5. Output only the question, nothing else.
6. Do not include explanations, answers, or more than one question.

Now generate the question:
"""

synthetic_template_V2 = """
You are a question generator for database schema analysis.

Background Context:
{entity_schema}

Available Addresses:
- Indexers: 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6, 0xF64476a9A06ABC89da3CE502c6E09b22B676C14E
- Consumer: 0x31E99bdA5939bA2e7528707507b017f43b67F89B

Available Era Range: 0x40 to 0x51 (hexadecimal)

Task: Generate ONE natural question about numerical data from the schema above.

Requirements:
1. Ask about a specific numerical value, metric, or calculation
2. Ensure the question is answerable using the provided schema
3. Focus on indexer/consumer operations or performance
4. Use natural, conversational language
5. You may reference the specific addresses above if relevant
6. The question must specify a single era from the range 0x40 to 0x51
7. If the answer would be a list, limit results to the first 3 items

Question Examples:
- "How many blocks did indexer 0xe60554D90AF0e84A9C3d1A8643e41e49403945a6 process in era 0x45?"
- "What is the total gas consumed by all indexers in era 0x48?"
- "How many queries did the consumer submit during era 0x4A, showing only the first 3 results?"
- "What percentage of indexing operations completed successfully in era 0x51?"
- "Show me the top 3 highest transaction counts per block in era 0x40"

Output: [Question only, no explanations]
"""

SYNTHETIC_PROMPT = PromptTemplate(
    input_variables=["entity_schema"],
    template=synthetic_template_V2
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