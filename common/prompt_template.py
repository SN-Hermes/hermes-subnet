from langchain.prompts import PromptTemplate

synthetic_template = """
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

SYNTHETIC_PROMPT = PromptTemplate(
    input_variables=["entity_schema"],
    template=synthetic_template
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