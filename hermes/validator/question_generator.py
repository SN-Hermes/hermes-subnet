from typing import Dict
from langchain_core.messages import HumanMessage
from collections import deque
import difflib
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from loguru import logger

from agent.stats import Phase, TokenUsageMetrics
from agent.subquery_graphql_agent.base import ProjectConfig, create_graphql_toolkit
from agent.subquery_graphql_agent.tools import GraphQLSchemaInfoTool
from common.prompt_template import SYNTHETIC_PROMPT, SYNTHETIC_PROMPT_FALLBACK

class QuestionGenerator:
    max_history: int
    similarity_threshold: float
    max_retries: int
    project_question_history: Dict[str, deque]

    def __init__(self, max_history=10, similarity_threshold=0.75, max_retries=3):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.project_question_history = {}
        
    def format_history_constraint(self, recent_questions: deque) -> str:
        if not recent_questions:
            return ""
   
        formatted = "DO NOT REPEAT these recent questions:\n"
        for i, question in enumerate(recent_questions, 1):
            formatted += f"{i}. {question}\n"
        formatted += "\nGenerate a COMPLETELY DIFFERENT question with different metrics, addresses, or eras."
        return formatted

    async def generate_question(
            self,
            cid_hash: str,
            project: ProjectConfig,
            llm: ChatOpenAI,
            token_usage_metrics: TokenUsageMetrics | None = None,
            round_id: int = 0,
            weight_a: int = 60,
            weight_b: int = 40,
        ) -> tuple[str, str | None]:
        if not project.schema_content:
            return "", "schema not found"

        if cid_hash not in self.project_question_history:
            self.project_question_history[cid_hash] = deque(maxlen=self.max_history)

        recent_questions = self.format_history_constraint(self.project_question_history[cid_hash])

        async def try_with_tools():
            try:
                toolkit = create_graphql_toolkit(
                    project.endpoint,
                    project.schema_content,
                    node_type=project.node_type,
                    manifest=None
                )
                tools = toolkit.get_tools()
                schema_info_tool: GraphQLSchemaInfoTool = tools[0]
                prompt = SYNTHETIC_PROMPT.format(
                    entity_schema=project.schema_content,
                    recent_questions=recent_questions,
                    postgraphile_rules=schema_info_tool.postgraphile_rules,
                    weight_a=weight_a,
                    weight_b=weight_b
                )
                temp_executor = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt="",
                )
                response = await temp_executor.ainvoke(
                    { "messages": [{"role": "user", "content": prompt}] },
                    config={
                        "recursion_limit": 12,
                    },
                )
                question = response.get('messages', [])[-1].content
                if token_usage_metrics is not None:
                    token_usage_metrics.append(cid_hash, phase=Phase.GENERATE_QUESTION, response=response, extra={"round_id": round_id})
                return question, None

            except Exception as e:
                logger.error(f"Error occurred: {e}")
                return "", f"{e}"

        async def try_with_fallback():
            try:
                prompt = SYNTHETIC_PROMPT_FALLBACK.format(entity_schema=project.schema_content, recent_questions=recent_questions)
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                question = response.content.strip()
                if token_usage_metrics is not None:
                    token_usage_metrics.append(cid_hash, phase=Phase.GENERATE_QUESTION, response=response, extra = {"round_id": round_id})
                
                return question, None
            except Exception as e:
                logger.error(f"Error generating fallback question for project {cid_hash}: {e}")
                return "", f"{e}"

        question, error = await try_with_tools()
        if not question:
            question, error = await try_with_fallback()

        if not question:
            self.add_to_history(cid_hash, question)
        
        return question, error

    def _is_similar(self, new_question: str) -> bool:
        new_clean = new_question.lower().strip()
        
        for hist_question in self.question_history:
            hist_clean = hist_question.lower().strip()
            similarity = difflib.SequenceMatcher(None, new_clean, hist_clean).ratio()
            
            if similarity > self.similarity_threshold:
                return True
        
        return False

    def add_to_history(self, cid_hash, question: str):
        if cid_hash not in self.project_question_history:
            self.project_question_history[cid_hash] = deque(maxlen=self.max_history)

        self.project_question_history[cid_hash].append(question)

    def clear_history(self, cid_hash: str):
        if cid_hash in self.project_question_history:
            self.project_question_history[cid_hash].clear()


question_generator = QuestionGenerator(max_history=24)
