import random
from typing import Dict
from langchain_core.messages import HumanMessage
from collections import deque
import difflib
import json
from pathlib import Path
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from loguru import logger

from agent.stats import Phase, TokenUsageMetrics
from agent.subquery_graphql_agent.base import create_graphql_toolkit
from agent.subquery_graphql_agent.project import LocalProjectBase
from agent.subquery_graphql_agent.tools import GraphQLSchemaInfoTool

class QuestionGenerator:
    max_history: int
    similarity_threshold: float
    max_retries: int
    project_question_history: Dict[str, deque]
    save_path: str | None
    generation_count: int
    save_interval: int

    def __init__(
        self,
        max_history=10,
        similarity_threshold=0.75,
        max_retries=3,
        save_path: str | None = None,
        save_interval: int = 3
    ):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.project_question_history = {}
        self.save_path = save_path
        self.generation_count = 0
        self.save_interval = save_interval
        
        # Load existing history if save_path exists
        if self.save_path:
            self._load_history()

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
            project: LocalProjectBase,
            llm: ChatOpenAI,
            token_usage_metrics: TokenUsageMetrics | None = None,
            round_id: int = 0,
            weight_a: int = 70,
            weight_b: int = 30,
        ) -> tuple[str, dict | None, str | None]:
        if not project.schema_content:
            return "", None, "schema not found"

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
                prompt = project.prompt_for_challenge_with_tools(recent_questions, schema_info_tool.postgraphile_rules)
                temp_executor = create_react_agent(
                    model=llm,
                    tools=tools,
                    prompt=None,
                )
                response = await temp_executor.ainvoke(
                    { "messages": [{"role": "user", "content": prompt}] },
                    config={
                        "recursion_limit": 12,
                    },
                )
                question = response.get('messages', [])[-1].content
                d = None
                if token_usage_metrics is not None:
                    d = token_usage_metrics.parse(
                        cid_hash, phase=Phase.GENERATE_QUESTION, response=response, extra={"round_id": round_id}
                    )
                    token_usage_metrics.append(d)
                return question, d, None

            except Exception as e:
                logger.error(f"Error occurred: {e}")
                return "", None, f"{e}"

        async def try_with_fallback():
            try:
                prompt = project.prompt_for_challenge(recent_questions)
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                question = response.content.strip()
                d = None
                if token_usage_metrics is not None:
                    d = token_usage_metrics.parse(
                        cid_hash, phase=Phase.GENERATE_QUESTION, response=response, extra={"round_id": round_id}
                    )
                    token_usage_metrics.append(d)
                
                return question, d, None
            except Exception as e:
                logger.error(f"Error generating fallback question for project {cid_hash}: {e}")
                return "", None, f"{e}"

        v = random.randint(0, 100)
        if v <= weight_a:
            question, metrics_data, error = await try_with_fallback()
        else:
            question, metrics_data, error = await try_with_tools()

        if question:
            self.add_to_history(cid_hash, question)
            
            # Increment generation count and save if needed
            self.generation_count += 1
            if self.save_path and self.generation_count % self.save_interval == 0:
                self._save_history()
                self.generation_count = 0

        return question, metrics_data, error

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

    def _load_history(self):
        """Load question history from save_path if it exists"""
        if not self.save_path:
            return
        
        try:
            path = Path(self.save_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert lists back to deques with maxlen
                    for cid_hash, questions in data.items():
                        self.project_question_history[cid_hash] = deque(questions, maxlen=self.max_history)
                logger.info(f"Loaded question history from {self.save_path}")
        except Exception as e:
            logger.error(f"Error loading question history from {self.save_path}: {e}")

    def _save_history(self):
        """Save question history to save_path"""
        if not self.save_path:
            return
        
        try:
            path = Path(self.save_path)
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert deques to lists for JSON serialization
            data = {
                cid_hash: list(questions)
                for cid_hash, questions in self.project_question_history.items()
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved question history to {self.save_path} (generation count: {self.generation_count})")
        except Exception as e:
            logger.error(f"Error saving question history to {self.save_path}: {e}")
