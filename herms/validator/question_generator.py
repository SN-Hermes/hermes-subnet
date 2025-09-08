from typing import Dict
from langchain.schema import HumanMessage
from collections import deque
import difflib

from langchain_openai import ChatOpenAI
from loguru import logger

from common.prompt_template import SYNTHETIC_PROMPT

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

    def generate_question(self, project_cid: str, entity_schema: str, llm: ChatOpenAI | None) -> str:
        if not entity_schema:
            return ""
        if project_cid not in self.project_question_history:
            self.project_question_history[project_cid] = deque(maxlen=self.max_history)
        
        recent_questions = self.format_history_constraint(self.project_question_history[project_cid])
        prompt = SYNTHETIC_PROMPT.format(entity_schema=entity_schema, recent_questions=recent_questions)
        
        # logger.debug(f"Generated prompt for project {project_cid}:\n{prompt}")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        question = response.content.strip()
        
        self.add_to_history(project_cid, question)
        return question

    def _is_similar(self, new_question: str) -> bool:
        new_clean = new_question.lower().strip()
        
        for hist_question in self.question_history:
            hist_clean = hist_question.lower().strip()
            similarity = difflib.SequenceMatcher(None, new_clean, hist_clean).ratio()
            
            if similarity > self.similarity_threshold:
                return True
        
        return False
    def add_to_history(self, project_cid, question: str):
        if project_cid not in self.project_question_history:
            self.project_question_history[project_cid] = deque(maxlen=self.max_history)

        self.project_question_history[project_cid].append(question)

    def clear_history(self, project_cid: str):
        if project_cid in self.project_question_history:
            self.project_question_history[project_cid].clear()


question_generator = QuestionGenerator(max_history=5)

if __name__ == "__main__":
    import os
    import json

    file = '/Users/demon/Desktop/work/onf/network-hermes-subnet/projects/miner/QmQqqmwwaBben8ncfHo3DMnDxyWFk5QcEdTmbevzKj7DBd/config.json'
    with open(file, 'r', encoding='utf-8') as f:
        entity_schema = f.read()
        config = json.loads(entity_schema)
        entity_schema = config.get("schema_content", "")

    os.environ["OPENAI_API_KEY"] = "sk-"
    model_name = os.getenv("LLM_MODEL", "gpt-5")
    llm = ChatOpenAI(
        model=model_name,
        temperature=1
    )

    for _ in range(3):
        q = question_generator.generate_question("project_1", entity_schema, llm)
        print("generated question:", q)
