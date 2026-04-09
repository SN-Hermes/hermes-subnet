import random
from typing import Dict
from langchain_core.messages import HumanMessage
from collections import deque
import difflib
import json
import bittensor as bt
from pathlib import Path
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
from loguru import logger

from agent.stats import Phase, TokenUsageMetrics
from agent.subquery_graphql_agent.base import create_graphql_toolkit
from agent.subquery_graphql_agent.project import LocalProjectBase, RemoteChallenge
from agent.subquery_graphql_agent.tools import GraphQLSchemaInfoTool
from common.enums import RemoteChallengeType

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
        question_save_path: str | None = None,
        remote_played_save_path: str | None = None,
        save_interval: int = 3,
        wallet: bt.Wallet | None = None,
    ):
        self.max_history = max_history
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries
        self.project_question_history = {}
        self.save_path = question_save_path
        self.remote_played_save_path = remote_played_save_path
        self.generation_count = 0
        self.save_interval = save_interval
        self.wallet = wallet
        self.project_generation_counter = {}

        self.remote_played: dict[int, int] = {}  # challenge_id -> play_count
        self.remote_played_pivot = -1
        self.max_remote_played_cache = 1000

        # Load existing history if save_path exists
        if self.save_path:
            self._load_history()
        
        # Load remote played history
        if self.remote_played_save_path:
            self._load_remote_played()

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
            project_frequency: dict[str, int] = {}
        ) -> tuple[str, str, dict | None, str | None, RemoteChallenge | None]:
        if not project.schema_content:
            return "", "unknown", None, "schema not found", None

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

        async def try_with_generic():
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

        async def try_with_remote():
            freq = project_frequency.get(project.cid_hash, None)
            if freq is None or freq < 0:
                return "", None, "Remote challenge skipped due to frequency setting", None
            
            cur = self.project_generation_counter.get(cid_hash, 0)
            if freq > 0 and cur <= freq:
                return "", None, "Remote challenge skipped due to frequency control", None

            chs = await project.pull_remote_challenges(
                source=self.wallet.hotkey.ss58_address,
                sign_func=self.wallet.hotkey.sign,
            )
            if not chs:
                return "", None, "No available remote challenges", None
            
            allowed_versions = ["v1"]
            first_candidate = None
            second_candidate = None
            third_candidate = None
            for ch in chs:
                if ch.version in allowed_versions and ch.id > self.remote_played_pivot and ch.id not in self.remote_played:
                    first_candidate = ch
                    break

                if ch.version in allowed_versions and self.remote_played.get(ch.id, 0) < ch.max_count:
                    third_candidate = ch if third_candidate is None else third_candidate
                    if ch.id > self.remote_played_pivot:
                        second_candidate = ch if second_candidate is None else second_candidate

            challenge = first_candidate or second_candidate or third_candidate
            if not challenge:
                return "", None, "No available candidate challenges", None

            if challenge.type == RemoteChallengeType.FIXED.value:
                return challenge.question, None, None, challenge
            elif challenge.type == RemoteChallengeType.TOPIC.value:
                prompt = project.prompt_for_challenge_with_topic(
                    recent_questions,
                    challenge.version,
                    challenge.question,
                    challenge.instruction
                )

                toolkit = create_graphql_toolkit(
                    project.endpoint,
                    project.schema_content,
                    node_type=project.node_type,
                    manifest=None
                )
                tools = toolkit.get_tools()
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

                return question, d, None, challenge
            else:
                return "", None, f"Unknown challenge type: {challenge.type}", None

        typ = "remote"
        question, metrics_data, error, challenge = await try_with_remote()
        if not question:
            v = random.randint(0, 100)
            if v <= weight_a:
                question, metrics_data, error = await try_with_generic()
                typ = "generic"
            else:
                question, metrics_data, error = await try_with_tools()
                typ = "tools"
        
        if question:
            self.add_to_history(cid_hash, question)
        
        return question, typ, metrics_data, error, challenge

    def mark_success(self, question: str, cid_hash: str, typ: str, challenge: RemoteChallenge | None):
        if question:
            
            # Increment generation count and save if needed
            self.generation_count += 1
            if self.save_path and self.generation_count % self.save_interval == 0:
                self._save_history()
                self.generation_count = 0
            
            # Initialize counter for this project if not exists
            if cid_hash not in self.project_generation_counter:
                self.project_generation_counter[cid_hash] = 0
            if typ != "remote":
                self.project_generation_counter[cid_hash] += 1
            else:
                self.project_generation_counter[cid_hash] = 0

            if challenge:
                self.remote_played_pivot = challenge.id
                self.remote_played[challenge.id] = self.remote_played.get(challenge.id, 0) + 1
                self._cleanup_remote_played_cache()
                
                if self.remote_played_save_path:
                    self._save_remote_played()
    
    def _cleanup_remote_played_cache(self):
        if len(self.remote_played) <= self.max_remote_played_cache:
            return
        
        # Sort by challenge id (descending) and keep only the most recent ones
        sorted_ids = sorted(self.remote_played.keys(), reverse=True)
        
        # Keep only the most recent max_remote_played_cache records
        ids_to_keep = set(sorted_ids[:self.max_remote_played_cache])
        
        # Remove old records
        ids_to_remove = [cid for cid in self.remote_played if cid not in ids_to_keep]
        for cid in ids_to_remove:
            del self.remote_played[cid]
        
        if ids_to_remove:
            logger.info(f"Cleaned up {len(ids_to_remove)} old remote played records (kept {len(self.remote_played)})")
    
    def _load_remote_played(self):
        if not self.remote_played_save_path:
            return
        
        try:
            path = Path(self.remote_played_save_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.remote_played_pivot = data.get("pivot", -1)
                    # Convert string keys back to int
                    self.remote_played = {int(k): v for k, v in data.get("played", {}).items()}
                    logger.info(f"Loaded remote played history from {self.remote_played_save_path} ({len(self.remote_played)} records)")
        except Exception as e:
            logger.error(f"Error loading remote played history from {self.remote_played_save_path}: {e}")
    
    def _save_remote_played(self):
        if not self.remote_played_save_path:
            return
        
        try:
            path = Path(self.remote_played_save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "pivot": self.remote_played_pivot,
                "played": self.remote_played
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved remote played history to {self.remote_played_save_path} ({len(self.remote_played)} records)")
        except Exception as e:
            logger.error(f"Error saving remote played history to {self.remote_played_save_path}: {e}")

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
