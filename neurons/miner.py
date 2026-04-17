# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from loguru._logger import Logger
from bittensor.core.stream import StreamingSynapse
from agent.stats import Phase, ProjectUsageMetrics, TokenUsageMetrics
from common.prompt_template import get_miner_self_tool_prompt, fill_miner_self_tool_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from common.table_formatter import table_formatter
from common.agent_manager import AgentManager
from common.enums import ErrorCode, RoleFlag
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, StatsMiddleware, SyntheticNonStreamSynapse
from common.sqlite_manager import SQLiteManager
import common.utils as utils
from common.settings import settings
from hermes.base import BaseNeuron
from hermes.validator.scorer_manager import ScorerManager

ROLE = "miner"

settings.load_env_file(ROLE)
LOGGER_DIR = os.getenv("LOGGER_DIR", f"logs/{ROLE}")


class Miner(BaseNeuron):

    @property
    def role(self) -> str:
        return ROLE

    def __init__(self, config_loguru: bool = True):
        if config_loguru:
            HermesLogger.configure_loguru(
                file=f"{LOGGER_DIR}/{self.role}.log",
                error_file=f"{LOGGER_DIR}/{self.role}_error.log"
            )
        super().__init__()
        
        # Event to signal that agents are ready
        self.agents_ready_event = asyncio.Event()
        self._mock_config_shm = None

    async def start(self):
        try:
            super().start(flag=RoleFlag.MINER)

            score_model_name = os.getenv("LLM_MODEL") or os.getenv("SCORE_LLM_MODEL", "google/gemini-3-flash-preview")
            score_model_base_url = os.getenv("SCORE_LLM_MODEL_BASE_URL", None)
            score_model_api_key = os.getenv("SCORE_LLM_MODEL_API_KEY", None)
            score_model_args = {}
            if score_model_base_url:
                score_model_args["base_url"] = score_model_base_url
            if score_model_api_key:
                score_model_args["api_key"] = score_model_api_key

            score_timeout = int(os.getenv("SCORE_TIMEOUT", 60))
            self.llm_score = ChatOpenAI(
                model=score_model_name,
                temperature=0,
                timeout=score_timeout,
                max_retries=3,
                **score_model_args
            )
            self.scorer_manager = ScorerManager(
                llm_score=self.llm_score,
            )

            self.project_usage_metrics = ProjectUsageMetrics()
            self.token_usage_metrics = TokenUsageMetrics()

            self.db_queue = asyncio.Queue()
            self.sqlite_manager = SQLiteManager(f".data/{self.role}.db")
            self.axon = bt.axon(
                wallet=self.settings.wallet,
                port=self.settings.port,
                ip=self.settings.external_ip,
                external_ip=self.settings.external_ip,
                external_port=self.settings.port
            )
            self.axon.app.add_middleware(
                StatsMiddleware,
                sqlite_manager=self.sqlite_manager,
                project_usage_metrics=self.project_usage_metrics,
                token_usage_metrics=self.token_usage_metrics,
            )

            def allow_all(synapse: CapacitySynapse) -> None:
                return None

            self.axon.attach(
                forward_fn=self.forward_organic_stream,
            )

            self.axon.attach(
                forward_fn=self.forward_organic_non_stream
            )

            self.axon.attach(
                forward_fn=self.forward_synthetic_non_stream
            )

            self.axon.attach(
                forward_fn=self.forward_capacity,
                verify_fn=allow_all
            )

            # self.axon.serve(netuid=self.settings.netuid, subtensor=self.settings.subtensor)

            self.axon.start()
            logger.info(f"Miner starting at block: {self.settings.subtensor.block}")
            logger.info(f"Axon serving on port: {self.settings.port}")
            logger.info(f"Axon created: {self.axon}")
            logger.info(f"Miner starting at block: {self.settings.subtensor.block}")
            logger.info(f"Stats at: http://{self.settings.external_ip}:{self.settings.port}/stats")

            self.agent_manager = None
            self._running_tasks = [
                asyncio.create_task(
                    self.refresh_agents()
                ),
                asyncio.create_task(
                    self.profile_tools_stats()
                ),
                asyncio.create_task(
                    self.db_writer()
                )
            ]

            if self.settings.is_running_mock_mode:
                logger.info("[Miner] Mock mode enabled - waiting for agents to be ready...")
                try:
                    await asyncio.wait_for(self.agents_ready_event.wait(), timeout=60 * 5)
                except asyncio.TimeoutError:
                    logger.error("[Miner] Timeout waiting for agents to be ready")
                    raise RuntimeError("Agents failed to initialize within 5 mins")
                logger.info("[Miner] Agents are ready - setting up mock configuration...")
                
                from common.mock_config import MockConfigSharedMemory
                mock_config = MockConfigSharedMemory()
                config_data = {
                    "uid": self.uid,
                    "external_ip": self.settings.external_ip,
                    "port": self.settings.port,
                    "miner_project_dir": str(self.agent_manager.save_project_dir),
                    "env_file": self.settings.env_file,
                }
                
                # Write to shared memory
                if mock_config.write(config_data):
                    logger.info("[Miner] ✅ Mock configuration written to shared memory")
                    self._mock_config_shm = mock_config
                else:
                    raise RuntimeError("[Miner] Failed to write configuration to shared memory")

            await asyncio.gather(*self._running_tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("[Miner] Miner start process interrupted")
            # Cancel all running tasks
            if hasattr(self, '_running_tasks'):
                for task in self._running_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to complete cancellation
                await asyncio.gather(*self._running_tasks, return_exceptions=True)
            raise  # Re-raise to allow graceful shutdown at higher level
        except Exception as e:
            logger.error(f"[Miner] Failed to start miner: {e}")
            raise
        finally:
            # Always cleanup shared memory on exit
            if self._mock_config_shm:
                logger.info("[Miner] Cleaning up shared memory...")
                try:
                    self._mock_config_shm.cleanup(unlink=True)  # Miner is the owner, so unlink
                    logger.info("[Miner] Shared memory cleaned up")
                except Exception as e:
                    logger.warning(f"[Miner] Error cleaning up shared memory: {e}")

    async def db_writer(self):
        try:
            last_check_time = 0

            while True:
                if int(time.time()) - last_check_time > 60 * 10:
                    self.sqlite_manager.cleanup_old_records()
                    last_check_time = int(time.time())

                item = await self.db_queue.get()

                type = item.get("type")
                status_code = item.get("status_code")
                project_id = item.get("project_id")

                target = self.project_usage_metrics.synthetic_project_usage if type == 0 else self.project_usage_metrics.organic_project_usage
                target.incr(
                    project_id,
                    success=False if status_code != 200 else True
                )

                tool_hit = item.get("tool_hit")

                logger.info(f"[DB Writer] - Inserting request log for project {project_id} with status code {status_code}, type:{type}, tool_hit: {tool_hit}")

                if tool_hit and tool_hit != '[]':
                    tool_hit_list = json.loads(tool_hit)
                    target = self.project_usage_metrics.synthetic_tool_usage if type == 0 else self.project_usage_metrics.organic_tool_usage
                    for tool_name, count in tool_hit_list:
                        target.incr(tool_name, count)

                self.sqlite_manager.insert_request(**item)
                self.db_queue.task_done()
        except KeyboardInterrupt:
            logger.info("[Miner] DB writer interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[Miner] DB writer error: {e}")
            raise

    def _read_solution_prompt(self, cid_hash: str) -> str | None:
        """Read solution prompt from projects/miner/{cid_hash}/solution/001.MD"""
        try:
            solution_file = self.agent_manager.save_project_dir / cid_hash / "solution" / "001.MD"
            if solution_file.exists():
                with open(solution_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        logger.debug(f"[Miner] Loaded solution prompt for {cid_hash} ({len(content)} chars)")
                        return content
            return None
        except Exception as e:
            logger.warning(f"[Miner] Failed to read solution prompt for {cid_hash}: {e}")
            return None

    async def _handle_task(
            self,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse,
            log: Logger,
    ) -> SyntheticNonStreamSynapse | OrganicNonStreamSynapse:
        question = task.get_question()

        cid_hash = task.cid_hash
        agent = self.agent_manager.get_agent(cid_hash)

        enable_fallback = os.getenv("ENABLE_FALL_BACK_GRAPHQL_AGENT", "false").lower() == "true"
        
        # Read solution prompt if exists
        solution_prompt = self._read_solution_prompt(cid_hash)
        solution_prompt += f" # block height is {task.block_height}"
        
        print('---------solution prompt---------\n')
        print(solution_prompt)
        print('---------\n')

        # simi, err = await self.scorer_manager.cal_solution_similarity_score(
        #     task.solution,
        #     solution_prompt
        # )
        # if err:
        #     log.error(f"[Miner] Similarity evaluation error for task {task.id}: {err}")
        #     return
        # logger.info(f"[Miner] Solution similarity for task {task.id}: {simi}")


        if isinstance(task, SyntheticNonStreamSynapse):
            tag = "Synthetic"
            type = 0
            is_synthetic = True
            phase = Phase.MINER_SYNTHETIC
            
            # Build system prompt
            # base_system_prompt = get_miner_self_tool_prompt(
            #     block_height=task.block_height, 
            #     node_type=agent.project.node_type if agent else "unknown",
            #     enable_fallback=enable_fallback
            # )
            messages = [
                SystemMessage(content=solution_prompt),
                HumanMessage(content=question)
            ]

        elif isinstance(task, OrganicNonStreamSynapse):
            tag = "Organic"
            type = 1
            is_synthetic = False
            phase = Phase.MINER_ORGANIC_NONSTREAM
            messages = [
                SystemMessage(content=get_miner_self_tool_prompt(
                    block_height=task.block_height, 
                    node_type=agent.project.node_type if agent else "unknown"
                ))
            ] + task.to_messages()
        else:
            raise ValueError("Unsupported task type")

        answer = None
        usage_info = {}
        tool_hit = []
        graphql_agent_inner_tool_calls = []
        error = None
        status_code = ErrorCode.SUCCESS

        before = time.perf_counter()

        try:
            if not agent:
                log.warning(f"[{tag}] - {task.id} No agent found for project {cid_hash}")
                error = f"No agent found for project {cid_hash}"
                status_code = ErrorCode.AGENT_NOT_FOUND
            else:
                r = await agent.ainvoke({"messages": messages, "block_height": task.block_height})
                print(r)
                (
                    answer,
                    usage_info,
                    tool_hit,
                    graphql_agent_inner_tool_calls,
                    error,
                    status_code
                ) = self.get_answer(phase, task, r)
        except Exception as e:
            log.error(f"handle task error {task.id} - {question}. {e}\n")
            error = str(e)
            status_code = ErrorCode.INTERNAL_SERVER_ERROR
        
        score = 0.0
        # self-check score
        if status_code == ErrorCode.SUCCESS:
            summary_response, e = await self.scorer_manager._cal_ground_truth(task.ground_truth, answer)
            if not e:
                log.info(f"Self-check score summary: {summary_response}")
                score_raw = summary_response.content.strip() if summary_response.content else "0.0"
                score = utils.safe_float_convert(score_raw)
        else:
            solution_prompt = None

        elapsed = utils.fix_float(time.perf_counter() - before)
        
        self.print_table(
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
            error=error,
            status_code=status_code,

            tag=tag,
            task=task,
            elapsed=elapsed,
            log=log,
        )

        task.response = answer
        task.ground_truth = None
        task.error = error
        task.status_code = status_code.value
        task.usage_info = usage_info
        task.graphql_agent_inner_tool_calls = graphql_agent_inner_tool_calls
        task.miner_model_name = self.llm.model_name
        task.graphql_agent_model_name = agent.llm.model_name if agent else "unknown"
        task.solution = solution_prompt
        task.score = str(score)

        self.put_db(
            type=type,
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            error=error,
            status_code=status_code,
            elapsed=elapsed,
            task=task,
        )
        return task

    def get_answer(
        self,
        phase: Phase,
        task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
        r: dict
    ) -> tuple[str | None, dict, list, list[str], str | None, ErrorCode]:
        # logger.info(f"[{tag}] - {task.id} Agent response: {r}")
        
        usage_info = self.token_usage_metrics.parse(task.cid_hash, phase, r)
        self.token_usage_metrics.append(usage_info)

        # check tool stats
        tool_hit = utils.try_get_tool_hit(
            r.get('messages', []),
        )

        if r.get('graphql_agent_hit', False):
            tool_hit.append(("graphql_agent_tool", 1))

        graphql_agent_inner_tool_calls: list[str] = r.get('tool_calls', [])

        error = None
        status_code = ErrorCode.SUCCESS

        answer = None
        if r.get('error', None) is not None:
            error = r.get('error')
            status_code = ErrorCode.LLM_ERROR
        else:
            answer = r.get('messages')[-1].content or None
            if not answer:
                error = utils.try_get_invalid_tool_messages(r.get('messages', []))
                status_code = ErrorCode.TOOL_ERROR if error is not None else status_code

        
        return answer, usage_info, tool_hit, graphql_agent_inner_tool_calls, error, status_code
        
    def print_table(
            self,
            answer: str,
            usage_info: str,
            tool_hit: list,
            graphql_agent_inner_tool_calls: list,
            error: str,
            status_code: ErrorCode,
            tag: str,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
            elapsed: float,
            log: Logger,
    ):
        tool_hit_names = [t[0] for t in tool_hit]
        rows = [f"💬 Answer: {answer}\n"]
        if error:
            rows.append(f"⚠️ {status_code.value} | {error}\n")
        
        rows.append(f" 📊 Metrics Data: {usage_info}\n")

        if os.getenv("ENABLE_GRAPHQL_AGENT_TOOL_CALLS_LOG", "false").lower() == "true":
            rows.append(f" 📊 GraphQL Agent tools: {graphql_agent_inner_tool_calls}\n")

        if len(tool_hit_names) > 0:
            rows.append(f"🛠️ Tools Hit: {', '.join(tool_hit_names)}\n")
        rows.append(f"⏱️ Cost: {elapsed}s")
        
        status_icon = "✅" if status_code == ErrorCode.SUCCESS else "❌"
        output = table_formatter.create_single_column_table(
            f"🤖 {status_icon} {tag}: {task.get_question()} ({task.id})",
            rows,
            caption=task.cid_hash
        )
        log.info(f"\n{output}")

    def put_db(
            self,
            type: int,
            answer: str,
            usage_info: str,
            tool_hit: list,
            error: str | None,
            status_code: ErrorCode,
            elapsed: float,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
    ):
        response_data = answer if status_code == ErrorCode.SUCCESS else error
        
        self.db_queue.put_nowait({
            "type": type,
            "source": task.dendrite.hotkey,
            "task_id": task.id,
            "project_id": task.cid_hash,
            "cid": task.cid_hash,
            "request_data": task.get_question(),
            "response_data": response_data or '',
            "status_code": status_code.value,
            "tool_hit": json.dumps(tool_hit),
            "cost": elapsed,
            "token_usage_info": json.dumps(usage_info) if usage_info else ''
        })

    async def forward_synthetic_non_stream(self, task: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        now = int(datetime.now().timestamp())
        log = logger.bind(source=task.dendrite.hotkey)
        log.info(f"[Miner] receiving synthetic task: {task.id} at {now}")

        task.recv_start_time = now
        await self._handle_task(task, log)
        return task

    async def forward_organic_stream(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        from starlette.types import Send
        log = logger.bind(source=synapse.dendrite.hotkey)
        log.info(f"\n🤖 [Miner] Received organic stream: {synapse.id}")

        messages = synapse.to_messages()
        graph, graphql_agent = self.agent_manager.get_miner_agent(synapse.cid_hash)

        if not graph:
            error_msg = f"Error: No agent found for project {synapse.cid_hash}"
            log.warning(f"[Miner] - {synapse.id} {error_msg}")
            async def error_streamer(send: Send):
                error_line = json.dumps({
                    "type": "data",
                    "data": error_msg
                }) + "\n"
                await send({
                    "type": "http.response.body",
                    "body": error_line.encode('utf-8'),
                    "more_body": False
                })
            
            return synapse.create_streaming_response(error_streamer)
        
        fill_miner_self_tool_prompt(messages, block_height=synapse.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")

        async def token_streamer(send: Send):
            r = None
            tag = "Organic-S"
            phase = Phase.MINER_ORGANIC_STREAM
            before = time.perf_counter()
            async for event in graph.astream(
                {
                    "messages": messages,
                    "block_height": synapse.block_height
                },
                version="v2"
            ):
                for key, value in event.items():
                    if key == "final":
                        r = value
                        message = value.get("messages", [])[-1].content

                        if r.get('error', None) is not None:
                            message = r.get('error')
                        idx = 0
                        while idx < len(message):
                            chunk = message[idx:idx+10]
                            # Send data chunks in JSONL format
                            data_line = json.dumps({
                                "type": "data",
                                "data": chunk
                            }) + "\n"
                            await send({
                                "type": "http.response.body",
                                "body": data_line.encode('utf-8'),
                                "more_body": True
                            })
                            await asyncio.sleep(0.25)
                            idx += 10
                
            elapsed = utils.fix_float(time.perf_counter() - before)
            synapse.elapsed_time = elapsed
            (
                answer,
                usage_info,
                tool_hit,
                graphql_agent_inner_tool_calls,
                response,
                error,
                status_code
            ) = self.get_answer(phase, synapse, r)

            # Send metadata in JSONL format
            metadata_line = json.dumps({
                "type": "meta",
                "data": {
                    "miner_model_name": self.llm.model_name,
                    "graphql_agent_model_name": graphql_agent.llm.model_name,
                    "elapsed": elapsed,
                    "status_code": status_code.value,
                    "error": error,
                    "graphql_agent_inner_tool_calls": graphql_agent_inner_tool_calls,
                    "usage_info": usage_info
                }
            }) + "\n"
            await send({
                "type": "http.response.body",
                "body": metadata_line.encode('utf-8'),
                "more_body": False
            })

            self.print_table(
                answer=answer,
                usage_info=usage_info,
                tool_hit=tool_hit,
                graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
                error=error,
                status_code=status_code,
                tag=tag,
                task=synapse,
                elapsed=elapsed,
                log=log,
            )
            self.put_db(
                type=2,
                answer=answer,
                usage_info=usage_info,
                tool_hit=tool_hit,
                error=error,
                status_code=status_code,
                elapsed=elapsed,
                task=synapse,
            )

        return synapse.create_streaming_response(token_streamer)

    async def forward_organic_non_stream(self, task: OrganicNonStreamSynapse) -> OrganicNonStreamSynapse:
        log = logger.bind(source=task.dendrite.hotkey)
        await self._handle_task(task, log)
        return task
        
    async def forward_capacity(self, synapse: CapacitySynapse) -> CapacitySynapse:
        logger.debug(f"[Miner] Received capacity request")
        if not self.agent_manager:
            logger.warning(f"[Miner] No agent manager found")
            synapse.response = {
                "role": "miner",
                "capacity": {
                    "projects": []
                }
            }
            return synapse

        cid_hashs = self.agent_manager.get_agents().keys()
        synapse.response = {
            "role": "miner",
            "capacity": {
                "projects": list(cid_hashs)
            }
        }
        return synapse

    async def invoke_graphql_agent(self, synapse: SyntheticNonStreamSynapse) -> str:
        _, _, graphql_agent = self.agent_manager.get_miner_agent(synapse.project_id)
        # For synthetic challenges, always attempt to answer without domain limitations
        response = await graphql_agent.query_no_stream(synapse.question, is_synthetic=True)
        answer = response.get('messages')[-1].content
        return answer

    async def invoke_miner_agent(self, synapse: SyntheticNonStreamSynapse) -> str:
        agent_graph, _, _ = self.agent_manager.get_miner_agent(synapse.project_id)
        response = await agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": synapse.question}]}
        )
        answer = response.get('messages')[-1].content
        return answer

    async def refresh_agents(self, force_load=False):
        current_dir = Path(__file__).parent
        save_project_dir = current_dir.parent / "projects" / self.role

        model = os.environ.get("MINER_LLM_MODEL", "google/gemini-3-flash-preview")
        self.llm = ChatOpenAI(
            model=model,
            temperature=1
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm,
        )

        mode = 'load' if force_load else os.getenv("PROJECT_PULL_MODE", "pull")
        # await self.agent_manager.start(mode == "pull", role="miner")

        refresh_agents_interval = int(os.getenv("REFRESH_AGENTS_INTERVAL", 60 * 5))  # seconds

        logger.info(f"[MINER] Using LLM model: {model} for miner self-owned agent")
        logger.info(f"[MINER] Using KEY: {utils.format_openai_key()}")

        silent = False
        first_load_complete = False
        while True:
            try:
                self.settings.reread()
                await self.agent_manager.start(mode == "pull", role="miner", silent=silent)
                
                # Signal that agents are ready after first successful load
                if not first_load_complete:
                    first_load_complete = True
                    self.agents_ready_event.set()
                    logger.info("[MINER] Agents initialized and ready - event signaled")
                
                silent = True
                mode = 'pull'  # after first load, always pull updates
            except Exception as e:
                logger.error(f"refresh_agents error: {e}")
            finally:
                await asyncio.sleep(refresh_agents_interval)

    async def profile_tools_stats(self):
        try:
            while True:
                await asyncio.sleep(60 * 1)
                logger.info(f"[MINER] usage stats: {json.dumps(self.project_usage_metrics.stats())}")
        except KeyboardInterrupt:
            logger.info("[Miner] Profile tools stats interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[Miner] Profile tools stats error: {e}")
            raise

if __name__ == "__main__":
    try:
        miner = Miner()
        asyncio.run(miner.start())

        # Keep the miner running
        while True:
            time.sleep(60 * 2)
    except KeyboardInterrupt:
        logger.info("[Miner] Received interrupt signal, shutting down gracefully...")
    except Exception as e:
        logger.error(f"[Miner] Unexpected error: {e}")
        logger.exception(e)
    finally:
        logger.info("[Miner] Shutdown complete")
        sys.exit(0)
