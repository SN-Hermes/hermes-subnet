# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 <your name>

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
import json
import os
from pathlib import Path
import time
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from loguru._logger import Logger
from bittensor.core.stream import StreamingSynapse
from agent.stats import Phase, ProjectUsageMetrics, TokenUsageMetrics
from common.prompt_template import get_miner_self_tool_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from common.table_formatter import table_formatter
from common.agent_manager import AgentManager
from common.errors import ErrorCode
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, StatsMiddleware, SyntheticNonStreamSynapse
from common.sqlite_manager import SQLiteManager
import common.utils as utils
from common.settings import settings
from hermes.base import BaseNeuron
import agent.graphql_agent as subAgent

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

    async def start(self):
        try:
            super().start()

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

            self.axon.serve(netuid=self.settings.netuid, subtensor=self.settings.subtensor)

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

            await asyncio.gather(*self._running_tasks)
        except KeyboardInterrupt:
            logger.info("[Miner] Miner start process interrupted by user")
            # Cancel all running tasks
            if hasattr(self, '_running_tasks'):
                for task in self._running_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to complete cancellation
                await asyncio.gather(*self._running_tasks, return_exceptions=True)
            logger.info("[Miner] All tasks cancelled successfully")
            raise  # Re-raise to allow graceful shutdown at higher level
        except Exception as e:
            logger.error(f"[Miner] Failed to start miner: {e}")
            raise

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

    async def _handle_task(
            self,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse,
            log: Logger,
    ) -> SyntheticNonStreamSynapse | OrganicNonStreamSynapse:
        tag = "Synthetic"
        phase = Phase.MINER_SYNTHETIC
        type = 0
        question = task.get_question()
        is_synthetic = True

        cid_hash = task.cid_hash
        graph, graphql_agent = self.agent_manager.get_miner_agent(cid_hash)

        messages = [
            SystemMessage(content=get_miner_self_tool_prompt(block_height=task.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")),
            HumanMessage(content=question)
        ]

        if isinstance(task, OrganicNonStreamSynapse):
            tag = "Organic"
            type = 1
            is_synthetic = False
            phase = Phase.MINER_ORGANIC
            messages = task.to_messages()

        tool_hit = []
        graphql_agent_inner_tool_calls = []
        answer = None
        response = None
        error = None
        status_code = ErrorCode.SUCCESS

        before = time.perf_counter()

        usage_info = ''
        try:
            if not graph:
                log.warning(f"[{tag}] - {task.id} No agent found for project {cid_hash}")
                error = f"No agent found for project {cid_hash}"
                status_code = ErrorCode.AGENT_NOT_FOUND
            else:
                r = await graph.ainvoke({"messages": messages, "block_height": task.block_height})

                usage_info = self.token_usage_metrics.append(cid_hash, phase, r)

                # check tool stats
                tool_hit = utils.try_get_tool_hit(
                    r.get('messages', []),
                    # exclude_tools=exclude_tools
                )

                if r.get('graphql_agent_hit', False):
                    tool_hit.append(("graphql_agent_tool", 1))

                graphql_agent_inner_tool_calls = r.get('tool_calls', [])

                answer = r.get('messages')[-1].content or None
                if not answer:
                    error = utils.try_get_invalid_tool_messages(r.get('messages', []))
                    status_code = ErrorCode.TOOL_ERROR if error is not None else status_code

                if status_code == ErrorCode.SUCCESS:
                    # For both organic and synthetic, only return the final answer
                    response = answer

        except Exception as e:
            # log.error(f"[Synthetic] - {task.id} Agent error: {e}")
            error = str(e)
            status_code = ErrorCode.INTERNAL_SERVER_ERROR

        elapsed = utils.fix_float(time.perf_counter() - before)
        
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
            f"🤖 {status_icon} {tag}: {question} ({task.id})",
            rows,
            caption=cid_hash
        )
        log.info(f"\n{output}")

        task.response = response
        task.error = error
        task.status_code = status_code

        response_data = answer if status_code == ErrorCode.SUCCESS else task.error
        self.db_queue.put_nowait({
            "type": type,
            "source": task.dendrite.hotkey,
            "task_id": task.id,
            "project_id": task.cid_hash,
            "cid": task.cid_hash,
            "request_data": question,
            "response_data": response_data or '',
            "status_code": task.status_code,
            "tool_hit": json.dumps(tool_hit),
            "cost": elapsed,
            "token_usage_info": json.dumps(usage_info) if usage_info else ''
        })
        return task

    async def forward_synthetic_non_stream(self, task: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        log = logger.bind(source=task.dendrite.hotkey)
        await self._handle_task(task, log)
        return task

    async def forward_organic_stream(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        from starlette.types import Send
        log = logger.bind(source=synapse.dendrite.hotkey)
        log.info(f"\n🤖 [Miner] Received organic stream: {synapse.completion}")

        # user_messages = [msg for msg in synapse.completion.messages if msg.role == "user"]
        # user_input = user_messages[-1].content
        messages = synapse.to_messages()
        graph, graphql_agent = self.agent_manager.get_miner_agent(synapse.cid_hash)

        async def token_streamer(send: Send):
            # logger.info(f"\n🤖 [Miner] Starting agent stream: {messages}")
            # buffered_stream = []
            # active_checkpoint_ns = None
            # async for event in graph.astream_events({"messages": messages}, version="v2"):
            #     logger.info(f"oooooooooooooooooooooooo: {event}\n")
            #     if event["event"] == "on_chat_model_stream":
            #         checkpoint_ns = event["metadata"].get("checkpoint_ns", "")
            #         message = event["data"].get("chunk", {})
            #         content = message.content
            #         words.append(content)
            #         graph_node_name = checkpoint_ns.split(":")[0] if checkpoint_ns else "unknown"
                    # logger.info(f"   {event['metadata']} ---- {event['metadata'].get('langgraph_node','')} - {event['data']} -----graph_node_name: {graph_node_name} ==== {content} type: {type(content)}")

                    # if graph_node_name == "final_filter":
                    #     await send({
                    #             "type": "http.response.body",
                    #             "body": content.encode('utf-8'),
                    #             "more_body": True
                    #     })

                    # if active_checkpoint_ns is None:
                    #     active_checkpoint_ns = graph_node_name

                    # if graph_node_name == active_checkpoint_ns:
                    #     if graph_node_name == "graphql_agent":
                    #         await send({
                    #             "type": "http.response.body",
                    #             "body": content.encode('utf-8'),
                    #             "more_body": True
                    #         })
                    #     else:
                    #         buffered_stream.append(content)
                    # else:
                    #     buffered_stream.clear()
                    #     active_checkpoint_ns = graph_node_name
                    #     await send({
                    #         "type": "http.response.body",
                    #         "body": content.encode('utf-8'),
                    #         "more_body": True
                    #     })

            # for content in buffered_stream:
            #     # logger.info(f"\n🤖 [Miner] Agent chunk type: {type(content)}, content: {content}")
            #     await send({
            #         "type": "http.response.body",
            #         "body": content.encode('utf-8'),
            #         "more_body": True
            #     })
            #     await asyncio.sleep(0.25)

            # useful for debug

            async for event in graph.astream({"messages": messages}, version="v2"):
                # logger.info(f"mmmmmmmmmmmmmmmmmmmm {event} {type(event)}\n")
                for _, value in event.items():
                    message = value.get("messages", [])[-1].content
                    idx = 0
                    while idx < len(message):
                        chunk = message[idx:idx+10]
                        await send({
                            "type": "http.response.body",
                            "body": chunk.encode('utf-8'),
                            "more_body": True
                        })
                        await asyncio.sleep(0.25)
                        idx += 10
                
                # if "final_filter" in event:
                #     message = event["final_filter"].get("messages", [])[-1].content
                #     # logger.info(f"final message: {message}")
                #     idx = 0
                #     while idx < len(message):
                #         chunk = message[idx:idx+10]
                #         await send({
                #             "type": "http.response.body",
                #             "body": chunk.encode('utf-8'),
                #             "more_body": True
                #         })
                #         await asyncio.sleep(0.25)
                #         idx += 10
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False
            })

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

        cid_hashs = self.agent_manager.get_miner_agent().keys()
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

        model = os.environ.get("MINER_LLM_MODEL", "gpt-4o-mini")
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
        while True:
            try:
                await self.agent_manager.start(mode == "pull", role="miner", silent=silent)
                await asyncio.sleep(refresh_agents_interval)
                silent = True
            except Exception as e:
                logger.error(f"refresh_agents error: {e}")

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
        # Additional cleanup can be added here if needed
        logger.info("[Miner] Shutdown complete")
    except Exception as e:
        logger.error(f"[Miner] Unexpected error: {e}")
        raise


