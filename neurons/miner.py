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
from agent.stats import Metrics
from common.table_formatter import table_formatter
from common.agent_manager import AgentManager
from common.errors import ErrorCode
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, StatsMiddleware, SyntheticNonStreamSynapse
from common.sqlite_manager import SQLiteManager
import common.utils as utils
from hermes.base import BaseNeuron
import agent.graphql_agent as subAgent



class Miner(BaseNeuron):

    @property
    def role(self) -> str:
        return "miner"

    def __init__(self, config_loguru: bool = True):
        if config_loguru:
            HermesLogger.configure_loguru(file=f"logs/{self.role}.log")
        super().__init__()

    async def start(self):
        super().start()

        self.metrics = Metrics()
        
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
            metrics=self.metrics
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

        tasks = [
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

        await asyncio.gather(*tasks)

    async def db_writer(self):
        last_check_time = 0

        while True:
            if int(time.time()) - last_check_time > 60 * 10:
                self.sqlite_manager.cleanup_old_records()
                last_check_time = int(time.time())

            item = await self.db_queue.get()

            type = item.get("type")
            status_code = item.get("status_code")
            project_id = item.get("project_id")

            target = self.metrics.synthetic_project_usage if type == 0 else self.metrics.organic_project_usage
            target.incr(
                project_id, 
                success=False if status_code != 200 else True
            )

            tool_hit = item.get("tool_hit")

            logger.info(f"[DB Writer] - Inserting request log for project {project_id} with status code {status_code}, type:{type}, tool_hit: {tool_hit}")

            if tool_hit and tool_hit != '[]':
                tool_hit_list = json.loads(tool_hit)
                target = self.metrics.synthetic_tool_usage if type == 0 else self.metrics.organic_tool_usage
                for tool_name, count in tool_hit_list:
                    target.incr(tool_name, count)

            self.sqlite_manager.insert_request(**item)
            self.db_queue.task_done()

    async def _handle_task(
            self,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse,
            log: Logger,
    ) -> SyntheticNonStreamSynapse | OrganicNonStreamSynapse:
        tag = "Synthetic"
        type = 0
        question = task.get_question()

        if isinstance(task, OrganicNonStreamSynapse):
            tag = "Organic"
            type = 1

        project_id = task.project_id
        agent_graph, _, graphql_agent = self.agent_manager.get_miner_agent(project_id)

        tool_hit = []
        answer = None
        response = None
        error = None
        status_code = ErrorCode.SUCCESS

        exclude_tools = [t.name for t in graphql_agent.tools]
        before = time.perf_counter()

        try:
            if not agent_graph:
                log.warning(f"[{tag}] - {task.id} No agent found for project {project_id}")
                error = f"No agent found for project {project_id}"
                status_code = ErrorCode.AGENT_NOT_FOUND
            else:
                r = await agent_graph.ainvoke(
                    {"messages": [{"role": "user", "content": question}]},
                    # config={"callbacks": [counter]}
                )

                # check tool stats
                tool_hit = utils.try_get_tool_hit(
                    r.get('messages', []),
                    # exclude_tools=exclude_tools
                )

                answer = r.get('messages')[-1].content or None
                if not answer:
                    error = utils.try_get_invalid_tool_messages(r.get('messages', []))
                    status_code = ErrorCode.TOOL_ERROR if error is not None else status_code

                if status_code == ErrorCode.SUCCESS:
                    response = r if type == 1 else answer

        except Exception as e:
            # log.error(f"[Synthetic] - {task.id} Agent error: {e}")
            error = str(e)
            status_code = ErrorCode.INTERNAL_SERVER_ERROR

        elapsed = time.perf_counter() - before
        
        tool_hit_names = [t[0] for t in tool_hit]
        rows = [f"💬 Answer: {answer}\n"]
        if error:
            rows.append(f"⚠️ {status_code.value} | {error}\n")
        if len(tool_hit_names) > 0:
            rows.append(f"🛠️ Tools Hit: {', '.join(tool_hit_names)}\n")
        rows.append(f"⏱️ Cost: {elapsed:.4f}s")
        
        status_icon = "✅" if status_code == ErrorCode.SUCCESS else "❌"
        output = table_formatter.create_single_column_multiple_row_table(
            f"🤖 {status_icon} {tag}: {question} ({task.id})",
            rows,
        )
        log.info(f"\n{output}")

        task.response = response
        task.error = error
        task.status_code = status_code
        self.db_queue.put_nowait({
            "type": type,
            "source": task.dendrite.hotkey,
            "task_id": task.id,
            "project_id": task.project_id,
            "cid": task.project_id,
            "request_data": question,
            "response_data": answer if status_code == ErrorCode.SUCCESS else task.error,
            "status_code": task.status_code,
            "tool_hit": json.dumps(tool_hit),
            "cost": elapsed,
        })
        return task

    async def forward_synthetic_non_stream(self, task: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        log = logger.bind(source=task.dendrite.hotkey)
        await self._handle_task(task, log)
        return task

    async def forward_organic_stream(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        from starlette.types import Send
        logger.info(f"\n🤖 [Miner] Received organic stream: {synapse.completion}")

        user_messages = [msg for msg in synapse.completion.messages if msg.role == "user"]
        user_input = user_messages[-1].content

        async def token_streamer(send: Send):
            iter = subAgent.stream_chat_completion(self.serverAgent, user_input, synapse.completion)
            async for part in iter:
                logger.info(f"\n🤖 [Miner] Agent: {part}")
                await send({
                    "type": "http.response.body",
                    "body": part,
                    "more_body": True
                })
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
        logger.info(f"[Miner] Received capacity request")
        if not self.agent_manager:
            logger.warning(f"[Miner] No agent manager found")
            synapse.response = {
                "role": "miner",
                "capacity": {
                    "projects": []
                }
            }
            return synapse

        cids = self.agent_manager.get_miner_agent().keys()
        synapse.response = {
            "role": "miner",
            "capacity": {
                "projects": list(cids)
            }
        }
        return synapse

    async def invoke_graphql_agent(self, synapse: SyntheticNonStreamSynapse) -> str:
        _, _, graphql_agent = self.agent_manager.get_miner_agent(synapse.project_id)
        response = await graphql_agent.query_no_stream(synapse.question)
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
        await self.agent_manager.start(mode == "pull", role="miner")

        # while True:
        #     await asyncio.sleep(30 * 1)
        #     # TODO: reconstruct multi_agent_graph
        #     self.agents = AgentZoo.load_agents(project_dir)

    async def profile_tools_stats(self):
        while True:
            await asyncio.sleep(60 * 1)
            logger.info(f"[MINER] usage stats: {json.dumps(self.metrics.stats())}")


if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())

    while True:
        asyncio.sleep(60 * 2)


