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
import os
from pathlib import Path
import time
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from bittensor.core.stream import StreamingSynapse
from common.agent_manager import AgentManager
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, SyntheticNonStreamSynapse
from common.timer import Timer
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

        self.axon = bt.axon(
            wallet=self.settings.wallet, 
            port=self.settings.port,
            ip=self.settings.external_ip,
            external_ip=self.settings.external_ip,
            external_port=self.settings.port
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

        tasks = [
            asyncio.create_task(
                self.refresh_agents()
            ),
            asyncio.create_task(
                self.profile_tools_stats()
            )
        ]
        await asyncio.gather(*tasks)


    async def forward_synthetic_non_stream(self, task: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        log = logger.bind(source=task.dendrite.hotkey)
        project_id = task.project_id

        log.info(f"🤖 [Synthetic] - {task.id} Received {project_id} task: {task.question}")

        agent_config = self.agent_manager.get_miner_agent(project_id) or {}
        agent_graph = agent_config.get('agent_graph')
        counter = agent_config.get('counter')

        if not agent_graph:
            log.warning(f"[Synthetic] - {task.id} No agent found for project {project_id}")
            task.response = {"error": "No agent found"}
            return task

        with Timer(label=f"[Synthetic] - {task.id} Generating answer", log=log) as t:
            r = await agent_graph.ainvoke(
                {"messages": [{"role": "user", "content": task.question}]},
                config={"callbacks": [counter]}
            )
            # logger.info(f"Multi-agent response: {r}")
            response = r.get('messages')[-1].content
            t.response = response

        task.response = response
        # logger.info(f"Generated response: {synapse.response}")
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
        project_id = task.project_id
        user_messages = [msg for msg in task.completion.messages if msg.role == "user"]
        user_input = user_messages[-1].content

        log.info(f"🤖 [Organic] - {task.id} Received {project_id} task: {user_input}")

        agent_config = self.agent_manager.get_miner_agent(project_id) or {}
        agent = agent_config.get('agent_graph')
        counter = agent_config.get('counter')

        if not agent:
            log.warning(f"[Organic] - {task.id} No agent found for project {project_id}")
            task.response = {"error": "No agent found"}
            return task

        with Timer(label=f"[Organic] - {task.id} Generating answer", log=log):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"callbacks": [counter]}
            )
        task.response = response
        log.info(f"[Organic] - {task.id} Generated response: {task.model_dump_json()}")
        return task

    async def forward_capacity(self, synapse: CapacitySynapse) -> CapacitySynapse:
        logger.info(f"\n🤖 [Miner] Received capacity request")
        synapse.response = {
            "role": "miner",
            "capacity": {
                "projects": []
            }
        }
        return synapse

    async def invoke_graphql_agent(self, synapse: OrganicStreamSynapse) -> str:
        agent_config = self.agent_manager.get_miner_agent(synapse.project_id) or {}
        graphql_agent = agent_config.get('graphql_agent')
        response = await graphql_agent.query_no_stream(synapse.question)
        answer = response.get('messages')[-1].content
        return answer

    async def invoke_miner_agent(self, synapse: OrganicStreamSynapse) -> str:
        agent_config = self.agent_manager.get_miner_agent(synapse.project_id) or {}
        miner_agent = agent_config.get('agent_graph')
        response = await miner_agent.ainvoke(
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
            for project_id, config in self.agent_manager.get_miner_agent().items():
                counter = config.get('counter')
                logger.info(f"[MINER] Project {project_id} - Tool usage stats: {counter.stats()}")


if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())

    while True:
        asyncio.sleep(10)


