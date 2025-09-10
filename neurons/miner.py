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
from pathlib import Path
import time
from typing import Any, Dict
import bittensor as bt
from loguru import logger
from bittensor.core.stream import StreamingSynapse
from langgraph.graph.state import CompiledStateGraph
from agent.agent_zoo import AgentZoo
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, SyntheticNonStreamSynapse, SyntheticSynapse, SyntheticStreamSynapse
from common.timer import Timer
from hermes.base import BaseNeuron
import agent.graphql_agent as subAgent
from common.project_manager import ProjectManager


class Miner(BaseNeuron):
    version: str = '5'
    axon: bt.Axon | None
    agents:  Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph]] | None

    @property
    def role(self) -> str:
        return "miner"

    def __init__(self):
        super().__init__(file=f"logs/{self.role}/{time.strftime('%Y-%m-%d')}.log")
        self.agents = {}


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

    async def forward_synthetic_non_stream(self, synapse: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        log = logger.bind(source=synapse.dendrite.hotkey, synthetic_id=synapse.id)

        log.info(f"🤖 Received synthetic: {synapse.question}")
        project_id = synapse.project_id

        agent_config = self.agents.get(project_id, {})
        agent_graph = agent_config.get('agent_graph')
        counter = agent_config.get('counter')

        if not agent_graph:
            log.warning(f"No agent found for project {project_id}")
            synapse.response = {"error": "No agent found"}
            return synapse

        with Timer(label=f"Generating answer for task: {synapse.id}", log=log) as t:
            r = await agent_graph.ainvoke(
                {"messages": [{"role": "user", "content": synapse.question}]},
                config={"callbacks": [counter]}
            )
            # logger.info(f"Multi-agent response: {r}")
            response = r.get('messages')[-1].content
            t.response = response

        synapse.response = response
        # logger.info(f"Generated response: {synapse.response}")
        return synapse

    async def forward_organic_non_stream(self, synapse: OrganicNonStreamSynapse) -> OrganicNonStreamSynapse:
        logger.info(f"\n🤖 [Miner] Received organic non stream: {synapse}")
        project_id = synapse.project_id

        agent_config = self.agents.get(project_id, {})
        agent = agent_config.get('agent')
        counter = agent_config.get('counter')

        if not agent:
            logger.warning(f"[MINER] No agent found for project {project_id}")
            synapse.response = {"error": "No agent found"}
            return synapse

        user_messages = [msg for msg in synapse.completion.messages if msg.role == "user"]
        user_input = user_messages[-1].content

        with Timer(label=f"Generating query for task: {synapse.model_dump_json()}"):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"callbacks": [counter]}
            )
        synapse.response = response
        logger.info(f"Generated response: {synapse.model_dump_json()}")
        return synapse

    async def forward_capacity(self, synapse: CapacitySynapse) -> CapacitySynapse:
        logger.info(f"\n🤖 [Miner] Received capacity request")
        synapse.response = {
            "role": "miner",
            "capacity": {
                "projects": []
            }
        }
        return synapse

    async def invoke_server_agent(self, synapse: OrganicStreamSynapse) -> str:
        agent_config = self.agents.get(synapse.project_id, {})
        server_agent = agent_config.get('server_agent')
        response = await server_agent.query_no_stream(synapse.question)
        answer = response.get('messages')[-1].content
        return answer

    async def invoke_miner_agent(self, synapse: OrganicStreamSynapse) -> str:
        agent_config = self.agents.get(synapse.project_id, {})
        miner_agent = agent_config.get('miner_agent')
        response = await miner_agent.ainvoke(
            {"messages": [{"role": "user", "content": synapse.question}]}
        )
        answer = response.get('messages')[-1].content
        return answer

    def load_agents(self) -> Dict[str, Any]:
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent / "projects" / self.role
        pm = ProjectManager(project_dir)
        projects = pm.load()
        return projects

    async def refresh_agents(self, pull=True):
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent / "projects" / self.role
        pm = ProjectManager(project_dir)
        
        if pull:
            await pm.pull()
        else:
            pm.load()
        
        self.agents = AgentZoo.load_agents(project_dir)

        return self.agents
        # while True:
        #     await asyncio.sleep(30 * 1)
        #     # TODO: reconstruct multi_agent_graph
        #     self.agents = AgentZoo.load_agents(project_dir)

    async def profile_tools_stats(self):
        while True:
            await asyncio.sleep(60 * 1)
            agents = self.agents
    
            for project_id, config in agents.items():
                counter = config.get('counter')
                logger.info(f"[MINER] Project {project_id} - Tool usage stats: {counter.stats()}")
    
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())

    while True:
        time.sleep(1)


