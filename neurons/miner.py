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
from typing import Dict
import bittensor as bt
from loguru import logger
from bittensor.core.stream import StreamingSynapse
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, MessagesState, START, END

from agent.agent_zoo import AgentZoo
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, SyntheticNonStreamSynapse, SyntheticSynapse, SyntheticStreamSynapse
from common.timer import Timer
from hermes.base import BaseNeuron

import agent.graphql_agent as subAgent
from common.project_manager import ProjectManager


SUBQL_CID = 'QmfUNJC1Qz8m3F67sQmxrwjuSAu4WaCR1iBdPPdzBruQ7P'

def allow_all(synapse: CapacitySynapse) -> None:
    return None

class Miner(BaseNeuron):
    version: str = '5'
    axon: bt.Axon | None
    agents:  Dict[str, Dict[str, Dict[str, str] | CompiledStateGraph]] | None
    multi_agent_graph: CompiledStateGraph | None

    @property
    def role(self) -> str:
        return "miner"

    def __init__(self):
        super().__init__()
        
        # Configure loguru to intercept and control third-party logging  
        import common.utils as utils
        utils.configure_loguru()
        
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
        logger.info(f"\n🤖 [Miner] Received synthetic: {synapse}")
        projectId = synapse.projectId

        agent_config = self.agents.get(projectId, {})
        agent = agent_config.get('agent')
        counter = agent_config.get('counter')

        if not agent:
            logger.warning(f"[MINER] No agent found for project {projectId}")
            synapse.response = {"error": "No agent found"}
            return synapse

        with Timer(label=f"[Miner] Generating answer for task: {synapse.id}") as t:
            # response = await self.server_agent.query_no_stream(
            #     synapse.question,
            # )
            # response = response.get('messages', [])[-1].content

            # response = await agent.ainvoke(
            #     {"messages": [{"role": "user", "content": synapse.question}]},
            #     config={"callbacks": [counter]}
            # )
            # response = response.get('messages')[-1].content
            r = await self.multi_agent_graph.ainvoke(
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
        projectId = synapse.projectId

        agent_config = self.agents.get(projectId, {})
        agent = agent_config.get('agent')
        counter = agent_config.get('counter')

        if not agent:
            logger.warning(f"[MINER] No agent found for project {projectId}")
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

    async def refresh_agents(self):
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent / "projects" / self.role
        pm = ProjectManager(project_dir)
        await pm.pull()

        self.agents = AgentZoo.load_agents(project_dir)

        self.server_agent = subAgent.initServerAgentWithConfig(pm.get_project(SUBQL_CID))
        self.miner_agent = self.agents.get(SUBQL_CID).get('agent')

        # counter = self.agents.get('QmQqqmwwaBben8ncfHo3DMnDxyWFk5QcEdTmbevzKj7DBd').get('counter')
        # miner_agent_with_counter = self.miner_agent.with_config({"callbacks": [counter]})

        def miner_router(state):
            # logger.info(f'----min router---{state}')
            messages = state["messages"]
            for m in messages:
                # TODO: check tool call has real output
                if m.type == 'ai' and len(m.tool_calls) > 0:
                    return END
            
            # TODO: trim
            first_message = messages[0:1]
            # return {"next": "server_agent", "state_update": {"messages": first_message}}
            return "server_agent"
        

        builder = StateGraph(MessagesState)
        builder.add_node("miner_agent", self.miner_agent)
        builder.add_node("server_agent", self.server_agent.executor)
        builder.add_conditional_edges(
            "miner_agent",
            miner_router
        )
        builder.add_edge(START, "miner_agent")
        self.multi_agent_graph = builder.compile()

        while True:
            await asyncio.sleep(30 * 1)
            # TODO: reconstruct multi_agent_graph
            # self.agents = AgentZoo.load_agents(project_dir)

    async def profile_tools_stats(self):
        while True:
            await asyncio.sleep(60 * 1)
            agents = self.agents
    
            for projectId, config in agents.items():
                counter = config.get('counter')
                logger.info(f"[MINER] Project {projectId} - Tool usage stats: {counter.stats()}")
    
if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())

    while True:
        time.sleep(1)


