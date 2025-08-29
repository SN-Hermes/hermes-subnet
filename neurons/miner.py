# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
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
import time
import bittensor as bt
from loguru import logger
from ollama import AsyncClient
from bittensor.core.stream import StreamingSynapse

from common.protocol import CapacitySynapse, OrganicStreamSynapse, SyntheticNonStreamSynapse, SyntheticSynapse, SyntheticStreamSynapse
from herms.base import BaseNeuron
import agent.graphql_agent as subAgent


ollama_client = AsyncClient(host='http://localhost:11434')

def allow_all(synapse: CapacitySynapse) -> None:
    return None


class Miner(BaseNeuron):
    version: str = '4'

    axon: bt.Axon | None

    @property
    def role(self) -> str:
        return "miner2"

    def __init__(self):
        super().__init__()

    async def forward(self, synapse: SyntheticSynapse) -> SyntheticSynapse:
        logger.info(f"\n🤖 [Miner] Received question: {synapse.question}")

        message = {'role': 'user', 'content': synapse.question}
        iter = await ollama_client.chat(model='llama3.2', messages=[message], stream=True)
        async for part in iter:
            logger.info(f"\n🤖 [Miner] Agent: {part}")

        synapse.response = {"message": 'ok'}
        return synapse

    async def forward_stream(self, synapse: SyntheticStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        from starlette.types import Send
        logger.info(f"\n🤖 [Miner] Received stream question: {synapse.question}")

        message = {'role': 'user', 'content': synapse.question}
        async def token_streamer(send: Send):
            iter = await ollama_client.chat(model='llama3.2', messages=[message], stream=True)
            async for part in iter:
                logger.info(f"\n🤖 [Miner] Agent: {part}")
                text = part["message"]["content"] if "message" in part else str(part)
                await send({
                    "type": "http.response.body",
                    "body": text.encode("utf-8"),
                    "more_body": True
                })
                await asyncio.sleep(0.5)  # Simulate some delay
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False
            })

        return synapse.create_streaming_response(token_streamer)
    
    async def forward_stream_agent(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        from starlette.types import Send
        logger.info(f"\n🤖 [Miner] Received stream question22: {synapse.completion}")

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

    async def forward_non_stream_agent(self, synapse: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        logger.info(f"\n🤖 [Miner] Received non stream question22: {synapse}")
        response = await self.exampleAgent.query(synapse.question)
        synapse.response = response
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
            forward_fn=self.forward,
        )

        self.axon.attach(
            forward_fn=self.forward_stream,
        )

        self.axon.attach(
            forward_fn=self.forward_stream_agent,
        )

        self.axon.attach(
            forward_fn=self.forward_non_stream_agent
        )

        self.axon.attach(
            forward_fn=self.forward_capacity,
            verify_fn=allow_all
        )

        self.axon.serve(netuid=self.settings.netuid, subtensor=self.settings.subtensor)

        self.axon.start()
        logger.info(f"Miner starting at block: {self.settings.subtensor.block}")
        logger.info(f"Axon created: {self.axon}")
        logger.info(f"Miner starting at block: {self.settings.subtensor.block}")


if __name__ == "__main__":
    miner = Miner()
    asyncio.run(miner.start())

    while True:
        time.sleep(1)


