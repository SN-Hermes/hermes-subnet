# The MIT License (MIT)
# Copyright © 2025 Subquery

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
from collections import defaultdict
import os
from pathlib import Path
import traceback
import torch.multiprocessing as mp
import time
from fastapi.responses import StreamingResponse
from loguru import logger
import bittensor as bt
import uvicorn
from multiprocessing.synchronize import Event
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, ChatCompletionRequest, OrganicNonStreamSynapse, OrganicStreamSynapse
import common.utils as utils
from hermes.validator.challenge_manager import ChallengeManager
from hermes.base import BaseNeuron

HermesLogger.configure_loguru(file=f"logs/hermes_validator.log")

class Validator(BaseNeuron):
    dendrite: bt.Dendrite

    @property
    def role(self) -> str:
        return "validator"
    
    def __init__(self):
        super().__init__()
        self.dendrite = bt.dendrite(wallet=self.settings.wallet)
        
        self.forward_miner_timeout = int(os.getenv("FORWARD_MINER_TIMEOUT", 60 * 3))  # seconds
        logger.info(f"Set forward miner timeout to {self.forward_miner_timeout} seconds")

    async def run_challenge(self, organic_score_queue: list, synthetic_score: list, event_stop: Event):
        self.challenge_manager = ChallengeManager(
            settings=self.settings,
            save_project_dir=Path(__file__).parent.parent / "projects" / self.role,
            uid=self.uid,
            dendrite=self.dendrite,
            organic_score_queue=organic_score_queue,
            synthetic_score=synthetic_score,
            event_stop=event_stop
        )
        tasks = [
            asyncio.create_task(
                self.challenge_manager.start()
            ),
        ]
        await asyncio.gather(*tasks)

    async def run_api(self, organic_score_queue: list, miners_dict: dict[int, dict], synthetic_score: list):
        super().start()
        self.organic_score_queue = organic_score_queue
        self.miners_dict = miners_dict
        self.synthetic_score = synthetic_score
        self.uid_select_count = defaultdict(int)

        try:
            from hermes.validator.api import app

            external_ip = utils.try_get_external_ip()
            logger.info(f"external_ip: {external_ip}")

            logger.info(f"Starting serve API on http://0.0.0.0:{self.settings.port}")
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.settings.port,
                loop="asyncio",
                reload=False,
                log_config=None,  # Disable uvicorn's default logging config
                access_log=False,  # Disable access logs to reduce noise
            )
            app.state.validator = self

            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Failed to serve API: {e}")

    async def run_miner_checking(self, miners_dict: dict):

        async def handle_availability(
            metagraph: "bt.Metagraph",
            dendrite: "bt.Dendrite",
            uid: int,
        ) -> dict[str, any]:
            try:
                synapse = CapacitySynapse()
                r = await dendrite.forward(
                    axons=metagraph.axons[uid],
                    synapse=synapse,
                    deserialize=True,
                    timeout=30,
                )
                if r.is_success and r.response.get("role", "") == "miner":
                    return {
                        "uid": uid,
                        "projects": r.response.get("capacity", {}).get("projects", []),
                        "hotkey": r.dendrite.hotkey
                    }
            except Exception:
                return None

        while True:
            try:
                miner_uids, miner_hotkeys = self.settings.miners()
                all_miner_uids = []
                for uid, _ in zip(miner_uids, miner_hotkeys):
                    if uid == self.uid:
                        continue
                    all_miner_uids.append(uid)
                logger.debug(f"[CheckMiner] Current miners: {miners_dict}")

                tasks = []
                for uid in all_miner_uids:
                    tasks.append(
                        asyncio.create_task(
                            handle_availability(
                                self.settings.metagraph,
                                self.dendrite,
                                uid,
                            )
                        )
                    )
                responses: list[any] = await asyncio.gather(*tasks)

                # Filter out None responses
                responses = [res for res in responses if res is not None]
                logger.debug(f"[CheckMiner] Miner availability responses: {responses}")

                for r in responses:
                    miners_dict[r["uid"]] = {
                        "hotkey": r["hotkey"],
                        "projects": r["projects"]
                    }

            except Exception as e:
                logger.error(f"Error in miner checking: {e}")

            await asyncio.sleep(30)

    async def forward_miner(self, cid: str, body: ChatCompletionRequest):
        try:
            available_miners = []
            for uid, info in self.miners_dict.items():
                projects = info.get("projects", [])
                if cid in projects:
                    available_miners.append(uid)
            if len(available_miners) == 0:
                logger.warning(f"No available miners found for project {cid}. Falling back to any miner.")
                return

            synthetic_score: dict[int, tuple[float, str]] = self.synthetic_score[0] if self.synthetic_score else {}
            miner_uid, _ = utils.select_uid(synthetic_score, available_miners, self.uid_select_count)
            if not miner_uid:
                logger.error(f"[Validator] No miner selected for project {cid}.")
                synapse = OrganicNonStreamSynapse(id=body.id, project_id=cid, completion=body)
                synapse.response = {"error": "No miner available"}
                return synapse

            if body.stream:
                async def streamer():
                    synapse = OrganicStreamSynapse(project_id=cid, completion=body)
                    responses = await self.dendrite.forward(
                        axons=self.settings.metagraph.axons[miner_uid],
                        synapse=synapse,
                        deserialize=True,
                        timeout=self.forward_miner_timeout,
                        streaming=True,
                    )
                    async for part in responses:
                        # logger.info(f"V3 got part: {part}, type: {type(part)}")
                        yield part
                return StreamingResponse(streamer(), media_type="text/plain")

            synapse = OrganicNonStreamSynapse(id=body.id, project_id=cid, completion=body)
            axons = self.settings.metagraph.axons[miner_uid]
            if not axons:
                logger.error(f"[Validator] organic task({body.id}) No axons found for miner_uid: {miner_uid}")
                synapse.response = {"error": "No axons found"}
                return synapse

            logger.info(f"[Validator] Received organic task({body.id}) cid: {cid}, body: {body}, forward to miner_uid: {miner_uid}({axons.hotkey})")

            start_time = time.perf_counter()
            response = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                deserialize=True,
                timeout=self.forward_miner_timeout,
            )
            elapsed_time = time.perf_counter() - start_time
            response.elapsed_time = elapsed_time

            logger.info(f"[Validator] organic task({body.id}), miner response: {response}")

            # logger.info(f'----{response.dendrite.status_code}')
            # logger.info(f'----{response.dendrite.status_message}')
            if len(self.organic_score_queue) < 1000:
                logger.debug(f"[Validator] organic_score_queue size: {len(self.organic_score_queue)}, is_success: {response.is_success}")
                if response.is_success:
                    self.organic_score_queue.append((miner_uid, axons.hotkey, response.dict()))
            return response
        
        except Exception as e:
            logger.error(f"[Validator] forward_miner error: {e} {traceback.format_exc()}")
            synapse = OrganicNonStreamSynapse(id=body.id, project_id=cid, completion=body)
            synapse.response = {"error": str(e)}
            return synapse

def run_challenge(organic_score_queue: list, synthetic_score: list, event_stop: Event):
    proc = mp.current_process()
    HermesLogger.configure_loguru(file=f"logs/{proc.name}.log")

    logger.info(f"run_challenge process id: {os.getpid()}")
    asyncio.run(Validator().run_challenge(organic_score_queue, synthetic_score, event_stop))

def run_api(organic_score_queue: list, miners_dict: dict, synthetic_score: list):
    proc = mp.current_process()
    HermesLogger.configure_loguru(file=f"logs/{proc.name}.log")

    logger.info(f"run_api process id: {os.getpid()}")
    asyncio.run(Validator().run_api(organic_score_queue, miners_dict, synthetic_score))

def run_miner_checking(miners_dict: dict):
    proc = mp.current_process()
    HermesLogger.configure_loguru(file=f"logs/{proc.name}.log")

    logger.info(f"run_miner_checking process id: {os.getpid()}")
    asyncio.run(Validator().run_miner_checking(miners_dict))

async def main():
    with mp.Manager() as manager:
        try:
            organic_score_queue = manager.list([])
            miners_dict = manager.dict({})
            synthetic_score = manager.list([{}])

            processes: list[mp.Process] = []
            event_stop = mp.Event()
        
            challenge_process = mp.Process(
                target=run_challenge,
                args=(organic_score_queue, synthetic_score, event_stop),
                name="ChallengeProcess",
                daemon=True,
            )
            challenge_process.start()
            processes.append(challenge_process)

            api_process = mp.Process(
                target=run_api,
                args=(organic_score_queue, miners_dict, synthetic_score),
                name="APIProcess",
                daemon=True,
            )
            api_process.start()
            processes.append(api_process)

            miner_checking_process = mp.Process(
                target=run_miner_checking,
                args=(miners_dict,),
                name="MinerCheckingProcess",
                daemon=True,
            )
            miner_checking_process.start()
            processes.append(miner_checking_process)

            logger.info(f"main process id: {os.getpid()}")
            while True:
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            event_stop.set()
            logger.info("KeyboardInterrupt detected. Shutting down gracefully...")

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            raise

        finally:
            utils.kill_process_group()

if __name__ == "__main__":
    try:
        os.setpgrp()
    except BaseException:
        logger.warning("Failed to set process group.")

    asyncio.run(main())




