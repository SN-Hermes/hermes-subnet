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
import random
import traceback
import torch.multiprocessing as mp
import time
from typing import TYPE_CHECKING
from fastapi.responses import StreamingResponse
from loguru import logger
import uvicorn
from multiprocessing.synchronize import Event
from common.meta_config import MetaConfig
from common.table_formatter import table_formatter
from common.enums import ErrorCode, RoleFlag
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, ChatCompletionRequest, OrganicNonStreamSynapse, OrganicStreamSynapse
import common.utils as utils
from common.settings import settings
from hermes.validator.challenge_manager import ChallengeManager
from hermes.base import BaseNeuron

if TYPE_CHECKING:
    import bittensor as bt

ROLE = "validator"

settings.load_env_file(ROLE)
LOGGER_DIR = os.getenv("LOGGER_DIR", f"logs/{ROLE}")

HermesLogger.configure_loguru(
    file=f"{LOGGER_DIR}/hermes_validator.log",
    error_file=f"{LOGGER_DIR}/hermes_validator_error.log"
)

class Validator(BaseNeuron):
    dendrite: "bt.Dendrite"

    @property
    def role(self) -> str:
        return ROLE
    
    def __init__(self):
        super().__init__()
        # Import bittensor here to avoid multiprocessing spawn issues
        import bittensor as bt
        self.dendrite = bt.dendrite(wallet=self.settings.wallet)
        
        self.forward_miner_timeout = int(os.getenv("FORWARD_MINER_TIMEOUT", 60 * 3))  # seconds
        logger.info(f"Set forward miner timeout to {self.forward_miner_timeout} seconds")

    async def cleanup(self):
        """Clean up resources before shutdown"""
        try:
            if hasattr(self, 'dendrite') and self.dendrite:
                # Close dendrite session properly using bittensor's async close method
                await self.dendrite.aclose_session()
                logger.info("Closed dendrite HTTP session")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def run_challenge(
            self,
            organic_score_queue: list,
            ipc_synthetic_score: list,
            ipc_miners_dict: dict,
            synthetic_token_usage: list,
            ipc_meta_config: dict,
            ipc_common_config: dict,
            event_stop: Event,
    ):
        self.challenge_manager = ChallengeManager(
            settings=self.settings,
            save_project_dir=Path(__file__).parent.parent / "projects" / self.role,
            uid=self.uid,
            dendrite=self.dendrite,
            organic_score_queue=organic_score_queue,
            ipc_synthetic_score=ipc_synthetic_score,
            ipc_miners_dict=ipc_miners_dict,
            synthetic_token_usage=synthetic_token_usage,
            ipc_meta_config=ipc_meta_config,
            ipc_common_config=ipc_common_config,
            event_stop=event_stop,
            score_state_path=Path(self.settings.base_dir) / ".data" / f"{self.role}_score_state.pt",
            work_state_path=Path(self.settings.base_dir) / ".data" / f"{self.role}_workload_state.pt",
            v=self,
        )
        tasks = [
            asyncio.create_task(
                self.challenge_manager.start()
            ),
        ]
        await asyncio.gather(*tasks)

    async def run_api(
            self,
            organic_score_queue: list,
            ipc_miners_dict: dict[int, dict],
            ipc_synthetic_score: list,
            synthetic_token_usage: list,
            ipc_common_config: dict,
            ipc_meta_config: dict,
            event_stop: Event
        ):
        super().start(flag=RoleFlag.VALIDATOR)
        self.organic_score_queue = organic_score_queue
        self.ipc_miners_dict = ipc_miners_dict
        self.ipc_synthetic_score = ipc_synthetic_score
        self.synthetic_token_usage = synthetic_token_usage
        self.uid_select_count = defaultdict(int)
        self.ipc_common_config = ipc_common_config
        self.ipc_meta_config = ipc_meta_config

        # { cid_hash: [block_height, last_acquired_timestamp, node_type, endpoint] }
        self.block_cache: dict[str, list[int, int, str, str]] = {}
        try:
            from hermes.validator.api import app
            
            external_ip = self.settings.external_ip
            if not external_ip:
                logger.error("Failed to get external IP")
                event_stop.set()
                return

            logger.info(f"Starting serve API on http://{external_ip}:{self.settings.port}")
            logger.info(f"Stats at http://{external_ip}:{self.settings.port}/validator/stats")
            config = uvicorn.Config(
                app,
                host=external_ip,
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
            raise

    async def run_miner_checking(self, ipc_miners_dict: dict, event_stop: Event):
        import bittensor as bt

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
                    logger.debug(f"Checking uid: {uid} r.dendrite: {r.dendrite} r.axon: {r.axon}")
                    return {
                        "uid": uid,
                        "projects": r.response.get("capacity", {}).get("projects", []),
                        "hotkey": r.axon.hotkey
                    }
                else:
                    logger.debug(f"UID {uid} request failed.")
            except Exception as e:
                logger.debug(f"Failed to check availability for uid {uid}: {e}")

            return {
                "uid": uid,
                "projects": [],
                "hotkey": ""
            }

        while not event_stop.is_set():
            try:
                miner_uids, miner_hotkeys = self.settings.miners()
                all_miner_uids = []
                for uid, _ in zip(miner_uids, miner_hotkeys):
                    if uid == self.uid:
                        continue
                    all_miner_uids.append(uid)
                logger.debug(f"[CheckMiner] all_miner_uids: {all_miner_uids}, Current miners: {ipc_miners_dict}")

                tasks = []
                logger.debug(f"all_miner_uids: {all_miner_uids}")
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

                for r in responses:
                    ipc_miners_dict[r["uid"]] = {
                        "hotkey": r["hotkey"],
                        "projects": r["projects"]
                    }

            except Exception as e:
                logger.error(f"Error in miner checking: {e}")

            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("[CheckMiner] Shutting down gracefully...")
                break
        
        # Clean up resources before exiting
        await self.cleanup()

    async def forward_miner(self, body: ChatCompletionRequest):
        now = int(time.time())
        cid_hash = body.cid_hash
        block_height, last_acquired_timestamp, node_type, endpoint = self.block_cache.get(cid_hash, [0, 0, "", ""])
        if not block_height or abs(now - last_acquired_timestamp) > 3:
            if not endpoint:
                project_config = self.ipc_common_config.get(cid_hash, None)
                if project_config:
                    node_type = project_config["node_type"]
                    endpoint = project_config["endpoint"]

            if endpoint:
                latest_block = await utils.get_latest_block(endpoint, node_type)
                if latest_block is not None:
                    block_height = latest_block
                self.block_cache[cid_hash] = [block_height, now, node_type, endpoint]

        logger.info(f"[Organic] - {body.id} cid_hash: {cid_hash}, block_height: {block_height}, last_acquired_timestamp: {last_acquired_timestamp}, node_type: {node_type}, endpoint: {endpoint}")
        synapse = OrganicNonStreamSynapse(id=body.id, cid_hash=cid_hash, block_height=block_height or 0, completion=body)
        try:
            available_miners: list[int] = []
            for uid, info in self.ipc_miners_dict.items():
                projects = info.get("projects", [])
                if cid_hash in projects:
                    available_miners.append(uid)

            if len(available_miners) == 0:
                logger.error(f"[Organic] - {body.id} No available miners found for project {cid_hash}.")
                synapse.status_code = ErrorCode.ORGANIC_NO_AVAILABLE_MINERS.value
                synapse.error = "No available miners"
                return synapse

            synthetic_score: dict[int, tuple[float, str]] = self.ipc_synthetic_score[0] if self.ipc_synthetic_score else {}
            synthetic_counter: dict[int, tuple[int, int]] = self.ipc_synthetic_score[1] if self.ipc_synthetic_score else {}
            organic_success_rate_threshold = self.ipc_meta_config.get("organic_success_rate_threshold", 0)

            miner_uid, _ = utils.select_uid(
                organic_success_rate_threshold,
                synthetic_score,
                synthetic_counter,
                available_miners,
                self.uid_select_count
            )
            if not miner_uid:
                logger.error(f"[Organic] - {body.id} No miner selected for project {cid_hash}.")
                synapse.status_code = ErrorCode.ORGANIC_NO_SELECTED_MINER.value
                synapse.error = "No selected miner"
                return synapse

            logger.info(f"[Organic] - {body.id} Received organic request for project: {cid_hash}, block: {block_height}  body: {body}, forward to miner_uid: {miner_uid}")

            dd = self.dendrite
            if body.stream:
                before = time.perf_counter()

                async def streamer():
                    synapse = OrganicStreamSynapse(id=body.id, cid_hash=cid_hash, block_height=block_height or 0, completion=body)
                    response_generator = await dd.forward(
                        axons=self.settings.metagraph.axons[miner_uid],
                        synapse=synapse,
                        deserialize=False,
                        timeout=self.forward_miner_timeout,
                        streaming=True,
                    )
                    final_synapse = None
                    async for part in response_generator:
                        if isinstance(part, OrganicStreamSynapse):
                            final_synapse = part
                            break
                        else:
                            formatted_chunk = utils.format_openai_message(part)
                            yield f"{formatted_chunk}"
                    
                    if final_synapse:
                        final_synapse.elapsed_time = final_synapse.elapsed_time or utils.fix_float(time.perf_counter() - before)
                        if final_synapse.status_code == 200 and len(self.organic_score_queue) < 1000:
                            self.organic_score_queue.append((
                                miner_uid,
                                final_synapse.hotkey or self.settings.metagraph.axons[miner_uid].hotkey,
                                {
                                    "id": synapse.id,
                                    "cid_hash": synapse.cid_hash,
                                    "completion": body,
                                    "block_height": synapse.block_height,
                                    "response": final_synapse.response,
                                    "status_code": final_synapse.status_code,
                                    "error": final_synapse.error,
                                    "miner_model_name": final_synapse.miner_model_name,
                                    "graphql_agent_model_name": final_synapse.graphql_agent_model_name,
                                    "elapsed_time": final_synapse.elapsed_time,
                                    "usage_info": final_synapse.usage_info,
                                    "graphql_agent_inner_tool_calls": final_synapse.graphql_agent_inner_tool_calls,
                                    "dendrite": {
                                        "status_code": final_synapse.dendrite.status_code,
                                        "status_message": final_synapse.dendrite.status_message,
                                    }
                                }
                            ))
                        else:
                            logger.warning(f"[Organic-Stream] - {body.id} Not adding to queue. status_code={final_synapse.status_code}, response={final_synapse.response}")

                    yield f"{utils.format_openai_message('', finish_reason='stop')}"
                    yield f"data: [DONE]\n\n"

                return StreamingResponse(
                    streamer(), 
                    media_type="text/plain"
                )

            axons = self.settings.metagraph.axons[miner_uid]
            if not axons:
                logger.error(f"[Organic] - {body.id} No axons found for miner_uid: {miner_uid}")
                synapse.status_code = ErrorCode.ORGANIC_NO_AXON.value
                synapse.error = "No axon found"
                return synapse

            start_time = time.perf_counter()
            response: OrganicStreamSynapse = await self.dendrite.forward(
                axons=axons,
                synapse=synapse,
                deserialize=True,
                timeout=self.forward_miner_timeout,
            )

            elapsed_time = utils.fix_float(time.perf_counter() - start_time)
            response.elapsed_time = elapsed_time
            if not response.is_success:
                response.status_code = response.dendrite.status_code if response.dendrite is not None else ErrorCode.ORGANIC_ERROR_RESPONSE.value
                response.error = response.dendrite.status_message if response.dendrite is not None else "Unknown error from dendrite"

            if len(self.organic_score_queue) < 1000:
                logger.info(f"[Organic] - {body.id} organic_score_queue size: {len(self.organic_score_queue)}, is_success: {response.is_success}")
                if response.is_success and response.status_code == ErrorCode.SUCCESS.value:
                    self.organic_score_queue.append((miner_uid, axons.hotkey, response.dict()))
            table_formatter.create_organic_challenge_table(
                id=body.id,
                cid=cid_hash,
                question=synapse.get_question(),
                response=response,
            )
            return response
        
        except Exception as e:
            logger.error(f"[Validator] forward_miner error: {e}\n{traceback.format_exc()}")
            synapse.status_code = ErrorCode.ORGANIC_ERROR_RESPONSE.value
            synapse.error = str(e)
            return synapse

def run_challenge(
        organic_score_queue: list,
        ipc_synthetic_score: list,
        ipc_miners_dict: dict,
        synthetic_token_usage: list,
        ipc_meta_config: dict,
        ipc_common_config: dict,
        event_stop: Event
):
    proc = mp.current_process()
    HermesLogger.configure_loguru(
        file=f"{LOGGER_DIR}/{proc.name}.log",
        error_file=f"{LOGGER_DIR}/{proc.name}_error.log"
    )

    logger.info(f"run_challenge process id: {os.getpid()}")
    try:
        asyncio.run(Validator().run_challenge(
            organic_score_queue,
            ipc_synthetic_score,
            ipc_miners_dict,
            synthetic_token_usage,
            ipc_meta_config,
            ipc_common_config,
            event_stop
        ))
    except KeyboardInterrupt:
        logger.info("Challenge process received shutdown signal, exiting gracefully...")
    except Exception as e:
        logger.error(f"Challenge process error: {e}")
        raise

def run_api(
        organic_score_queue: list,
        ipc_miners_dict: dict,
        ipc_synthetic_score: list,
        synthetic_token_usage: list,
        ipc_meta_config: dict,
        ipc_common_config: dict,
        event_stop: Event
    ):
    proc = mp.current_process()
    HermesLogger.configure_loguru(
        file=f"{LOGGER_DIR}/{proc.name}.log",
        error_file=f"{LOGGER_DIR}/{proc.name}_error.log"
    )

    logger.info(f"run_api process id: {os.getpid()}")
    try:
        asyncio.run(Validator().run_api(
            organic_score_queue,
            ipc_miners_dict,
            ipc_synthetic_score,
            synthetic_token_usage,
            ipc_common_config=ipc_common_config,
            ipc_meta_config=ipc_meta_config,
            event_stop=event_stop
        ))
    except KeyboardInterrupt:
        logger.info("API process received shutdown signal, exiting gracefully...")
    except Exception as e:
        logger.error(f"API process error: {e}")
        raise

def run_miner_checking(ipc_miners_dict: dict, event_stop: Event):
    proc = mp.current_process()
    HermesLogger.configure_loguru(
        file=f"{LOGGER_DIR}/{proc.name}.log",
        error_file=f"{LOGGER_DIR}/{proc.name}_error.log"
    )

    logger.info(f"run_miner_checking process id: {os.getpid()}")
    try:
        asyncio.run(Validator().run_miner_checking(ipc_miners_dict, event_stop))
    except KeyboardInterrupt:
        logger.info("MinerChecking process received shutdown signal, exiting gracefully...")
    except Exception as e:
        logger.error(f"MinerChecking process error: {e}")
        raise

async def main():
    with mp.Manager() as manager:
        try:
            organic_score_queue = manager.list([])
            ipc_miners_dict = manager.dict({})
            ipc_synthetic_score = manager.list([{}, {}])
            synthetic_token_usage = manager.list([])
            ipc_meta_config = manager.dict({})
            ipc_common_config = manager.dict({})

            processes: list[mp.Process] = []
            event_stop = mp.Event()
        
            challenge_process = mp.Process(
                target=run_challenge,
                args=(
                    organic_score_queue,
                    ipc_synthetic_score,
                    ipc_miners_dict,
                    synthetic_token_usage,
                    ipc_meta_config,
                    ipc_common_config,
                    event_stop
                ),
                name="ChallengeProcess",
                daemon=True,
            )
            challenge_process.start()
            processes.append(challenge_process)

            api_process = mp.Process(
                target=run_api,
                args=(
                    organic_score_queue,
                    ipc_miners_dict,
                    ipc_synthetic_score,
                    synthetic_token_usage,
                    ipc_meta_config,
                    ipc_common_config,
                    event_stop
                ),
                name="APIProcess",
                daemon=True,
            )
            api_process.start()
            processes.append(api_process)

            miner_checking_process = mp.Process(
                target=run_miner_checking,
                args=(ipc_miners_dict, event_stop),
                name="MinerCheckingProcess",
                daemon=True,
            )
            miner_checking_process.start()
            processes.append(miner_checking_process)

            meta = MetaConfig()
            logger.info(f"main process id: {os.getpid()}")
            while not event_stop.is_set():
                try:
                    new_meta = await meta.pull()
                    logger.debug(f"Pulled new meta config: {new_meta}")
                    if new_meta.data:
                        new_min_latency_improvement_ratio = new_meta.data.get("min_latency_improvement_ratio", 0.2)
                        new_benchmark_mode = new_meta.data.get("benchmark_mode", "sample")
                        new_benchmark_sample_rate = new_meta.data.get("benchmark_sample_rate", 0.8)
                        new_benchmark_batch_size = new_meta.data.get("benchmark_batch_size", 0)

                        new_organic_success_score_threshold = new_meta.data.get("organic_success_score_threshold", 5)
                        new_organic_success_rate_threshold = new_meta.data.get("organic_success_rate_threshold", 0.7)

                        if new_min_latency_improvement_ratio != ipc_meta_config.get("min_latency_improvement_ratio", 0.2):
                            ipc_meta_config.update({
                                "min_latency_improvement_ratio": new_min_latency_improvement_ratio
                            })
                            logger.info(f"Updating min_latency_improvement_ratio from {ipc_meta_config.get('min_latency_improvement_ratio', 0.2)} to {new_min_latency_improvement_ratio}")

                        if new_benchmark_mode != ipc_meta_config.get("benchmark_mode", "sample"):
                            ipc_meta_config.update({
                                "benchmark_mode": new_benchmark_mode
                            })
                            logger.info(f"Updating benchmark_mode from {ipc_meta_config.get('benchmark_mode', 'sample')} to {new_benchmark_mode}")

                        if new_benchmark_sample_rate != ipc_meta_config.get("benchmark_sample_rate", 0.1):
                            ipc_meta_config.update({
                                "benchmark_sample_rate": new_benchmark_sample_rate
                            })
                            logger.info(f"Updating benchmark_sample_rate from {ipc_meta_config.get('benchmark_sample_rate', 0.1)} to {new_benchmark_sample_rate}")

                        if new_benchmark_batch_size != ipc_meta_config.get("benchmark_batch_size", 0):
                            ipc_meta_config.update({
                                "benchmark_batch_size": new_benchmark_batch_size
                            })
                            logger.info(f"Updating benchmark_batch_size from {ipc_meta_config.get('benchmark_batch_size', 0)} to {new_benchmark_batch_size}")


                        if new_organic_success_score_threshold != ipc_meta_config.get("organic_success_score_threshold", 0):
                            ipc_meta_config.update({
                                "organic_success_score_threshold": new_organic_success_score_threshold
                            })
                            logger.info(f"Updating organic_success_score_threshold from {ipc_meta_config.get('organic_success_score_threshold', 0)} to {new_organic_success_score_threshold}")

                        if new_organic_success_rate_threshold != ipc_meta_config.get("organic_success_rate_threshold", 0):
                            ipc_meta_config.update({
                                "organic_success_rate_threshold": new_organic_success_rate_threshold
                            })
                            logger.info(f"Updating organic_success_rate_threshold from {ipc_meta_config.get('organic_success_rate_threshold', 0)} to {new_organic_success_rate_threshold}")

                except Exception as e:
                    logger.error(f"Failed to refresh meta config: {e}")
                
                try:
                    await asyncio.sleep(5 * 60 + random.randint(0, 30))
                except asyncio.CancelledError:
                    logger.info("Meta config refresh task cancelled, shutting down...")
                    raise KeyboardInterrupt()  # Trigger graceful shutdown

        except KeyboardInterrupt:
            event_stop.set()
            logger.info("KeyboardInterrupt detected. Shutting down gracefully...")
            
            # Give processes time to shutdown gracefully
            for p in processes:
                logger.info(f"Waiting for {p.name} to finish...")
                p.join(timeout=5)
                
            # Terminate processes that didn't finish
            for p in processes:
                if p.is_alive():
                    logger.warning(f"{p.name} still alive, terminating...")
                    p.terminate()
                    p.join(timeout=2)
                    
            # Force kill if still alive
            for p in processes:
                if p.is_alive():
                    logger.error(f"{p.name} still alive after terminate, killing...")
                    p.kill()
                    p.join()

        except Exception as e:
            logger.error(f"Main loop error: {e}")
            event_stop.set()
            raise

        finally:
            logger.info("Cleaning up processes...")
            # Ensure all processes are terminated
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=1)
                
            utils.kill_process_group()

if __name__ == "__main__":
    try:
        os.setpgrp()
    except BaseException:
        logger.warning("Failed to set process group.")

    asyncio.run(main())




