import asyncio
import signal
import sys
import bittensor as bt
from loguru import logger

from common import utils
from common.enums import ProjectPhase
from common.mock_config import MockConfigSharedMemory
from common.settings import Settings
from common.table_formatter import table_formatter
from common.logger import HermesLogger
from hermes.validator.challenge_manager import ChallengeManager
from hermes.validator.multiprocess_query import query_single_miner

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    logger.warning("[MockValidator] Received interrupt signal, shutting down gracefully...")
    shutdown_event.set()

async def run_mock_validator():
    # Configure logging
    HermesLogger.configure_loguru(
        file="logs/mock_validator/mock_validator.log",
        error_file="logs/mock_validator/mock_validator_error.log"
    )
    
    logger.info("[MockValidator] Starting mock validator process...")
    logger.info("[MockValidator] Waiting for configuration from miner process...")
    
    # Read configuration from shared memory
    mock_config = MockConfigSharedMemory()
    config = None
    
    try:
        config = mock_config.wait_for_config(shutdown_event=shutdown_event, timeout=60 * 5)
        
        if config is None:
            logger.error("[MockValidator] Failed to read configuration from shared memory.")
            logger.error("[MockValidator] Make sure the miner is running with mock mode enabled.")
            return
        
        logger.info(f"[MockValidator] Configuration received from shared memory")
        logger.debug(f"[MockValidator] Config: {config}")
        
    finally:
        # Close connection immediately after reading config
        # Note: unlink=False because validator is not the owner (miner is)
        mock_config.cleanup(unlink=False)
        logger.info("[MockValidator] Shared memory connection closed")
    
    # Continue with the config (shared memory already closed)
    if config is None:
        return
    
    miner_uid = config.get("uid")
    miner_ip = config.get("external_ip")
    miner_port = config.get("port")
    miner_project_dir = config.get("miner_project_dir")
    env_file = config.get("env_file")

    settings = Settings.from_env_file(env_file)

    challenge_manager = ChallengeManager(
        settings=settings,
        save_project_dir=miner_project_dir,
        uid=1000,
        dendrite=bt.dendrite(wallet=settings.wallet),
        organic_score_queue=[],
        ipc_synthetic_score=[],
        ipc_miners_dict={},
    )
    await challenge_manager.agent_manager.start(pull=False, role="validator")

    round_id = 1
    block_cache: dict[str, int] = {}
    project_score_matrix = []

    logger.info("[MockValidator] Entering main loop to pull and validate challenges")
    while not shutdown_event.is_set():
        await asyncio.sleep(5)
        
        if shutdown_event.is_set():
            logger.info("[MockValidator] Shutdown requested, exiting loop")
            break
            
        challenges = []
        try:
            challenges = await challenge_manager.agent_manager.project_manager.pull_mock_challenges()
        except Exception as e:
            logger.error(f"[MockValidator] Error pulling challenges: {e}")

        logger.info(f"[MockValidator] Found {len(challenges)} challenges")

        for c in challenges:
            if shutdown_event.is_set():
                logger.info("[MockValidator] Shutdown requested during challenge processing")
                break
            cid_hash = c.cid_hash
            validator_agent = challenge_manager.agent_manager.get_agent(cid_hash)

            logger.info(f"[MockValidator] start challenge {cid_hash} - round {round_id}...")
            if not validator_agent:
                logger.error(f"[MockValidator] No validator agent found for challenge {cid_hash}")
                continue

            # generate ground truth
            block_height = block_cache.get(cid_hash, None)
            if block_height is None:
                latest_block = await utils.get_latest_block(validator_agent.config.endpoint, validator_agent.config.node_type)
                if latest_block is not None:
                    block_cache[cid_hash] = latest_block - 1000
                    block_height = block_cache[cid_hash]

            if shutdown_event.is_set():
                logger.warning("[MockValidator] Shutdown event triggered, stopping acquiring block height...")
                return

            if block_height is None:
                logger.error(f"[MockValidator] Unable to determine block height for challenge {cid_hash}, skipping ground truth generation")
                continue

            success, ground_truth, ground_cost, metrics_data, model_name = await challenge_manager.generate_ground_truth(
                cid_hash=cid_hash,
                question=c.question,
                token_usage_metrics=None,
                round_id=round_id,
                block_height=block_height
            )
            if shutdown_event.is_set():
                logger.warning("[MockValidator] Shutdown event triggered, stopping generating ground truth...")
                return

            is_valid = success and utils.is_ground_truth_valid(ground_truth)

            table_formatter.create_synthetic_challenge_table(
                round_id=round_id,
                challenge_id=c.challenge_id,
                project_phase_str=utils.get_project_phase_str(c.project_phase),
                cid=cid_hash,
                question=c.question,
                success=is_valid,
                ground_truth=ground_truth,
                ground_cost=ground_cost,
                metrics_data=metrics_data
            )
            if not is_valid:
                logger.error(f"[MockValidator] Invalid ground truth for challenge {cid_hash}, skipping validation")
                continue

            # forward to miner
            miner_axon = bt.AxonInfo._from_dict({
                "version": 9012002,
                "ip": miner_ip,
                "port": miner_port,
                "ip_type": 4,
                "placeholder1": "0",
                "placeholder2": "0",
                "protocol": 4,
                "hotkey": settings.wallet.hotkey.ss58_address,
                "coldkey": settings.wallet.hotkey.ss58_address,
            })
            logger.info(f"[MockValidator] Forwarding to miner {miner_uid}...")
            r = await query_single_miner(
                dendrite=challenge_manager.dendrite,
                uid=miner_uid,
                hotkey=settings.wallet.hotkey.ss58_address,
                axon=miner_axon,
                is_ip_duplicated=False,
                cid_hash=cid_hash,
                challenge_id=c.challenge_id,
                question=c.question,
                block_height=block_height,
                timeout=60*3,
                process_id=9999
            )
            if shutdown_event.is_set():
                logger.warning("[MockValidator] Shutdown event triggered, stopping forwarding...")
                return

            # calculate score
            (
                zip_scores,
                ground_truth_scores,
                elapse_weights,
                miners_elapse_time,
                ground_truth_scores_error
            ) = await challenge_manager.scorer_manager.compute_challenge_score(
                ground_truth,
                ground_cost,
                [r],
                challenge_id=c.challenge_id,
                cid_hash=cid_hash,
                token_usage_metrics=None,
                min_latency_improvement_ratio=0.2,
                round_id=round_id
            )
            if shutdown_event.is_set():
                logger.warning("[MockValidator] Shutdown event triggered, stopping calculating scores...")
                return

            table_formatter.create_synthetic_miners_response_table(
                round_id=round_id,
                challenge_id=c.challenge_id,
                uids=[miner_uid],
                hotkeys=[settings.wallet.hotkey.ss58_address],
                responses=[r],
                ground_truth_scores=ground_truth_scores,
                ground_truth_scores_error=ground_truth_scores_error,
                elapse_weights=elapse_weights,
                zip_scores=zip_scores,
                cid=cid_hash,
                max_table_rows=2
            )

            if c.project_phase == ProjectPhase.WARMUP.value:
                logger.info("warmup phase, skipping score update and final ranking table\n\n")
                continue
        
            project_score_matrix.append(zip_scores) 
            new_ema_scores = challenge_manager.scorer_manager.update_scores(
                uids=[miner_uid],
                hotkeys=[settings.wallet.hotkey.ss58_address],
                project_score_matrix=project_score_matrix,
                workload_score=None,
                challenge_id=c.challenge_id
            )

            table_formatter.create_synthetic_final_ranking_table(
                round_id=round_id,
                challenge_id=c.challenge_id,
                uids=[miner_uid],
                hotkeys=[settings.wallet.hotkey.ss58_address],
                workload_counts=[0.0],
                quality_scores=[[0.0]],
                workload_score=[0.0],
                new_ema_scores=new_ema_scores,
                max_table_rows=2
            )

            project_score_matrix = []
            round_id += 1

            print("\n\n")
            await asyncio.sleep(5)

        if shutdown_event.is_set():
            logger.info("[MockValidator] Shutdown requested, exiting")
            break
        
        logger.info("[MockValidator] No more challenges found.")
        while not shutdown_event.is_set():
            choice = input("No more challenges. Restart? (y/n): ").strip().lower()
            if choice in ["y", "yes"]:
                logger.info("[MockValidator] Restarting.")
                break
            elif choice in ["n", "no"]:
                logger.info("[MockValidator] Exiting validator.")
                return
            else:
                print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        asyncio.run(run_mock_validator())
    except KeyboardInterrupt:
        logger.info("[MockValidator] Shutdown complete")
    except Exception as e:
        logger.error(f"[MockValidator] Fatal error: {e}")
        sys.exit(1)

