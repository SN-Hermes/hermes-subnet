"""
Multi-process miner query module for ChallengeManager.
Handles parallel querying of miners using multiprocessing to improve performance.
"""
import asyncio
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import List, Tuple

import bittensor as bt
from loguru import logger

from common.enums import ErrorCode
from common.protocol import SyntheticNonStreamSynapse
from common.settings import Settings
import common.utils as utils
from hermes.validator.dendrite import HighConcurrencyDendrite



async def query_single_miner(
    dendrite: HighConcurrencyDendrite,
    uid: int,
    hotkey: str,
    axon: bt.AxonInfo,
    is_ip_duplicated: bool,
    cid_hash: str,
    challenge_id: str,
    question: str,
    block_height: int,
    timeout: int,
    process_id: int,
) -> SyntheticNonStreamSynapse:
    """
    Query a single miner (async).
    Handles all error cases and always returns a SyntheticNonStreamSynapse object.
    """
    synapse = SyntheticNonStreamSynapse(
        id=challenge_id,
        uid=uid,
        cid_hash=cid_hash,
        question=question,
        block_height=block_height
    )
    query_start_time = time.perf_counter()
    
    # Initialize response with error defaults
    r = SyntheticNonStreamSynapse(
        id=challenge_id,
        uid=uid,
        cid_hash=cid_hash,
        question=question,
        block_height=block_height
    )
    r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
    r.error = "Unknown error"
    
    try:
        if not hotkey:
            r.dendrite = bt.TerminalInfo(status_code=200)
            r.status_code = ErrorCode.NOT_HEALTHY.value
            r.error = "Miner is not healthy"
        elif is_ip_duplicated:
            r.dendrite = bt.TerminalInfo(status_code=200)
            r.status_code = ErrorCode.DUPLICATED_IP.value
            r.error = "Miner has duplicated IP"
        else:
            # Query the miner directly
            r = await dendrite.forward(
                axons=axon,
                synapse=synapse,
                deserialize=False,
                timeout=timeout,
            )
            logger.debug(
                f"🔍 [Process-{process_id}] - {challenge_id} MINER RESPONSE [UID: {uid}] - "
                f"✅ is_success: {r.is_success} - {r.dendrite.status_code} - {r.dendrite.status_message}"
            )
    except Exception as e:
        logger.error(f"🔍 [Process-{process_id}] - {challenge_id} UID {uid} query failed: {e}")
        if not hasattr(r, 'status_code'):
            r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
        if not hasattr(r, 'error'):
            r.error = str(e)
    finally:
        r.uid = uid
        r.elapsed_time = utils.fix_float(time.perf_counter() - query_start_time)
        return r


async def query_miner_batch(
    process_id: int,
    miner_data_list: List[Tuple[int, str, str, bool]],  # [(uid, hotkey, axon_str, is_ip_duplicated), ...]
    cid_hash: str,
    challenge_id: str,
    question: str,
    block_height: int,
    timeout: int,
) -> List[SyntheticNonStreamSynapse]:
    """
    Query a batch of miners in one process using asyncio.gather for concurrency.
    This function runs in a separate process and recreates necessary objects.
    """
    logger.info(f"[Process-{process_id}] Starting batch query for {len(miner_data_list)} miners")
    start_time = time.perf_counter()
    
    # Recreate Settings in this process (will read from environment variables)
    settings = Settings()
    
    # Create dendrite for this process
    dendrite = HighConcurrencyDendrite(wallet=settings.wallet, max_connections=200)
    
    try:
        # Use asyncio.gather to query all miners concurrently within this process
        responses = await asyncio.gather(
            *(query_single_miner(
                dendrite=dendrite,
                uid=uid,
                hotkey=hotkey,
                axon=bt.AxonInfo.from_string(axon),
                is_ip_duplicated=is_ip_duplicated,
                cid_hash=cid_hash,
                challenge_id=challenge_id,
                question=question,
                block_height=block_height,
                timeout=timeout,
                process_id=process_id,
            ) for uid, hotkey, axon, is_ip_duplicated in miner_data_list),
        )
        
        elapsed = time.perf_counter() - start_time

        logger.info(f"[Process-{process_id}] Batch query done in {elapsed:.2f}s, {len(responses)} responses")
        return responses
    finally:
        await dendrite.aclose_session()


def run_query_process_batch(
    process_id: int,
    miner_data_list: List[Tuple[int, str, str, bool]],
    cid_hash: str,
    challenge_id: str,
    question: str,
    block_height: int,
    timeout: int,
) -> List[SyntheticNonStreamSynapse]:
    """
    Entry point for each process.
    Creates a new event loop and runs the async query_miner_batch function.
    """
    logger.info(f"[Process-{process_id}] Started with PID {os.getpid()}")
    
    loop = None
    try:
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            query_miner_batch(
                process_id,
                miner_data_list,
                cid_hash,
                challenge_id,
                question,
                block_height,
                timeout,
            )
        )
        
        logger.info(f"[Process-{process_id}] Completed with {len(result)} responses")
        return result
    except Exception as e:
        logger.error(f"[Process-{process_id}] Error: {e}")

        # Return error placeholder responses for all miners in this batch
        error_responses = []
        for uid, _, _, _ in miner_data_list:
            r = SyntheticNonStreamSynapse(
                id=challenge_id, uid=uid, cid_hash=cid_hash,
                question=question, block_height=block_height
            )
            r.status_code = ErrorCode.PROCESS_ERROR.value
            r.error = f"Process error: {e}"
            r.elapsed_time = 0.0
            error_responses.append(r)
        return error_responses

    finally:
        # Ensure event loop is properly closed
        if loop is not None:
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                # Run loop one more time to process cancellations
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()


async def query_miners_multiprocess(
    uids: List[int],
    hotkeys: List[str],
    axons: List[str],
    ips: List[str],
    seen_ips: dict,
    cid_hash: str,
    challenge_id: str,
    question: str,
    block_height: int,
    timeout: int,
    settings: Settings,
) -> List[SyntheticNonStreamSynapse]:
    """
    Main entry point for multi-process miner querying.
    
    Args:
        uids: List of miner UIDs
        hotkeys: List of miner hotkeys
        axons: List of miner axons
        ips: List of miner IPs
        seen_ips: Dict mapping IP to first UID that used it
        cid_hash: Project CID hash
        challenge_id: Challenge UUID
        question: Challenge question
        block_height: Block height for query
        timeout: Query timeout in seconds
        settings: Settings instance (only for reference, will be recreated in subprocesses)
    
    Returns:
        List of SyntheticNonStreamSynapse responses in the same order as input UIDs
    """
    # Determine number of processes
    available_cpus = settings.cpu_count
    max_processes = min(available_cpus, len(uids), 8)  # Cap at 8 processes
    
    logger.info(
        f"[MultiprocessQuery] - {challenge_id} Starting multiprocess query: "
        f"{len(uids)} miners, {max_processes} processes "
    )
    
    # Prepare miner data with IP duplication check
    miner_data_list = [
        (uid, hotkey, axon, bool(ip) and seen_ips.get(ip) != uid)
        for uid, hotkey, axon, ip in zip(uids, hotkeys, axons, ips)
    ]
    
    # Split miners into batches
    batch_size = (len(uids) + max_processes - 1) // max_processes
    batches = []
    
    for i in range(max_processes):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(uids))
        batch_data = miner_data_list[start_idx:end_idx]
        
        if batch_data:
            batches.append(batch_data)
            logger.info(
                f"[MultiprocessQuery] - {challenge_id} Process-{i} will handle "
                f"{len(batch_data)} miners (indices {start_idx}-{end_idx-1})"
            )
    
    # Start multiprocessing
    overall_start = time.perf_counter()
    
    # Use spawn method (now works because ChallengeProcess is not daemon)
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Prepare arguments for each process
        args_list = [
            (
                i,
                batch_data,
                cid_hash,
                challenge_id,
                question,
                block_height,
                timeout,
            )
            for i, batch_data in enumerate(batches)
        ]
        
        # Run processes in parallel
        results = pool.starmap(run_query_process_batch, args_list)
    
    overall_elapsed = time.perf_counter() - overall_start
    
    # Flatten results (maintain order)
    all_responses = []
    for batch_result in results:
        if batch_result:
            all_responses.extend(batch_result)
    
    logger.info(
        f"[MultiprocessQuery] - {challenge_id} All processes done in {overall_elapsed:.2f}s, "
        f"collected {len(all_responses)} responses"
    )
    
    return all_responses
