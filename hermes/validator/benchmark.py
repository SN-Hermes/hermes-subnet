import random
from typing import Any
import aiohttp
import os
import bittensor as bt
from loguru import logger
import hashlib
import time
import base64


class BenchMark:
    def __init__(self, wallet: bt.wallet, ipc_meta_config: dict[str, Any] = None):
        self.wallet = wallet
        self.pending_uploads: dict[str, list[dict]] = {}
        if ipc_meta_config is None:
            self.ipc_meta_config = {}
        else:
            self.ipc_meta_config = ipc_meta_config

    async def upload(
        self,
        uid: int,
        address: str,
        cid: str,
        challenge_type: int,
        challenge_id: str,
        question: str,
        question_generator_model_name: str,
        ground_truth_model_name: str,
        score_model_name: str,
        ground_truth: str,
        ground_cost: float,
        ground_truth_tools: list[dict[str, str]],
        ground_input_tokens: int,
        ground_input_cache_read_tokens: int,
        ground_output_tokens: int,
        miners_answer: list[dict[str, any]],
    ):
        """
        Upload benchmark data based on mode:
        - 'sample': Upload randomly sampled data based on sample_rate, batched by cid_hash
        - 'all': Upload all data immediately
        """
        benchmark_mode = self.ipc_meta_config.get("benchmark_mode", "sample")
        benchmark_sample_rate = self.ipc_meta_config.get("benchmark_sample_rate", 0.5)
        benchmark_batch_size = self.ipc_meta_config.get("benchmark_batch_size", 0)
        benchmark_url = self.ipc_meta_config.get("benchmark_url") or os.environ.get('BOARD_SERVICE')

        if not benchmark_url:
            logger.warning("[Benchmark] No benchmark URL configured, skipping upload")
            return

        # Prepare benchmark data
        benchmark_data = {
            "uid": uid,
            "address": address,
            "cid": cid,
            "challengeType": challenge_type,
            "challengeId": challenge_id,
            "question": question,
            "questionGeneratorModelName": question_generator_model_name,
            "groundTruthModelName": ground_truth_model_name,
            "scoreModelName": score_model_name,
            "groundTruth": ground_truth,
            "groundTruthCost": ground_cost,
            "groundTruthTools": ground_truth_tools,
            "groundInputTokens": ground_input_tokens,
            "groundInputCacheReadTokens": ground_input_cache_read_tokens,
            "groundOutputTokens": ground_output_tokens,
            "minersAnswer": miners_answer,
        }

        # Determine if we should add this data
        should_upload = False
        if benchmark_mode == "all":
            should_upload = True
        elif benchmark_mode == "sample":
            should_upload = random.random() < benchmark_sample_rate
        else:
            return

        logger.debug(f"[Benchmark] Prepared benchmark data {benchmark_data}. {should_upload}")

        if should_upload:
            # Add to pending uploads for this cid
            if cid not in self.pending_uploads:
                self.pending_uploads[cid] = []
            
            self.pending_uploads[cid].append(benchmark_data)
            
            # Check if batch size reached for this cid
            if len(self.pending_uploads[cid]) >= benchmark_batch_size:
                await self._flush_cid(cid)

    async def _flush_cid(self, cid: str):
        """Flush pending uploads for a specific cid"""
        if cid not in self.pending_uploads or not self.pending_uploads[cid]:
            return

        batch = self.pending_uploads[cid]
        self.pending_uploads[cid] = []
        
        await self._send_to_server(batch)

    def _normalize_numbers(self, obj):
        """
        Recursively normalize numbers to ensure consistency between Python and TypeScript:
        - Convert float that are actually integers (e.g., 0.0, 1.0) to int
        - This ensures JSON serialization matches between Python and TypeScript
        """
        if isinstance(obj, dict):
            return {key: self._normalize_numbers(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_numbers(item) for item in obj]
        elif isinstance(obj, float):
            # If the float is actually an integer, convert it
            if obj.is_integer():
                return int(obj)
            return obj
        else:
            return obj

    async def _send_to_server(self, data_batch: list[dict]):
        """Send batch data to benchmark server"""
        try:
            # Step 1: Add timestamp and normalize data
            timestamp = int(time.time())
            
            # Normalize numbers to match TypeScript serialization
            normalized_batch = self._normalize_numbers(data_batch)
            
            payload_to_hash = {
                "benchmarks": normalized_batch,
                "timestamp": timestamp,
            }
            import msgpack
            msgpack.packb(payload_to_hash)

            b = msgpack.packb(
                payload_to_hash,
                use_bin_type=True,
                strict_types=True
            )
            h = hashlib.sha256(b).hexdigest()

            # Convert msgpack data to base64
            b_base64 = base64.b64encode(b).decode('utf-8')

            # Step 2: Sign the hash with wallet
            signature = f"0x{self.wallet.hotkey.sign(h).hex()}"
            
            # Step 3: Send hash, signature, timestamp along with data
            payload = {
                "msgpack": b_base64,
                "hash": h,
                "validator": self.wallet.hotkey.ss58_address,
                "signature": signature
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ipc_meta_config.get("benchmark_url") or f"{os.environ.get('BOARD_SERVICE')}/benchmark/msgpack",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"[Benchmark] Successfully uploaded {len(data_batch)} benchmark(s)")
                    else:
                        error_text = await resp.text()
                        logger.error(f"[Benchmark] Upload failed with status {resp.status}: {error_text}")
        except Exception as e:
            logger.error(f"[Benchmark] Failed to upload benchmark data: {e}")



