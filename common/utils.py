import json
import os
import signal
import time
import httpx
import aiohttp
from loguru import logger
import netaddr
import requests
import hashlib
from langchain_core.messages import BaseMessage, AIMessage
from datetime import datetime, timedelta


def try_get_external_ip() -> str | None:
    try:
        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        netaddr.IPAddress(external_ip)
        return external_ip

    except Exception as e:
        logger.warning(f"Failed to get external ip: {e}")
        return None
    
def get_elapse_weight_quadratic(elapsed_time: float, ground_truth_cost: float, min_latency_improvement_ratio: float) -> float:
    """
    Calculate weight based on elapsed time vs ground truth cost.
    
    Args:
        elapsed_time: Miner's response time
        ground_truth_cost: Ground truth generation time
        min_latency_improvement_ratio: Minimum improvement ratio required (e.g., 0.2 means miner must be 20% faster)
        
    Returns:
        float: Weight score between 0.0 and 1.0
    """
    if elapsed_time <= 0:
        return 1.0
    if ground_truth_cost <= 0:
        return 0.0

    # Check if miner meets minimum latency improvement requirement
    # e.g., if min_latency_improvement_ratio = 0.2, miner must be at least 20% faster
    # This means elapsed_time must be <= ground_truth_cost * (1 - 0.2) = ground_truth_cost * 0.8
    max_allowed_time = ground_truth_cost * (1.0 - min_latency_improvement_ratio)
    
    if elapsed_time > max_allowed_time:
        return 0.0

    time_ratio = elapsed_time / ground_truth_cost
    weight = 1.0 / ((1.0 + time_ratio) ** 2)

    return min(1.0, max(0.0, weight))

async def fetch_from_ipfs(cid: str, path: str = "") -> str:
    """
    Fetch content from IPFS using multiple methods with fallbacks.
    
    Args:
        cid: IPFS CID
        path: Optional path within the IPFS directory
        
    Returns:
        str: Content of the file
    """
    ipfs_path = f"{cid}/{path}" if path else cid
    IPFS_API_URL = os.getenv("IPFS_API_URL", "https://unauthipfs.subquery.network/ipfs/api/v0")
    
    # Try SubQuery IPFS node first, then gateway fallbacks
    sources = [
        # SubQuery IPFS node (cat API with POST method) - PRIMARY
        {
            "name": "SubQuery IPFS Cat API",
            "url": f"{IPFS_API_URL}/cat",
            "method": "post",
            "params": {"arg": ipfs_path}
        },
        # Gateway fallbacks
        {
            "name": "Gateway (ipfs.io)",
            "url": f"https://ipfs.io/ipfs/{ipfs_path}",
            "method": "get"
        },
        {
            "name": "Gateway (gateway.pinata.cloud)",
            "url": f"https://gateway.pinata.cloud/ipfs/{ipfs_path}",
            "method": "get"
        },
        {
            "name": "Gateway (dweb.link)",
            "url": f"https://dweb.link/ipfs/{ipfs_path}",
            "method": "get"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for source in sources:
            try:
                logger.debug(f"Trying {source['name']}: {source['url']}")
                
                if source["method"] == "post":
                    response = await client.post(source["url"], params=source.get("params", {}))
                else:
                    response = await client.get(source["url"])
                
                if response.status_code == 200:
                    content = response.text
                    logger.info(f"Successfully fetched from {source['name']} ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"{source['name']} failed: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                logger.error(f"{source['name']} error: {e}")
                continue
    
    # If all sources fail
    raise RuntimeError(f"Failed to fetch {ipfs_path} from all IPFS sources")

def create_system_prompt(
    domain_name: str,
    domain_capabilities: list,
    decline_message: str
) -> str:
    """
    Create a system prompt for langgraph GraphQL agent.
    
    Args:
        domain_name: Name of the domain/project (e.g., "SubQuery Network", "DeFi Protocol")
        domain_capabilities: List of capabilities/data types the agent can help with
        decline_message: Custom message when declining out-of-scope requests
        
    Returns:
        str: System prompt for langgraph agent
    """
    capabilities_text = '\n'.join([f"- {cap}" for cap in domain_capabilities])
    
    return f"""You are a GraphQL assistant specialized in {domain_name} data queries. You can help users find information about:
{capabilities_text}

RESPONSE STYLE: Provide complete, definitive responses. Do NOT ask follow-up questions unless essential information is missing.

WORKFLOW:

IF NOT RELATED to {domain_name}:
- Politely decline with: "{decline_message}"

IF RELATED to {domain_name} data:
1. Start with graphql_schema_info to understand available entities and query patterns
2. Construct proper GraphQL queries based on the schema
3. Validate queries with graphql_query_validator before execution
4. Execute queries with graphql_execute
5. Provide clear, user-friendly summaries of the results

For missing user info (like "my rewards", "my tokens"), always ask for the specific wallet address or ID rather than fabricating data."""

def select_uid(synthetic_score: dict, available_miners: list, uid_select_count: dict, max_count: int = 5) -> tuple[int | None, str | None]:
    sorted_miners = sorted(
        [(uid, synthetic_score[uid][0] if uid in synthetic_score else 0.0) for uid in available_miners],
        key=lambda x: x[1],
        reverse=True
    )
    logger.info(f"synthetic_score: {synthetic_score}, available miners: {available_miners}, sorted miners: {sorted_miners}, uid_select_count: {uid_select_count}")
    for uid, hotkey in sorted_miners:
        if uid_select_count.get(uid, 0) < max_count:
            uid_select_count[uid] = uid_select_count.get(uid, 0) + 1
            return uid, hotkey
    if sorted_miners:
        uid_select_count[uid] = 1
        return sorted_miners[0][0], sorted_miners[0][1]

    return None, None

def try_get_invalid_tool_messages(messages: list[BaseMessage] | BaseMessage) -> str | None:
    if not isinstance(messages, list):
        messages = [messages]

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if len(m.invalid_tool_calls) > 0:
                # logger.info(f"----> found invalid tool call, {m.invalid_tool_calls}")
                return json.dumps(m.invalid_tool_calls)
    return None

def is_ground_truth_valid(ground_truth: str | None) -> bool:
    """
    Validate if ground truth generation was successful.

    Returns False if:
    - ground_truth is None or empty
    - ground_truth starts with "ERROR:" (agent's standardized error format)
    - ground_truth contains known failure patterns like "Sorry, need more steps"

    Args:
        ground_truth: The generated ground truth response

    Returns:
        bool: True if valid, False if failed
    """
    if not ground_truth or not ground_truth.strip():
        return False

    ground_truth_lower = ground_truth.strip().lower()

    # Check for standardized ERROR format
    if ground_truth.strip().startswith("ERROR:"):
        return False

    # Check for known failure patterns (case-insensitive)
    failure_patterns = [
        "sorry, need more steps",
        "need more steps to process",
        "cannot process this request",
        "recursion limit",
        "unable to complete"
    ]

    for pattern in failure_patterns:
        if pattern in ground_truth_lower:
            return False

    return True

def try_get_tool_hit(messages: list[BaseMessage], exclude_tools=[]) -> list[tuple[str, int]]:
    tool_order = []
    tool_counts = {}
    for m in messages:
        if m.type == 'tool' and m.name not in exclude_tools:
            if m.name not in tool_counts:
                tool_order.append(m.name)
                tool_counts[m.name] = 1
            else:
                tool_counts[m.name] += 1
    tool_hit = [(name, tool_counts[name]) for name in tool_order]
    return tool_hit

def form_training_data(question: str, block_height: int, response_messages: list[BaseMessage], metrics_data: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": "[SYS_PROMPT]"
        },
        {
            "role": "system",
            "content": "[BLOCK_RULE_PROMPT]"
        },
        {
            "role": "user",
            "content": question
        },
    ]
    for msg in response_messages:
        if msg.type == 'ai' and hasattr(msg, 'tool_calls') and len(msg.tool_calls) > 0:
            first = msg.tool_calls[0]
            name = first.get("name")
            args = first.get("args")

            if name == 'graphql_schema_info':
                messages.append({
                    "role": "assistant",
                    "content": f"<tool name=\"{name}\">{json.dumps(args)}</tool>"
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"<tool name=\"{name}\">{json.dumps(args)}</tool>"
                })
        elif msg.type == 'tool':
            if msg.name == 'graphql_schema_info':
                messages.append({
                    "role": "tool",
                    "content": "[SCHEMA_INFO]"
                })
            else:
                messages.append({
                    "role": "tool",
                    "content": msg.content
                })

    
    result = response_messages[-1].content

    messages.append({
        "role": "assistant",
        "content": result
    })

    return {
        "messages": messages,
        "block_height": block_height,
    }

def format_openai_message(content: str, finish_reason=None) -> str:
    chunk_data = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk", 
        "created": int(time.time()),
        "model": "hermes-miner",
        "system_fingerprint": "fp_hermes",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": finish_reason
        }]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"

def format_openai_key() -> str:
    # Format API key to show only first 6 and last 4 characters
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and len(api_key) > 10:
        formatted_key = f"{api_key[:6]}****{api_key[-4:]}"
    else:
        formatted_key = "****" if api_key else "Not Set"
    return formatted_key

def extract_token_usage(messages: list[BaseMessage]) -> tuple[int, int, int]:
    if not messages:
        return 0, 0, 0

    if not isinstance(messages, list):
        messages = [messages]
    
    input_tokens = 0
    input_cache_read_tokens = 0
    output_tokens = 0

    for m in messages:
        if hasattr(m, 'usage_metadata') and m.usage_metadata:
            usage = m.usage_metadata
            input_tokens += usage.get("input_tokens", 0)

            input_token_details = usage.get("input_token_details", {})
            input_cache_read_tokens += input_token_details.get("cache_read", 0)

            output_tokens += usage.get("output_tokens", 0)
    return input_tokens, input_cache_read_tokens, output_tokens

def extract_tool_calls(messages: list[BaseMessage]) -> list[str]:
    tool_calls = []
    if not messages:
        return tool_calls
    for m in messages:
        if hasattr(m, 'tool_calls') and len(m.tool_calls) > 0:
            for tc in m.tool_calls:
                t = {
                    "name": tc.get("name"),
                    "args": tc.get('args')
                }
                tool_calls.append(json.dumps(t))
    return tool_calls

def get_func_name(f):
    if hasattr(f, "__name__"):
        return f.__name__
    elif hasattr(f, "func") and hasattr(f.func, "__name__"):
        return f.func.__name__
    else:
        return str(f)
    
def fix_float(elapsed: float) -> float:
    return int(elapsed * 100) / 100

def safe_float_convert(s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        return 0.0

def is_array(obj) -> bool:
    return isinstance(obj, (list, tuple))

def is_list(obj) -> bool:
    return isinstance(obj, list)

def hash256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()

def parse_time_range(time_range: str) -> int:
    """
    Parse time range string and return cutoff timestamp.
    
    Args:
        time_range: Time range string like "1h", "24h", "30min", "2d"
        
    Returns:
        int: Cutoff timestamp (seconds since epoch)
        
    Examples:
        parse_time_range("1h") -> timestamp 1 hour ago
        parse_time_range("30min") -> timestamp 30 minutes ago
        parse_time_range("2d") -> timestamp 2 days ago
    """
    current_time = datetime.now()
    cutoff_time = None
        
    if 'min' in time_range:
        # Extract number before 'min'
        value = int(time_range.replace('min', ''))
        cutoff_time = current_time - timedelta(minutes=value)
    elif 'h' in time_range:
        # Extract number before 'h'  
        value = int(time_range.replace('h', ''))
        cutoff_time = current_time - timedelta(hours=value)
    elif 'd' in time_range:
        # Extract number before 'd'
        value = int(time_range.replace('d', ''))
        cutoff_time = current_time - timedelta(days=value)
    else:
        # Default to 1 hour if format not recognized
        cutoff_time = current_time - timedelta(hours=1)
        
    return int(cutoff_time.timestamp())
       
def calculate_token_cost(
    input_tokens: int, 
    output_tokens: int, 
    input_cache_tokens: int = 0,
    model_name: str = "gpt-4o"
) -> dict:
    """
    Calculate token costs based on OpenAI pricing models.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens  
        input_cache_tokens: Number of cached input tokens (usually 50% discount)
        model_name: OpenAI model name for pricing lookup
        
    Returns:
        dict: Cost breakdown with total_cost, input_cost, cache_cost, output_cost, avg_token_price
    """
    
    # OpenAI pricing per 1M tokens (USD) - updated 2024 pricing
    pricing_table = {
        "moonshotai/kimi-k2-0905": {"input": 0.6, "output": 2.50},
        "zai-org/glm-4.6": {"input": 0.60, "output": 2.20, "input_cache": 0.0},
        "gpt-5-mini": {"input": 0.25, "output": 2.00, "input_cache": 0.025},
        "gpt-5": {"input": 1.25, "output": 10, "input_cache": 0.125},

        # for fine tuning
        "gpt-4.1-mini": {"input": 0.80, "output": 3.20, "input_cache": 0.20},

        "minimax/minimax-m2": {"input": 0.255, "output": 1.02, "input_cache": 0.0},

    }
    
    # Get pricing for the model, fallback to default if not found
    pricing = pricing_table.get(model_name)
    input_per_1m_price = pricing["input"]
    output_per_1m_price = pricing["output"]
    input_cache_1m_price = pricing.get("input_cache", 0.0)

    # Calculate actual input tokens (excluding cache)
    actual_input_tokens = input_tokens - input_cache_tokens
    
    # Calculate costs (prices are per 1M tokens)
    input_cost = (actual_input_tokens / 1_000_000) * input_per_1m_price
    cache_cost = 0 if input_cache_1m_price == 0 else (input_cache_tokens / 1_000_000) * input_cache_1m_price
    output_cost = (output_tokens / 1_000_000) * output_per_1m_price
    total_cost = input_cost + cache_cost + output_cost
    
    # Calculate average price per token
    total_tokens = input_tokens + output_tokens
    avg_token_price = total_cost / total_tokens if total_tokens > 0 else 0
    
    return {
        "model_name": model_name,
        "total_cost": round(total_cost, 8),
        "input_cost": round(input_cost, 8),
        "cache_cost": round(cache_cost, 8), 
        "output_cost": round(output_cost, 8),
        "avg_token_price": round(avg_token_price, 10),
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "actual_input_tokens": actual_input_tokens,
        "input_cache_tokens": input_cache_tokens,
        "output_tokens": output_tokens,
        "pricing_per_1m": {
            "input": input_per_1m_price,
            "output": output_per_1m_price
        }
    }

def pick(obj: dict, keys: list) -> dict:
    """
    Create a new dictionary with only the specified keys from the original dictionary.
    
    Args:
        obj: Source dictionary to pick from
        keys: List of keys to include in the result
        
    Returns:
        dict: New dictionary containing only the specified keys and their values
        
    Examples:
        pick({"a": 1, "b": 2, "c": 3}, ["a", "c"]) -> {"a": 1, "c": 3}
        pick({"x": 10, "y": 20}, ["x", "z"]) -> {"x": 10}
    """
    if not isinstance(obj, dict):
        return {}
    
    return {key: obj[key] for key in keys if key in obj}

def omit(obj: dict, keys: list) -> dict:
    """
    Create a new dictionary excluding the specified keys from the original dictionary.
    
    Args:
        obj: Source dictionary to omit from
        keys: List of keys to exclude from the result
        
    Returns:
        dict: New dictionary without the specified keys
        
    Examples:
        omit({"a": 1, "b": 2, "c": 3}, ["b"]) -> {"a": 1, "c": 3}
        omit({"x": 10, "y": 20, "z": 30}, ["x", "z"]) -> {"y": 20}
    """
    if not isinstance(obj, dict):
        return {}
    
    return {key: value for key, value in obj.items() if key not in keys}

async def get_latest_block(endpoint: str, node_type: str) -> int | None:
    """
    Get the latest block height from a GraphQL endpoint.
    
    Args:
        endpoint: The GraphQL endpoint URL
        node_type: Type of node - "subql" or "thegraph"
        
    Returns:
        int: Latest block height, or None if failed
        
    Examples:
        For SubQuery: returns lastProcessedHeight from _metadata
        For The Graph: returns block number from _meta
    """
    try:
        # Construct GraphQL query based on node type
        if node_type == "subql":
            query = """
            query {
              _metadata {
                lastProcessedHeight
              }
            }
            """
        elif node_type == "thegraph":
            query = """
            {
              _meta {
                block {
                  number
                }
              }
            }
            """
        else:
            logger.error(f"Unknown node_type: {node_type}")
            return None
        
        # Send GraphQL request using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout = aiohttp.ClientTimeout(total=30.0)
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to get latest block from {endpoint}: HTTP {response.status}")
                    return None
                
                data = await response.json()
                
                # Extract block height based on node type
                if node_type == "subql":
                    block_height = data.get("data", {}).get("_metadata", {}).get("lastProcessedHeight")
                elif node_type == "thegraph":
                    block_height = data.get("data", {}).get("_meta", {}).get("block", {}).get("number")
                else:
                    return None
                
                if block_height is not None:
                    logger.info(f"Latest block height from {endpoint} ({node_type}): {block_height}")
                    return int(block_height)
                else:
                    logger.error(f"Failed to extract block height from response: {data}")
                    return None
                
    except Exception as e:
        logger.error(f"Error getting latest block from {endpoint}: {e}")
        return None

def kill_process_group():
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception as e:
        logger.error(f"Failed to kill process group: {e}")


if __name__ == "__main__":
    ground_truth_cost = 15.0
    print(get_elapse_weight_quadratic(1, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(2, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(4, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(8, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(11, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(14, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(20, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(24, ground_truth_cost, 0.2))
    print(get_elapse_weight_quadratic(30, ground_truth_cost, 0.2))

    # total_cost_info = calculate_token_cost(
    #     input_tokens=72999,
    #     output_tokens=800,
    #     input_cache_tokens=0,
    #     model_name='moonshotai/kimi-k2-0905'
    # )
    # logger.info(f"Total Token: {total_cost_info['total_tokens']}")
    # logger.info(f"Total Cost: ${total_cost_info['total_cost']:.6f}")
    # logger.info(f"Average Token Price: ${total_cost_info['avg_token_price']:.8f}")
    # logger.info(f"Cost Breakdown - Input: ${total_cost_info['input_cost']:.6f}, Cache: ${total_cost_info['cache_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}")

    # total_cost_info = calculate_token_cost(
    #     input_tokens=77732,
    #     output_tokens=626,
    #     input_cache_tokens=27468,
    #     model_name='zai-org/glm-4.6'
    # )
    # logger.info(f"Total Token: {total_cost_info['total_tokens']}")
    # logger.info(f"Total Cost: ${total_cost_info['total_cost']:.6f}")
    # logger.info(f"Average Token Price: ${total_cost_info['avg_token_price']:.8f}")
    # logger.info(f"Cost Breakdown - Input: ${total_cost_info['input_cost']:.6f}, Cache: ${total_cost_info['cache_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}")


    # total_cost_info = calculate_token_cost(
    #     input_tokens=54770,
    #     output_tokens=4279,
    #     input_cache_tokens=38087,
    #     model_name='gpt-5-mini'
    # )
    # logger.info(f"Total Token: {total_cost_info['total_tokens']}")
    # logger.info(f"Total Cost: ${total_cost_info['total_cost']:.6f}")
    # logger.info(f"Average Token Price: ${total_cost_info['avg_token_price']:.8f}")
    # logger.info(f"Cost Breakdown - Input: ${total_cost_info['input_cost']:.6f}, Cache: ${total_cost_info['cache_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}")

    total_cost_info = calculate_token_cost(
        input_tokens=2000,
        output_tokens=2000,
        input_cache_tokens=0,
        model_name='gpt-4.1-mini'
    )
    logger.info(f"Total Token: {total_cost_info['total_tokens']}")
    logger.info(f"Total Cost: ${total_cost_info['total_cost']:.6f}")
    logger.info(f"Average Token Price: ${total_cost_info['avg_token_price']:.10f}")
    logger.info(f"Cost Breakdown - Input: ${total_cost_info['input_cost']:.6f}, Cache: ${total_cost_info['cache_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}")


def append_to_jsonl(
    file_path: str,
    data: any,
) -> bool:
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Append to file
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.debug(f"Successfully appended sample to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to append to JSONL file {file_path}: {e}")
        return False
