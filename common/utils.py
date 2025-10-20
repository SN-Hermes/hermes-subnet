import json
import os
import signal
import time
import httpx
from loguru import logger
import netaddr
import requests
import hashlib
from langchain.schema import BaseMessage
from langchain.schema import AIMessage


def try_get_external_ip() -> str | None:
    try:
        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        netaddr.IPAddress(external_ip)
        return external_ip

    except Exception as e:
        logger.warning(f"Failed to get external ip: {e}")
        return None
    
def get_elapse_weight_quadratic(elapsed_time: float, ground_truth_cost: float) -> float:
    if elapsed_time <= 0:
        return 1.0
    if ground_truth_cost <= 0:
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


def fix_float(elapsed: float) -> float:
    return int(elapsed * 100) / 100

def safe_float_convert(s: str) -> float:
    try:
        return float(s)
    except Exception as e:
        return 0.0

def hash256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()
            
def kill_process_group():
    try:
        os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception as e:
        logger.error(f"Failed to kill process group: {e}")


if __name__ == "__main__":
    ground_truth_cost = 15.0
    print(get_elapse_weight_quadratic(1, ground_truth_cost))
    print(get_elapse_weight_quadratic(2, ground_truth_cost))
    print(get_elapse_weight_quadratic(4, ground_truth_cost))
    print(get_elapse_weight_quadratic(8, ground_truth_cost))
    print(get_elapse_weight_quadratic(11, ground_truth_cost))
    print(get_elapse_weight_quadratic(14, ground_truth_cost))
    print(get_elapse_weight_quadratic(20, ground_truth_cost))
    print(get_elapse_weight_quadratic(24, ground_truth_cost))
    print(get_elapse_weight_quadratic(30, ground_truth_cost))