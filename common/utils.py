import os
import logging
import sys
import httpx
from loguru import logger
import netaddr
import requests


class InterceptHandler(logging.Handler):
    """Intercept standard library logging and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Use raw=True to prevent format string processing issues
        logger.opt(depth=depth, exception=record.exc_info, raw=True).log(level, record.getMessage() + "\n")


def configure_loguru():
    """
    Configure loguru to intercept standard logging and suppress noisy third-party libraries.
    """
    # Remove default loguru handler and add custom one
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>"
    )
    
    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Configure specific loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING) 
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Configure uvicorn loggers to use loguru format
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    logger.info("Configured loguru with standard library interception")


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