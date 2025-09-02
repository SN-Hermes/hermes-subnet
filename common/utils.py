import os
import httpx
from loguru import logger
import netaddr
import requests


def try_get_external_ip() -> str | None:
    try:
        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        netaddr.IPAddress(external_ip)
        return external_ip

    except Exception as e:
        logger.warning(f"Failed to get external ip: {e}")
        return None
    

# unit: seconds
WEIGHT_CONFIG = [
    ((0, 2), 0.5),     # [0, 2]
    ((2, 5), 0.2),     # (2, 5]
    ((5, 10), 0.2),    # (5, 10]
    ((10, float("inf")), 0.1),  # (10, +∞)
]

def get_elapse_weight(elapsed_time: float) -> float:
    for (low, high), weight in WEIGHT_CONFIG:
        if low <= elapsed_time <= high:
            return weight
    return 0.0



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

