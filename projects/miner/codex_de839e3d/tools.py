import aiohttp
from typing import Any, Dict

async def request_codex(options: Dict[str, Any]):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "<<your codex token>>"
    }

    async with aiohttp.ClientSession() as session:
        payload = {
            "variables": options.get("variables", {}),
             "query": options["query"]
        }
        url = options.get("url") or "https://graph.codex.io/graphql"
        timeout = options.get("timeout", 30)
        method = options.get("method", "POST").upper()
        async with session.request(
            method,
            url,
            headers=headers,
            json=payload,
            raise_for_status=True,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            result = await resp.json()
            res = result.get("data", {}).get(options["type"])
        return res

async def query_networks_quantity() -> dict[str, str]:
    """
    Query the quantity of networks supported on CODEX.
    """
    query = """
    {
      getNetworks {
        id
        name
      }
    }
    """

    r = []
    try:
        r = await request_codex({
            "query": query,
            "type": "getNetworks",
            "variables": {},
        })
    except Exception as e:
        print(f"Error occurred: {e}")
    

    print(f"networks: {r}")
    return {
        "result": f"networks count: {len(r)}",
        "query": query
    }

tools = [query_networks_quantity]
