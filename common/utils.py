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