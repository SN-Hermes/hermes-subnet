import asyncio
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse, urljoin
import aiohttp
from loguru import logger

from .types import CovalentConfig


ABSOLUTE_SCHEME_PATTERN = r'^[a-zA-Z][a-zA-Z0-9+.-]*:'
FETCH_TIMEOUT_MS = 30000


class CovalentResponse:
    def __init__(
        self,
        data: Any,
        error: bool,
        error_message: Optional[str] = None,
        error_code: Optional[int] = None
    ):
        self.data = data
        self.error = error
        self.error_message = error_message
        self.error_code = error_code


def resolve_covalent_url(path: str, base_url: str) -> str:
    if path.startswith('//'):
        raise ValueError('Protocol-relative URLs are not allowed')

    parsed = urlparse(path)
    if parsed.scheme:
        raise ValueError('Absolute URLs are not allowed')

    base_parsed = urlparse(base_url)
    # resolved = urljoin(base_url, path)
    resolved = f"{base_url.rstrip("/")}/{path.lstrip("/")}"

    resolved_parsed = urlparse(resolved)

    if resolved_parsed.netloc != base_parsed.netloc:
        raise ValueError('Cross-origin requests are not allowed')

    return resolved


class CovalentService:
    def __init__(self, config: CovalentConfig):
        self.config = config

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
        }
        if self.config.authorization:
            headers['Authorization'] = self.config.authorization
        return headers

    async def execute(
        self,
        path: str,
        params: Optional[Dict[str, Union[str, int, bool]]] = None
    ) -> CovalentResponse:
        base_url = self.config.base_url or 'https://api.covalenthq.com'
        url = resolve_covalent_url(path, base_url)
        params = params or {}
        params['passthrough'] = "novacaine"

        if params:
            param_strs = [f"{k}={v}" for k, v in params.items()]
            url = f"{url}?{'&'.join(param_strs)}"

        logger.info(f"Executing Covalent REST request: {url}")

        try:
            timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_MS / 1000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self._build_headers()) as response:
                    if not response.ok:
                        text = await response.text()
                        logger.error(f"Covalent API error: {response.status} - {text}")
                        return CovalentResponse(
                            data=None,
                            error=True,
                            error_message=f"HTTP {response.status}: {response.reason}",
                            error_code=response.status
                        )

                    result = await response.json()

                    if isinstance(result, dict) and 'error' in result and isinstance(result['error'], bool):
                        covalent_result = result

                        if covalent_result['error']:
                            logger.warning(f"Covalent API returned error: {covalent_result.get('error_message')}")
                            return CovalentResponse(
                                data=None,
                                error=True,
                                error_message=covalent_result.get('error_message'),
                                error_code=covalent_result.get('error_code')
                            )

                        if covalent_result.get('data') is None:
                            logger.warning("Covalent API returned success without data")
                            return CovalentResponse(
                                data=None,
                                error=True,
                                error_message='Missing data in Covalent response',
                                error_code=covalent_result.get('error_code')
                            )

                        logger.debug("Covalent request successful")
                        return CovalentResponse(
                            data=covalent_result['data'],
                            error=False
                        )

                    logger.warning("Unexpected response format")
                    return CovalentResponse(
                        data=result,
                        error=False
                    )

        except Exception as e:
            error_message = str(e)
            logger.error(f"Covalent request failed: {error_message}")
            return CovalentResponse(
                data=None,
                error=True,
                error_message=error_message
            )
