import aiohttp
import bittensor as bt

class HighConcurrencyDendrite(bt.dendrite):
    def __init__(self, wallet: bt.Wallet=None, max_connections=500, total_timeout=300):
        super().__init__(wallet)
        self.max_connections = max_connections
        self.total_timeout = total_timeout  
        self._custom_session = None
    
    @property
    async def session(self) -> aiohttp.ClientSession:
        """Override session property with custom limits"""
        if self._custom_session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=100, #  100 concurrent requests per IP (to prevent overloading a single IP)
                ttl_dns_cache=300,  # 5 minutes
                force_close=True,
                enable_cleanup_closed=True
            )
            self._custom_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.total_timeout)
            )
        return self._custom_session
    
    async def aclose_session(self):
        """Override close method"""
        if self._custom_session:
            await self._custom_session.close()
            self._custom_session = None
