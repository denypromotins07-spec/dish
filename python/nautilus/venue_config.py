"""
Binance-Specific Venue Configuration for NautilusTrader.

Dynamically fetches exchange info from Binance API to configure
precise tick sizes, fee models, and instrument definitions.
Ensures accurate order simulations and live execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import orjson

log = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Parsed information about a Binance trading symbol."""
    
    symbol: str
    base_asset: str
    quote_asset: str
    status: str
    
    # Price/quantity precision
    price_precision: int
    quantity_precision: int
    
    # Tick/lot sizes
    tick_size: float
    lot_size: float
    step_size: float
    
    # Notional limits
    min_notional: float
    max_notional: float
    
    # Fee rates (will be updated from VIP level if available)
    maker_fee: float = 0.0002  # Default 0.02%
    taker_fee: float = 0.0004  # Default 0.04%
    
    # Order types supported
    allowed_order_types: List[str] = field(default_factory=list)
    
    # Time in force modes
    allowed_tif: List[str] = field(default_factory=list)


@dataclass
class BinanceVenueConfig:
    """Complete venue configuration for Binance."""
    
    venue_name: str
    is_testnet: bool
    
    # API endpoints
    rest_api_url: str
    ws_base_url: str
    
    # Rate limits
    rate_limit_per_second: int = 1200
    weight_limit_per_minute: int = 60_000
    
    # Symbols/instruments
    symbols: Dict[str, SymbolInfo] = field(default_factory=dict)
    
    # Fee tiers (VIP levels)
    fee_tiers: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    # Server time offset (for timestamp signing)
    server_time_offset_ms: int = 0


class BinanceVenueConfigLoader:
    """
    Loads and parses Binance exchange configuration dynamically.
    
    Fetches live exchange info to ensure precise instrument definitions,
    tick sizes, and fee structures for accurate backtesting and live trading.
    """
    
    def __init__(self, api_key: Optional[str] = None, testnet: bool = False):
        self.api_key = api_key
        self.testnet = testnet
        
        if testnet:
            self.rest_api_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://testnet.binancefuture.com/ws"
        else:
            self.rest_api_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com"
            
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with optimized settings."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {}
            )
        return self._session
        
    async def fetch_exchange_info(self) -> Dict[str, Any]:
        """Fetch complete exchange info from Binance API."""
        session = await self._get_session()
        url = f"{self.rest_api_url}/fapi/v1/exchangeInfo"
        
        log.info("Fetching Binance exchange info...")
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to fetch exchange info: {text}")
                
            data = await response.read()
            return orjson.loads(data)
            
    async def fetch_server_time(self) -> int:
        """Fetch server time to calculate clock skew."""
        session = await self._get_session()
        url = f"{self.rest_api_url}/fapi/v1/time"
        
        async with session.get(url) as response:
            if response.status != 200:
                return 0
                
            data = await response.read()
            result = orjson.loads(data)
            return result.get('serverTime', 0)
            
    async def fetch_account_info(self) -> Optional[Dict[str, Any]]:
        """Fetch account info including fee tiers (requires API key)."""
        if not self.api_key:
            return None
            
        session = await self._get_session()
        url = f"{self.rest_api_url}/fapi/v2/account"
        
        # Requires HMAC signature (would use Rust signer in production)
        # For now, skip if no signing capability
        log.warning("Account info requires HMAC signing - skipping for now")
        return None
        
    def parse_symbol_info(self, raw_symbol: Dict[str, Any]) -> SymbolInfo:
        """Parse raw Binance symbol data into structured format."""
        symbol = raw_symbol['symbol']
        
        # Extract filters
        filters = {f['filterType']: f for f in raw_symbol.get('filters', [])}
        
        # PRICE_FILTER
        price_filter = filters.get('PRICE_FILTER', {})
        tick_size = float(price_filter.get('tickSize', '0.01'))
        price_precision = _calculate_precision(tick_size)
        
        # LOT_SIZE / STEP_SIZE
        lot_filter = filters.get('LOT_SIZE', {}) or filters.get('STEP_SIZE', {})
        step_size = float(lot_filter.get('stepSize', '0.001'))
        quantity_precision = _calculate_precision(step_size)
        
        # MIN_NOTIONAL
        notional_filter = filters.get('MIN_NOTIONAL', {})
        min_notional = float(notional_filter.get('notional', '5.0'))
        
        return SymbolInfo(
            symbol=symbol,
            base_asset=raw_symbol.get('baseAsset', ''),
            quote_asset=raw_symbol.get('quoteAsset', ''),
            status=raw_symbol.get('status', 'UNKNOWN'),
            price_precision=price_precision,
            quantity_precision=quantity_precision,
            tick_size=tick_size,
            lot_size=step_size,
            step_size=step_size,
            min_notional=min_notional,
            max_notional=float(filters.get('MAX_NOTIONAL', {}).get('maxNotional', '1000000000')),
            allowed_order_types=raw_symbol.get('orderTypes', []),
            allowed_tif=raw_symbol.get('timeInForce', [])
        )
        
    async def load_config(self) -> BinanceVenueConfig:
        """Load complete venue configuration from Binance API."""
        import time
        
        # Fetch exchange info
        exchange_info = await self.fetch_exchange_info()
        
        # Calculate server time offset
        server_time = await self.fetch_server_time()
        local_time = int(time.time() * 1000)
        server_time_offset = server_time - local_time if server_time else 0
        
        log.info(f"Binance server time offset: {server_time_offset}ms")
        
        # Parse all symbols
        symbols: Dict[str, SymbolInfo] = {}
        for raw_symbol in exchange_info.get('symbols', []):
            if raw_symbol.get('status') == 'TRADING':
                symbol_info = self.parse_symbol_info(raw_symbol)
                symbols[symbol_info.symbol] = symbol_info
                
        log.info(f"Parsed {len(symbols)} active trading symbols")
        
        # Build venue config
        config = BinanceVenueConfig(
            venue_name="BINANCE_FUTURES_TESTNET" if self.testnet else "BINANCE_FUTURES",
            is_testnet=self.testnet,
            rest_api_url=self.rest_api_url,
            ws_base_url=self.ws_base_url,
            symbols=symbols,
            server_time_offset_ms=server_time_offset
        )
        
        # Apply fee tiers if account info available
        # (Would require signed request in production)
        
        return config
        
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


def _calculate_precision(value: float) -> int:
    """Calculate decimal precision from tick/step size."""
    if value <= 0:
        return 8
    s = f"{value:.10f}".rstrip('0').rstrip('.')
    if '.' in s:
        return len(s.split('.')[1])
    return 0


async def load_binance_venue_config(
    api_key: Optional[str] = None,
    testnet: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to load Binance venue configuration.
    
    Args:
        api_key: Optional Binance API key for account-specific data
        testnet: Use testnet endpoints
        
    Returns:
        Dictionary containing venue configuration for NautilusTrader
    """
    loader = BinanceVenueConfigLoader(api_key=api_key, testnet=testnet)
    
    try:
        config = await loader.load_config()
        
        # Convert to dict format expected by Nautilus
        result = {
            'venue_name': config.venue_name,
            'is_testnet': config.is_testnet,
            'rest_api_url': config.rest_api_url,
            'ws_base_url': config.ws_base_url,
            'rate_limit_per_second': config.rate_limit_per_second,
            'weight_limit_per_minute': config.weight_limit_per_minute,
            'symbols': {
                sym: {
                    'symbol': info.symbol,
                    'base_asset': info.base_asset,
                    'quote_asset': info.quote_asset,
                    'price_precision': info.price_precision,
                    'quantity_precision': info.quantity_precision,
                    'tick_size': info.tick_size,
                    'lot_size': info.lot_size,
                    'min_notional': info.min_notional,
                    'maker_fee': info.maker_fee,
                    'taker_fee': info.taker_fee,
                    'allowed_order_types': info.allowed_order_types,
                    'allowed_tif': info.allowed_tif,
                }
                for sym, info in config.symbols.items()
            },
            'server_time_offset_ms': config.server_time_offset_ms,
        }
        
        log.info(f"Loaded Binance venue config: {config.venue_name}")
        return result
        
    finally:
        await loader.close()


if __name__ == "__main__":
    # Test loading configuration
    async def test():
        config = await load_binance_venue_config(testnet=True)
        print(f"Loaded {len(config['symbols'])} symbols")
        # Print first few symbols as sample
        for sym in list(config['symbols'].keys())[:3]:
            print(f"  {sym}: {config['symbols'][sym]}")
    
    asyncio.run(test())
