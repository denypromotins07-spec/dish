"""
Binance Instrument Universe Selection.

Fetches all trading pairs from Binance via REST API, filters them based on
liquidity/volume/ATR criteria, and generates strict Nautilus Instrument
definitions (CryptoFuture, CryptoPerpetual) for the trading universe.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import orjson

log = logging.getLogger(__name__)


@dataclass
class LiquidityCriteria:
    """Criteria for filtering instruments by liquidity."""
    
    # Minimum 24h volume in quote currency
    min_volume_24h_usd: float = 10_000_000.0
    
    # Minimum open interest (for futures)
    min_open_interest_usd: float = 5_000_000.0
    
    # Maximum bid-ask spread (in basis points)
    max_spread_bps: float = 10.0  # 0.1%
    
    # Minimum number of trades per day
    min_trades_24h: int = 10_000
    
    # Minimum average trade size
    min_avg_trade_size_usd: float = 100.0


@dataclass
class VolatilityCriteria:
    """Criteria for filtering instruments by volatility."""
    
    # Minimum ATR (Average True Range) as percentage
    min_atr_percent: float = 0.5  # 0.5% daily movement
    
    # Maximum ATR (avoid extremely volatile assets)
    max_atr_percent: float = 20.0  # 20% daily movement
    
    # Minimum price (avoid penny tokens)
    min_price_usd: float = 0.1
    
    # Maximum price (avoid ultra-expensive tokens)
    max_price_usd: float = 1_000_000.0


@dataclass
class FilteredInstrument:
    """Filtered instrument ready for Nautilus Trader."""
    
    symbol: str
    base_asset: str
    quote_asset: str
    
    # Price/quantity precision
    price_precision: int
    quantity_precision: int
    
    # Tick/lot sizes
    tick_size: float
    lot_size: float
    
    # Notional limits
    min_notional: float
    
    # Fee rates
    maker_fee: float
    taker_fee: float
    
    # Liquidity metrics
    volume_24h_usd: float
    open_interest_usd: float
    spread_bps: float
    
    # Volatility metrics
    atr_percent: float
    current_price: float
    
    # Nautilus instrument type
    instrument_type: str = "CryptoPerpetual"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BinanceUniverseSelector:
    """
    Select and filter Binance trading instruments for the active universe.
    
    Fetches live exchange data and applies configurable filters to identify
    high-quality trading candidates suitable for institutional strategies.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        is_testnet: bool = False,
        liquidity_criteria: Optional[LiquidityCriteria] = None,
        volatility_criteria: Optional[VolatilityCriteria] = None,
    ):
        self.api_key = api_key
        self.is_testnet = is_testnet
        self.liquidity_criteria = liquidity_criteria or LiquidityCriteria()
        self.volatility_criteria = volatility_criteria or VolatilityCriteria()
        
        if is_testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
                
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
            )
        return self._session
        
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            
    async def fetch_exchange_info(self) -> List[Dict[str, Any]]:
        """Fetch all symbols from Binance exchange info."""
        session = await self._get_session()
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        
        log.info("Fetching Binance exchange info...")
        async with session.get(url) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Failed to fetch exchange info: {text}")
                
            data = await response.read()
            result = orjson.loads(data)
            return result.get('symbols', [])
            
    async def fetch_ticker_24h(self) -> List[Dict[str, Any]]:
        """Fetch 24-hour ticker statistics for all symbols."""
        session = await self._get_session()
        url = f"{self.base_url}/fapi/v1/ticker/24hr"
        
        log.info("Fetching 24h ticker data...")
        async with session.get(url) as response:
            if response.status != 200:
                return []
                
            data = await response.read()
            return orjson.loads(data)
            
    async def fetch_open_interest(self) -> Dict[str, float]:
        """Fetch open interest for all symbols."""
        session = await self._get_session()
        url = f"{self.base_url}/fapi/v1/openInterest"
        
        try:
            log.info("Fetching open interest data...")
            async with session.get(url) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.read()
                result = orjson.loads(data)
                
                # Convert to dict: symbol -> open_interest_value
                oi_dict = {}
                for item in result.get('openInterestStats', []):
                    symbol = item.get('symbol', '')
                    oi_value = float(item.get('openInterest', '0')) * float(item.get('markPrice', '0'))
                    oi_dict[symbol] = oi_value
                    
                return oi_dict
        except Exception as e:
            log.warning(f"Failed to fetch open interest: {e}")
            return {}
            
    async def fetch_orderbook_depth(self, symbol: str, limit: int = 5) -> Dict[str, Any]:
        """Fetch order book depth for a specific symbol."""
        session = await self._get_session()
        url = f"{self.base_url}/fapi/v1/depth"
        
        params = {"symbol": symbol, "limit": limit}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    return {}
                    
                data = await response.read()
                return orjson.loads(data)
        except Exception:
            return {}
            
    def parse_symbol_filters(self, raw_symbol: Dict[str, Any]) -> Dict[str, Any]:
        """Parse filter information from symbol data."""
        filters = {f['filterType']: f for f in raw_symbol.get('filters', [])}
        
        # PRICE_FILTER
        price_filter = filters.get('PRICE_FILTER', {})
        tick_size = float(price_filter.get('tickSize', '0.01'))
        price_precision = self._calculate_precision(tick_size)
        
        # LOT_SIZE
        lot_filter = filters.get('LOT_SIZE', {})
        step_size = float(lot_filter.get('stepSize', '0.001'))
        quantity_precision = self._calculate_precision(step_size)
        
        # MIN_NOTIONAL
        notional_filter = filters.get('MIN_NOTIONAL', {})
        min_notional = float(notional_filter.get('notional', '5.0'))
        
        return {
            'tick_size': tick_size,
            'price_precision': price_precision,
            'step_size': step_size,
            'quantity_precision': quantity_precision,
            'min_notional': min_notional,
        }
        
    def _calculate_precision(self, value: float) -> int:
        """Calculate decimal precision from tick/step size."""
        if value <= 0:
            return 8
        s = f"{value:.10f}".rstrip('0').rstrip('.')
        if '.' in s:
            return len(s.split('.')[1])
        return 0
        
    def calculate_spread_bps(self, orderbook: Dict[str, Any]) -> float:
        """Calculate bid-ask spread in basis points."""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return float('inf')
            
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        
        if best_bid == 0:
            return float('inf')
            
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        return spread_bps
        
    async def filter_instruments(self) -> List[FilteredInstrument]:
        """
        Apply all filters and return qualified instruments.
        
        Returns list of instruments meeting all criteria.
        """
        # Fetch all required data
        exchange_info = await self.fetch_exchange_info()
        ticker_24h = await self.fetch_ticker_24h()
        open_interest = await self.fetch_open_interest()
        
        # Build ticker lookup
        ticker_lookup = {t['symbol']: t for t in ticker_24h}
        
        # Process each symbol
        qualified = []
        
        for raw_symbol in exchange_info:
            # Only consider trading symbols
            if raw_symbol.get('status') != 'TRADING':
                continue
                
            # Only perpetual contracts (USDT-margined)
            if raw_symbol.get('contractType') != 'PERPETUAL':
                continue
                
            symbol = raw_symbol['symbol']
            
            # Get ticker data
            ticker = ticker_lookup.get(symbol, {})
            
            # Parse filters
            filters = self.parse_symbol_filters(raw_symbol)
            
            # Get current price
            current_price = float(ticker.get('lastPrice', '0'))
            
            # Skip if price outside range
            if current_price < self.volatility_criteria.min_price_usd:
                continue
            if current_price > self.volatility_criteria.max_price_usd:
                continue
                
            # Check volume
            volume_24h_usd = float(ticker.get('quoteVolume', '0'))
            if volume_24h_usd < self.liquidity_criteria.min_volume_24h_usd:
                continue
                
            # Check open interest
            oi_usd = open_interest.get(symbol, 0)
            if oi_usd < self.liquidity_criteria.min_open_interest_usd:
                continue
                
            # Check trades count
            trades_24h = int(ticker.get('count', '0'))
            if trades_24h < self.liquidity_criteria.min_trades_24h:
                continue
                
            # Calculate ATR approximation (using high-low range)
            high = float(ticker.get('high', '0'))
            low = float(ticker.get('low', '0'))
            if low > 0:
                atr_percent = ((high - low) / low) * 100
            else:
                atr_percent = 0
                
            # Check volatility criteria
            if atr_percent < self.volatility_criteria.min_atr_percent:
                continue
            if atr_percent > self.volatility_criteria.max_atr_percent:
                continue
                
            # Get fee rates (default for non-VIP)
            maker_fee = 0.0002  # 0.02%
            taker_fee = 0.0004  # 0.04%
            
            # Create filtered instrument
            instrument = FilteredInstrument(
                symbol=symbol,
                base_asset=raw_symbol.get('baseAsset', ''),
                quote_asset=raw_symbol.get('quoteAsset', ''),
                price_precision=filters['price_precision'],
                quantity_precision=filters['quantity_precision'],
                tick_size=filters['tick_size'],
                lot_size=filters['step_size'],
                min_notional=filters['min_notional'],
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                volume_24h_usd=volume_24h_usd,
                open_interest_usd=oi_usd,
                spread_bps=0.0,  # Would need orderbook fetch
                atr_percent=atr_percent,
                current_price=current_price,
                instrument_type="CryptoPerpetual",
                metadata={
                    'contract_type': raw_symbol.get('contractType', ''),
                    'underlying': raw_symbol.get('underlyingType', ''),
                }
            )
            
            qualified.append(instrument)
            
        # Sort by volume (most liquid first)
        qualified.sort(key=lambda x: x.volume_24h_usd, reverse=True)
        
        log.info(f"Filtered {len(qualified)} instruments from {len(exchange_info)} total")
        
        return qualified
        
    def generate_nautilus_instruments(
        self,
        filtered: List[FilteredInstrument],
    ) -> List[Dict[str, Any]]:
        """
        Generate Nautilus Trader Instrument definitions.
        
        Returns list of dictionaries suitable for creating Nautilus Instrument objects.
        """
        nautilus_instruments = []
        
        for inst in filtered:
            instrument_def = {
                'instrument_class': 'crypto_perpetual',
                'symbol': inst.symbol,
                'base_currency': inst.base_asset,
                'quote_currency': inst.quote_asset,
                'settlement_currency': inst.quote_asset,  # USDT settled
                'price_precision': inst.price_precision,
                'size_precision': inst.quantity_precision,
                'price_increment': inst.tick_size,
                'size_increment': inst.lot_size,
                'maker_fee': inst.maker_fee,
                'taker_fee': inst.taker_fee,
                'margin_ratio': 0.05,  # 5% initial margin (20x leverage)
                'max_leverage': 20,
                'min_notional': inst.min_notional,
                'info': {
                    'volume_24h_usd': inst.volume_24h_usd,
                    'open_interest_usd': inst.open_interest_usd,
                    'atr_percent': inst.atr_percent,
                    'current_price': inst.current_price,
                }
            }
            
            nautilus_instruments.append(instrument_def)
            
        return nautilus_instruments


async def select_universe(
    api_key: Optional[str] = None,
    is_testnet: bool = False,
    max_instruments: int = 20,
) -> List[Dict[str, Any]]:
    """
    Convenience function to select trading universe.
    
    Args:
        api_key: Optional Binance API key
        is_testnet: Use testnet endpoints
        max_instruments: Maximum number of instruments to return
        
    Returns:
        List of Nautilus Instrument definitions
    """
    selector = BinanceUniverseSelector(
        api_key=api_key,
        is_testnet=is_testnet,
    )
    
    try:
        filtered = await selector.filter_instruments()
        
        # Take top N by volume
        top_instruments = filtered[:max_instruments]
        
        # Generate Nautilus definitions
        nautilus_defs = selector.generate_nautilus_instruments(top_instruments)
        
        log.info(f"Selected {len(nautilus_defs)} instruments for trading universe")
        
        return nautilus_defs
        
    finally:
        await selector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def main():
        instruments = await select_universe(
            is_testnet=True,
            max_instruments=10,
        )
        
        print(f"\n=== Selected Trading Universe ({len(instruments)} instruments) ===\n")
        
        for inst in instruments:
            print(f"Symbol: {inst['symbol']}")
            print(f"  Base/Quote: {inst['base_currency']}/{inst['quote_currency']}")
            print(f"  Price Precision: {inst['price_precision']}")
            print(f"  Min Notional: ${inst['min_notional']}")
            print(f"  Maker/Taker Fee: {inst['maker_fee']:.4f} / {inst['taker_fee']:.4f}")
            print(f"  24h Volume: ${inst['info']['volume_24h_usd']:,.0f}")
            print(f"  ATR: {inst['info']['atr_percent']:.2f}%")
            print()
            
    asyncio.run(main())
