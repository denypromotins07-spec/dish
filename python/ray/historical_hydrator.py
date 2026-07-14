"""
Ray Historical Data Hydrator.

Ray tasks for heavy but non-latency-critical background jobs,
such as downloading historical tick data and pre-computing initial
order book states before the live trading session begins.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import ray

log = logging.getLogger(__name__)


@dataclass
class HistoricalDataConfig:
    """Configuration for historical data hydration."""
    
    # Symbols to hydrate
    symbols: List[str] = None
    
    # Date range
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-31"
    
    # Data types
    download_trades: bool = True
    download_orderbook: bool = True
    download_klines: bool = True
    
    # Storage settings
    output_directory: str = "/data/historical"
    compression: str = "zstd"  # High compression, fast decompression
    
    # Rate limiting
    requests_per_second: int = 10
    max_retries: int = 3
    
    # Memory limits per task
    max_memory_per_task_mb: int = 500
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@ray.remote(max_calls=1)
class HistoricalDataDownloader:
    """
    Ray actor for downloading historical data from Binance.
    
    Designed to run as a background task with strict memory limits
    to avoid impacting the live trading engine.
    """
    
    def __init__(self, config: HistoricalDataConfig):
        self.config = config
        self.session = None
        self.stats = {
            "symbols_processed": 0,
            "days_processed": 0,
            "total_records": 0,
            "errors": 0,
        }
        
    async def _get_session(self):
        """Get aiohttp session."""
        if self.session is None:
            import aiohttp
            connector = aiohttp.TCPConnector(limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self.session
        
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            
    async def fetch_trades(
        self,
        symbol: str,
        date: str,
    ) -> List[Dict[str, Any]]:
        """Fetch historical trades for a specific date."""
        import aiohttp
        import orjson
        
        session = await self._get_session()
        base_url = "https://api.binance.com/api/v3/aggTrades"
        
        start_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = start_ts + 86400000  # +24 hours
        
        all_trades = []
        from_id = None
        
        while True:
            params = {
                "symbol": symbol,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": 1000,
            }
            
            if from_id:
                params["fromId"] = from_id
                
            try:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        text = await response.text()
                        log.warning(f"API error: {text}")
                        break
                        
                    data = await response.read()
                    trades = orjson.loads(data)
                    
                    if not trades:
                        break
                        
                    all_trades.extend(trades)
                    
                    # Update for next page
                    from_id = trades[-1]["a"]
                    
                    # Check if we've reached the end
                    if trades[-1]["T"] >= end_ts:
                        break
                        
            except Exception as e:
                log.error(f"Error fetching trades: {e}")
                self.stats["errors"] += 1
                break
                
        self.stats["total_records"] += len(all_trades)
        return all_trades
        
    async def fetch_klines(
        self,
        symbol: str,
        date: str,
        interval: str = "1m",
    ) -> List[Dict[str, Any]]:
        """Fetch historical klines (candlesticks)."""
        import aiohttp
        import orjson
        
        session = await self._get_session()
        base_url = "https://api.binance.com/api/v3/klines"
        
        start_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = start_ts + 86400000
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        
        try:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.read()
                klines = orjson.loads(data)
                
                # Convert to dict format
                result = []
                for k in klines:
                    result.append({
                        "open_time": k[0],
                        "open": k[1],
                        "high": k[2],
                        "low": k[3],
                        "close": k[4],
                        "volume": k[5],
                        "close_time": k[6],
                        "quote_volume": k[7],
                        "trades_count": k[8],
                    })
                    
                return result
                
        except Exception as e:
            log.error(f"Error fetching klines: {e}")
            return []
            
    async def process_symbol_day(
        self,
        symbol: str,
        date: str,
    ) -> Dict[str, Any]:
        """Process a single symbol-day combination."""
        log.info(f"Processing {symbol} for {date}")
        
        result = {
            "symbol": symbol,
            "date": date,
            "trades": [],
            "klines": [],
            "status": "success",
        }
        
        # Fetch trades
        if self.config.download_trades:
            trades = await self.fetch_trades(symbol, date)
            result["trades"] = trades
            result["trade_count"] = len(trades)
            
        # Fetch klines
        if self.config.download_klines:
            klines = await self.fetch_klines(symbol, date)
            result["klines"] = klines
            result["kline_count"] = len(klines)
            
        self.stats["days_processed"] += 1
        
        return result
        
    async def hydrate_all(self) -> Dict[str, Any]:
        """Hydrate all configured symbols and dates."""
        import asyncio
        
        log.info(
            f"Starting historical data hydration: "
            f"{len(self.config.symbols)} symbols, "
            f"{self.config.start_date} to {self.config.end_date}"
        )
        
        # Generate date range
        start = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
            
        log.info(f"Date range: {len(dates)} days")
        
        # Process all combinations
        tasks = []
        for symbol in self.config.symbols:
            for date in dates:
                task = self.process_symbol_day(symbol, date)
                tasks.append(task)
                
        # Execute with rate limiting
        results = []
        semaphore = asyncio.Semaphore(self.config.requests_per_second)
        
        async def limited_task(task):
            async with semaphore:
                return await task
                
        limited_tasks = [limited_task(t) for t in tasks]
        
        for coro in asyncio.as_completed(limited_tasks):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                log.error(f"Task failed: {e}")
                self.stats["errors"] += 1
                
        self.stats["symbols_processed"] = len(self.config.symbols)
        
        log.info(
            f"Hydration complete: {self.stats['days_processed']} days, "
            f"{self.stats['total_records']} trades, {self.stats['errors']} errors"
        )
        
        return {
            "results": results,
            "stats": self.stats,
        }


@ray.remote
def download_historical_trades(
    symbol: str,
    date: str,
) -> Dict[str, Any]:
    """
    Standalone Ray task to download historical trades.
    
    Can be distributed across multiple workers for parallel processing.
    """
    import asyncio
    import aiohttp
    import orjson
    
    async def fetch():
        connector = aiohttp.TCPConnector(limit=5)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            base_url = "https://api.binance.com/api/v3/aggTrades"
            
            start_ts = int(datetime.strptime(date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = start_ts + 86400000
            
            all_trades = []
            from_id = None
            
            while True:
                params = {
                    "symbol": symbol,
                    "startTime": start_ts,
                    "endTime": end_ts,
                    "limit": 1000,
                }
                
                if from_id:
                    params["fromId"] = from_id
                    
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        break
                        
                    data = await response.read()
                    trades = orjson.loads(data)
                    
                    if not trades:
                        break
                        
                    all_trades.extend(trades)
                    from_id = trades[-1]["a"]
                    
                    if trades[-1]["T"] >= end_ts:
                        break
                        
            return {
                "symbol": symbol,
                "date": date,
                "trade_count": len(all_trades),
                "trades": all_trades,
            }
            
    return asyncio.run(fetch())


@ray.remote
def compute_orderbook_snapshot(
    trades: List[Dict[str, Any]],
    initial_book: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Reconstruct order book snapshot from trade data.
    
    Runs as a Ray task for parallel processing of historical data.
    """
    # Simplified order book reconstruction
    # In production, would use the Rust orderbook_reconstructor
    
    bids = {}
    asks = {}
    
    for trade in trades:
        price = float(trade["p"])
        qty = float(trade["q"])
        is_maker = trade.get("m", False)
        
        # Simplified: just track last traded price levels
        if is_maker:
            bids[price] = bids.get(price, 0) + qty
        else:
            asks[price] = asks.get(price, 0) + qty
            
    # Sort and take top levels
    sorted_bids = sorted(bids.items(), key=lambda x: -x[0])[:20]
    sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:20]
    
    return {
        "bids": [{"price": p, "qty": q} for p, q in sorted_bids],
        "asks": [{"price": p, "qty": q} for p, q in sorted_asks],
        "trade_count": len(trades),
    }


async def run_historical_hydration(config: HistoricalDataConfig) -> Dict[str, Any]:
    """
    Main entry point for running historical data hydration.
    
    Initializes Ray if needed and orchestrates the download process.
    """
    import ray
    
    if not ray.is_initialized():
        ray.init(
            num_cpus=4,
            _memory=2 * 1024 * 1024 * 1024,  # 2GB limit
        )
        
    # Create downloader actor
    downloader = HistoricalDataDownloader.remote(config)
    
    # Run hydration
    result_ref = downloader.hydrate_all.remote()
    result = await result_ref
    
    # Cleanup
    await downloader.close.remote()
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = HistoricalDataConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        start_date="2024-01-01",
        end_date="2024-01-03",  # Short range for testing
        download_trades=True,
        download_klines=True,
    )
    
    print("Starting historical data hydration...")
    result = asyncio.run(run_historical_hydration(config))
    
    print(f"\nResults:")
    print(f"  Days processed: {result['stats']['days_processed']}")
    print(f"  Total trades: {result['stats']['total_records']}")
    print(f"  Errors: {result['stats']['errors']}")
