"""
Asynchronous X (Twitter) Streaming Scraper for Crypto Sentiment.
Uses raw aiohttp websockets with immediate hashing, spam filtering,
and memory queue limits to prevent overflow. Designed for <14GB RAM.
"""

import asyncio
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
import aiohttp


@dataclass
class Tweet:
    """Lightweight tweet container with minimal memory footprint."""
    id: str
    text: str
    author: str
    created_at: int  # Unix timestamp ms
    metrics: Dict[str, int] = field(default_factory=dict)
    is_verified: bool = False
    lang: str = "en"
    
    def __post_init__(self):
        # Drop any excess data immediately
        if len(self.text) > 280:
            self.text = self.text[:280]


class SpamFilter:
    """Memory-efficient spam and bot detection filter."""
    
    def __init__(self, max_seen_ids: int = 100000):
        self._seen_ids: deque = deque(maxlen=max_seen_ids)
        self._spam_patterns: Set[str] = {
            "follow me", "click here", "dm me", "telegram", 
            "whatsapp", "giveaway", "free crypto", "send eth",
            "double your", "100% guaranteed", "no risk",
        }
        self._author_rate_limit: Dict[str, deque] = {}
        self._max_posts_per_minute = 10
    
    def is_duplicate(self, tweet_id: str) -> bool:
        """Check if tweet was already processed."""
        if tweet_id in self._seen_ids:
            return True
        self._seen_ids.append(tweet_id)
        return False
    
    def is_spam(self, text: str) -> bool:
        """Detect spam based on patterns."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self._spam_patterns)
    
    def is_bot_author(self, author: str, timestamp_ms: int) -> bool:
        """Detect bot-like posting behavior."""
        if author not in self._author_rate_limit:
            self._author_rate_limit[author] = deque(maxlen=100)
        
        timestamps = self._author_rate_limit[author]
        timestamps.append(timestamp_ms)
        
        # Check posting frequency (last minute)
        one_minute_ago = timestamp_ms - 60000
        recent_posts = sum(1 for ts in timestamps if ts > one_minute_ago)
        
        return recent_posts > self._max_posts_per_minute
    
    def should_keep(self, tweet: Tweet) -> bool:
        """Full spam check for a tweet."""
        if self.is_duplicate(tweet.id):
            return False
        if self.is_spam(tweet.text):
            return False
        if self.is_bot_author(tweet.author, tweet.created_at):
            return False
        return True


class XStreamScraper:
    """
    Asynchronous X/Twitter streaming client using raw aiohttp websockets.
    Implements strict memory controls and immediate filtering.
    """
    
    # Target crypto accounts and keywords
    CRYPTO_ACCOUNTS = [
        "elonmusk", "VitalikButerin", "cz_binance", "SBF_FTX",
        "APompliano", "DocumentingBTC", "whale_alert", "CryptoKaleo",
        "Pentosh1", "CryptoCobain", "HsakaTrades", "TheMoonCarl",
    ]
    
    CRYPTO_KEYWORDS = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "defi",
        "altcoin", "trading", "cryptocurrency", "blockchain",
    ]
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        max_queue_size: int = 1000,
        enable_filtering: bool = True,
    ):
        self.bearer_token = bearer_token or os.getenv("X_BEARER_TOKEN", "")
        self.max_queue_size = max_queue_size
        self.enable_filtering = enable_filtering
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._tweet_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._spam_filter = SpamFilter()
        self._running = False
        self._stats = {
            "received": 0,
            "filtered": 0,
            "queued": 0,
            "errors": 0,
        }
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=5,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    def _hash_tweet(self, tweet: Tweet) -> str:
        """Generate deterministic hash for deduplication."""
        content = f"{tweet.author}:{tweet.text[:100]}:{tweet.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _connect_stream(self) -> aiohttp.ClientWebSocketResponse:
        """Connect to X streaming API."""
        await self._ensure_session()
        
        # Use filtered stream endpoint
        url = "https://api.twitter.com/2/tweets/search/stream"
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        
        # Build query rules
        rules_query = " OR ".join([f"({kw})" for kw in self.CRYPTO_KEYWORDS])
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,lang",
            "expansions": "author_id",
        }
        
        self._ws = await self._session.ws_connect(url, headers=headers, params=params)
        return self._ws
    
    async def _process_tweet(self, data: Dict[str, Any]) -> Optional[Tweet]:
        """Process raw tweet data into lightweight Tweet object."""
        try:
            tweet_data = data.get("data", {})
            includes = data.get("includes", {})
            
            tweet_id = tweet_data.get("id", "")
            text = tweet_data.get("text", "")
            created_at = tweet_data.get("created_at", "")
            lang = tweet_data.get("lang", "en")
            
            # Parse author info
            author_id = tweet_data.get("author_id", "")
            author_info = next(
                (a for a in includes.get("users", []) if a.get("id") == author_id),
                {}
            )
            author = author_info.get("username", "unknown")
            is_verified = author_info.get("verified", False)
            
            # Parse metrics
            metrics = tweet_data.get("public_metrics", {})
            
            # Convert timestamp
            try:
                from datetime import datetime
                dt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                created_ms = int(dt.timestamp() * 1000)
            except Exception:
                created_ms = int(time.time() * 1000)
            
            tweet = Tweet(
                id=tweet_id,
                text=text,
                author=author,
                created_at=created_ms,
                metrics={
                    "retweets": metrics.get("retweet_count", 0),
                    "likes": metrics.get("like_count", 0),
                    "replies": metrics.get("reply_count", 0),
                },
                is_verified=is_verified,
                lang=lang,
            )
            
            return tweet
            
        except Exception as e:
            self._stats["errors"] += 1
            return None
    
    async def _filter_and_queue(self, tweet: Tweet):
        """Apply filters and queue valid tweets."""
        self._stats["received"] += 1
        
        if self.enable_filtering:
            if not self._spam_filter.should_keep(tweet):
                self._stats["filtered"] += 1
                return
        
        # Queue with backpressure handling
        try:
            self._tweet_queue.put_nowait(tweet)
            self._stats["queued"] += 1
        except asyncio.QueueFull:
            # Drop oldest if queue full (memory protection)
            try:
                self._tweet_queue.get_nowait()
                self._tweet_queue.put_nowait(tweet)
                self._stats["queued"] += 1
            except Exception:
                pass
    
    async def stream(
        self,
        callback: Optional[Callable[[Tweet], None]] = None,
        duration_seconds: Optional[int] = None,
    ):
        """
        Stream tweets from X with optional callback processing.
        
        Args:
            callback: Async function to call for each tweet
            duration_seconds: How long to stream (None = indefinite)
        """
        self._running = True
        start_time = time.time()
        
        try:
            ws = await self._connect_stream()
            
            while self._running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Receive message
                msg = await ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Skip keep-alive messages
                        if "data" not in data:
                            continue
                        
                        tweet = await self._process_tweet(data)
                        if tweet:
                            if callback:
                                await callback(tweet)
                            else:
                                await self._filter_and_queue(tweet)
                    
                    except json.JSONDecodeError:
                        self._stats["errors"] += 1
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self._stats["errors"] += 1
                    await asyncio.sleep(5)  # Backoff
                    ws = await self._connect_stream()
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    await asyncio.sleep(5)
                    ws = await self._connect_stream()
        
        except Exception as e:
            self._stats["errors"] += 1
            raise
        finally:
            self._running = False
    
    async def get_tweets(self, max_count: int = 100) -> List[Tweet]:
        """Retrieve queued tweets."""
        tweets = []
        while len(tweets) < max_count and not self._tweet_queue.empty():
            try:
                tweet = self._tweet_queue.get_nowait()
                tweets.append(tweet)
            except asyncio.QueueEmpty:
                break
        return tweets
    
    def get_stats(self) -> Dict[str, int]:
        """Get streaming statistics."""
        return self._stats.copy()
    
    async def stop(self):
        """Stop the streaming connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


async def main():
    """Example usage of X stream scraper."""
    async with XStreamScraper() as scraper:
        print("Starting X stream scraper (demo mode)...")
        print(f"Tracking keywords: {scraper.CRYPTO_KEYWORDS[:5]}...")
        
        # In production, this would connect to real API
        # For demo, we'll show the structure
        
        print("\nScraper initialized successfully!")
        print(f"Max queue size: {scraper.max_queue_size}")
        print(f"Filtering enabled: {scraper.enable_filtering}")


if __name__ == "__main__":
    asyncio.run(main())
