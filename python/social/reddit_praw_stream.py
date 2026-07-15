"""
Reddit PRAW Streaming Client for Crypto Subreddits.
Extracts text and upvote metrics with strict queue limits
and automatic garbage collection to prevent memory leaks.
Designed for <14GB RAM constraint.
"""

import asyncio
import gc
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set


@dataclass
class RedditPost:
    """Lightweight Reddit post container."""
    id: str
    title: str
    text: str
    author: str
    subreddit: str
    created_utc: int
    score: int
    upvote_ratio: float
    num_comments: int
    url: str = ""
    flair: str = ""
    
    def __post_init__(self):
        # Truncate long text to save memory
        if len(self.text) > 5000:
            self.text = self.text[:5000]
        if len(self.title) > 300:
            self.title = self.title[:300]


class RedditMemoryManager:
    """Automatic memory management for Reddit streaming."""
    
    def __init__(
        self,
        max_posts_tracked: int = 50000,
        memory_threshold_mb: float = 500.0,
    ):
        self.max_posts_tracked = max_posts_tracked
        self.memory_threshold_mb = memory_threshold_mb
        
        self._seen_ids: deque = deque(maxlen=max_posts_tracked)
        self._processed_count = 0
        self._gc_calls = 0
    
    def is_duplicate(self, post_id: str) -> bool:
        """Check if post was already processed."""
        if post_id in self._seen_ids:
            return True
        self._seen_ids.append(post_id)
        return False
    
    def check_memory_and_gc(self) -> bool:
        """Check memory usage and trigger GC if needed."""
        try:
            import resource
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on Linux
            
            if mem_usage > self.memory_threshold_mb:
                self._gc_calls += 1
                gc.collect()
                return True
        except Exception:
            pass
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory manager statistics."""
        return {
            "tracked_ids": len(self._seen_ids),
            "processed_count": self._processed_count,
            "gc_calls": self._gc_calls,
        }


class RedditPRAWStream:
    """
    Reddit PRAW streaming client for crypto subreddits.
    Implements strict memory controls and automatic cleanup.
    """
    
    # Target crypto subreddits
    CRYPTO_SUBREDDITS = [
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "ethtrader",
        "CryptoMarkets",
        "altcoin",
        "defi",
        "NFT",
        "btc",
        "binance",
    ]
    
    # Keywords to filter relevant posts
    RELEVANT_KEYWORDS = [
        "price", "pump", "dump", "moon", "crash", "bull", "bear",
        "buy", "sell", "hold", "trade", "analysis", "prediction",
        "news", "adoption", "regulation", "sec", "etf", "halving",
    ]
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "crypto-trading-bot/1.0",
        max_queue_size: int = 500,
        enable_memory_management: bool = True,
    ):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = user_agent
        self.max_queue_size = max_queue_size
        self.enable_memory_management = enable_memory_management
        
        self._reddit = None
        self._post_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._memory_manager = RedditMemoryManager() if enable_memory_management else None
        self._running = False
        self._stats = {
            "received": 0,
            "filtered": 0,
            "queued": 0,
            "errors": 0,
        }
    
    def _ensure_praw(self):
        """Ensure PRAW is initialized."""
        if self._reddit is None:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                    read_only=True,
                )
            except ImportError:
                raise ImportError("praw library required. Install with: pip install praw")
    
    def _is_relevant(self, title: str, text: str) -> bool:
        """Check if post contains relevant trading keywords."""
        content = f"{title} {text}".lower()
        return any(kw in content for kw in self.RELEVANT_KEYWORDS)
    
    def _parse_post(self, submission: Any) -> RedditPost:
        """Convert PRAW submission to lightweight RedditPost."""
        return RedditPost(
            id=submission.id,
            title=submission.title or "",
            text=submission.selftext or "",
            author=str(submission.author) if submission.author else "[deleted]",
            subreddit=submission.subreddit.display_name,
            created_utc=int(submission.created_utc * 1000),
            score=submission.score or 0,
            upvote_ratio=submission.upvote_ratio or 0.5,
            num_comments=submission.num_comments or 0,
            url=submission.url or "",
            flair=submission.link_flair_text or "",
        )
    
    async def _process_submission(self, submission: Any) -> Optional[RedditPost]:
        """Process and validate a submission."""
        try:
            post = self._parse_post(submission)
            
            # Check for duplicates
            if self._memory_manager and self._memory_manager.is_duplicate(post.id):
                self._stats["filtered"] += 1
                return None
            
            # Filter by relevance
            if not self._is_relevant(post.title, post.text):
                self._stats["filtered"] += 1
                return None
            
            self._stats["received"] += 1
            
            if self._memory_manager:
                self._memory_manager._processed_count += 1
            
            return post
            
        except Exception as e:
            self._stats["errors"] += 1
            return None
    
    async def _queue_post(self, post: RedditPost):
        """Queue post with backpressure handling."""
        try:
            self._post_queue.put_nowait(post)
            self._stats["queued"] += 1
        except asyncio.QueueFull:
            # Drop oldest post if queue full (memory protection)
            try:
                self._post_queue.get_nowait()
                self._post_queue.put_nowait(post)
                self._stats["queued"] += 1
            except Exception:
                pass
    
    async def stream_subreddit(
        self,
        subreddit_name: str,
        callback: Optional[Callable[[RedditPost], None]] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[RedditPost, None]:
        """
        Stream posts from a single subreddit.
        
        Args:
            subreddit_name: Name of subreddit to stream
            callback: Optional async callback for each post
            limit: Maximum posts to yield (None = unlimited)
        """
        self._ensure_praw()
        self._running = True
        
        subreddit = self._reddit.subreddit(subreddit_name)
        count = 0
        
        try:
            # Use stream API for real-time updates
            for submission in subreddit.stream.submissions(pause_after=30):
                if not self._running:
                    break
                
                if submission is None:
                    # No new posts, check memory
                    if self._memory_manager:
                        self._memory_manager.check_memory_and_gc()
                    continue
                
                post = await self._process_submission(submission)
                if post:
                    if callback:
                        await callback(post)
                    else:
                        await self._queue_post(post)
                    
                    yield post
                    count += 1
                    
                    if limit and count >= limit:
                        break
                
                # Periodic memory check
                if count % 100 == 0 and self._memory_manager:
                    self._memory_manager.check_memory_and_gc()
        
        finally:
            self._running = False
    
    async def stream_all_subreddits(
        self,
        callback: Optional[Callable[[RedditPost], None]] = None,
        duration_seconds: Optional[int] = None,
    ):
        """
        Stream from all configured crypto subreddits concurrently.
        
        Args:
            callback: Async callback for each post
            duration_seconds: How long to stream (None = indefinite)
        """
        self._running = True
        start_time = time.time()
        
        tasks = [
            asyncio.create_task(
                self._stream_single_with_timeout(sub, callback, duration_seconds)
            )
            for sub in self.CRYPTO_SUBREDDITS
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._running = False
    
    async def _stream_single_with_timeout(
        self,
        subreddit: str,
        callback: Optional[Callable],
        duration: Optional[int],
    ):
        """Stream single subreddit with timeout."""
        try:
            async for post in self.stream_subreddit(subreddit, callback):
                if duration and (time.time() - start_time) > duration:
                    break
        except Exception as e:
            self._stats["errors"] += 1
    
    async def get_posts(self, max_count: int = 100) -> List[RedditPost]:
        """Retrieve queued posts."""
        posts = []
        while len(posts) < max_count and not self._post_queue.empty():
            try:
                post = self._post_queue.get_nowait()
                posts.append(post)
            except asyncio.QueueEmpty:
                break
        return posts
    
    def get_stats(self) -> Dict[str, int]:
        """Get streaming statistics."""
        stats = self._stats.copy()
        if self._memory_manager:
            stats.update(self._memory_manager.get_stats())
        return stats
    
    async def stop(self):
        """Stop all streaming."""
        self._running = False
        if self._reddit:
            self._reddit = None
    
    async def __aenter__(self):
        self._ensure_praw()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def main():
    """Example usage of Reddit PRAW stream client."""
    print("Reddit PRAW Stream Client")
    print("=" * 50)
    
    scraper = RedditPRAWStream()
    
    print(f"Target subreddits: {scraper.CRYPTO_SUBREDDITS[:5]}...")
    print(f"Max queue size: {scraper.max_queue_size}")
    print(f"Memory management: {scraper.enable_memory_management}")
    print("\nClient initialized successfully!")
    
    # Show how to use (would require valid credentials in production)
    print("\nUsage example:")
    print("  async with RedditPRAWStream() as reddit:")
    print("      async for post in reddit.stream_subreddit('CryptoCurrency'):")
    print("          print(f'{post.subreddit}: {post.title}')")


if __name__ == "__main__":
    main()
