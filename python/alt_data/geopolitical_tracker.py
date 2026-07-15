"""
Geopolitical Event Tracker using RSS feeds and keyword matching.
Triggers immediate risk-off regime flags and dynamic stop-loss tightening.
Lightweight regex-based approach for <14GB RAM constraint.
"""

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import aiohttp


@dataclass
class GeopoliticalEvent:
    """Detected geopolitical event."""
    id: str
    timestamp_ms: int
    source: str
    title: str
    summary: str
    url: str
    risk_level: str        # LOW, MEDIUM, HIGH, CRITICAL
    categories: List[str]  # war, sanction, election, etc.
    affected_regions: List[str]
    processed: bool = False
    
    @property
    def requires_risk_off(self) -> bool:
        """Check if event triggers risk-off mode."""
        return self.risk_level in ("HIGH", "CRITICAL")


class GeopoliticalKeywordMatcher:
    """
    High-performance keyword matcher for geopolitical events.
    Uses compiled regex patterns for fast matching.
    """
    
    # Risk level keywords with associated severity
    RISK_KEYWORDS = {
        "CRITICAL": [
            r"\bwar\b", r"\bconflict\b", r"\binvasion\b", r"\battack\b",
            r"\bmissile\b", r"\bbombing\b", r"\bstrike\b", r"\bcrisis\b",
            r"\bemergency\b", r"\bmartial law\b", r"\bcoup\b",
        ],
        "HIGH": [
            r"\bsanction\b", r"\bembargo\b", r"\btrade war\b", r"\btariff\b",
            r"\belection\b", r"\breferendum\b", r"\bprotest\b", r"\briot\b",
            r"\bunrest\b", r"\btension\b", r"\bthreat\b", r"\bescalat",
        ],
        "MEDIUM": [
            r"\bnegotiation\b", r"\bdiplomat\b", r"\btreaty\b", r"\bagreement\b",
            r"\bsummit\b", r"\bmeeting\b", r"\btalks\b", r"\bdebate\b",
        ],
        "LOW": [
            r"\bpolicy\b", r"\bregulation\b", r"\bannouncement\b", r"\bstatement\b",
        ],
    }
    
    # Category detection patterns
    CATEGORY_PATTERNS = {
        "war": [r"\bwar\b", r"\bconflict\b", r"\binvasion\b", r"\bmilitary\b"],
        "sanction": [r"\bsanction\b", r"\bembargo\b", r"\btrade restriction\b"],
        "election": [r"\belection\b", r"\bvoting\b", r"\bpoll\b", r"\bballot\b"],
        "protest": [r"\bprotest\b", r"\bdemonstration\b", r"\brally\b", r"\bmarch\b"],
        "policy": [r"\bpolicy\b", r"\bregulation\b", r"\blaw\b", r"\blegislation\b"],
        "diplomatic": [r"\bdiplomat\b", r"\bembassy\b", r"\btreaty\b", r"\bsummit\b"],
        "economic": [r"\btrade\b", r"\btariff\b", r"\beconomic\b", r"\bfinance\b"],
    }
    
    # Region detection patterns
    REGION_PATTERNS = {
        "US": [r"\bUS\b", r"\bUnited States\b", r"\bAmerica\b", r"\bWashington\b"],
        "EU": [r"\bEU\b", r"\bEuropean Union\b", r"\bEurope\b", r"\bBrussels\b"],
        "CN": [r"\bChina\b", r"\bChinese\b", r"\bBeijing\b"],
        "RU": [r"\bRussia\b", r"\bRussian\b", r"\bMoscow\b", r"\bKremlin\b"],
        "UK": [r"\bUK\b", r"\bBritain\b", r"\bBritish\b", r"\bLondon\b"],
        "JP": [r"\bJapan\b", r"\bJapanese\b", r"\bTokyo\b"],
        "KR": [r"\bKorea\b", r"\bKorean\b", r"\bSeoul\b"],
        "IR": [r"\bIran\b", r"\bIranian\b", r"\bTehran\b"],
        "ME": [r"\bMiddle East\b", r"\bIsrael\b", r"\bPalestine\b", r"\bGaza\b"],
    }
    
    def __init__(self):
        # Pre-compile all patterns for performance
        self._compiled_risk: Dict[str, List[re.Pattern]] = {}
        self._compiled_categories: Dict[str, List[re.Pattern]] = {}
        self._compiled_regions: Dict[str, List[re.Pattern]] = {}
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for fast matching."""
        for risk_level, patterns in self.RISK_KEYWORDS.items():
            self._compiled_risk[risk_level] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            self._compiled_categories[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for region, patterns in self.REGION_PATTERNS.items():
            self._compiled_regions[region] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def detect_risk_level(self, text: str) -> str:
        """Detect highest risk level in text."""
        for risk_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            for pattern in self._compiled_risk.get(risk_level, []):
                if pattern.search(text):
                    return risk_level
        return "LOW"
    
    def detect_categories(self, text: str) -> List[str]:
        """Detect event categories in text."""
        categories = []
        for category, patterns in self._compiled_categories.items():
            for pattern in patterns:
                if pattern.search(text):
                    categories.append(category)
                    break
        return categories
    
    def detect_regions(self, text: str) -> List[str]:
        """Detect affected regions in text."""
        regions = []
        for region, patterns in self._compiled_regions.items():
            for pattern in patterns:
                if pattern.search(text):
                    regions.append(region)
                    break
        return regions
    
    def analyze(self, title: str, summary: str) -> Tuple[str, List[str], List[str]]:
        """Full analysis of text content."""
        combined = f"{title} {summary}"
        
        risk_level = self.detect_risk_level(combined)
        categories = self.detect_categories(combined)
        regions = self.detect_regions(combined)
        
        return risk_level, categories, regions


class GeopoliticalTracker:
    """
    Real-time geopolitical event tracker using RSS feeds.
    Triggers risk-off flags for high-impact events.
    """
    
    # RSS feed sources for geopolitical news
    RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/worldNews",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ]
    
    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        check_interval_seconds: int = 60,
        max_history_events: int = 1000,
    ):
        self._session = session
        self._check_interval = check_interval_seconds
        self._max_history = max_history_events
        
        self._matcher = GeopoliticalKeywordMatcher()
        self._events: deque = deque(maxlen=max_history_events)
        self._seen_ids: Set[str] = set()
        self._risk_off_active = False
        self._risk_off_until_ms: int = 0
        self._current_risk_level = "LOW"
        self._stats = {
            "feeds_checked": 0,
            "events_detected": 0,
            "risk_off_triggers": 0,
            "errors": 0,
        }
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=5, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session
    
    def _generate_event_id(self, title: str, source: str) -> str:
        """Generate unique event ID."""
        import hashlib
        content = f"{title}:{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def parse_rss_feed(self, feed_url: str) -> List[Dict[str, str]]:
        """Parse RSS feed and extract entries."""
        try:
            session = await self._ensure_session()
            
            async with session.get(feed_url) as response:
                if response.status != 200:
                    return []
                
                content = await response.text()
                
                # Simple RSS parsing (in production, use feedparser)
                entries = []
                
                # Extract items
                item_pattern = re.compile(r'<item>(.*?)</item>', re.DOTALL)
                title_pattern = re.compile(r'<title>(.*?)</title>')
                desc_pattern = re.compile(r'<description>(.*?)</description>')
                link_pattern = re.compile(r'<link>(.*?)</link>')
                pub_date_pattern = re.compile(r'<pubDate>(.*?)</pubDate>')
                
                for item_match in item_pattern.finditer(content):
                    item_content = item_match.group(1)
                    
                    title_match = title_pattern.search(item_content)
                    desc_match = desc_pattern.search(item_content)
                    link_match = link_pattern.search(item_content)
                    date_match = pub_date_pattern.search(item_content)
                    
                    entries.append({
                        "title": title_match.group(1) if title_match else "",
                        "summary": desc_match.group(1) if desc_match else "",
                        "url": link_match.group(1) if link_match else "",
                        "published": date_match.group(1) if date_match else "",
                        "source": feed_url,
                    })
                
                return entries
                
        except Exception as e:
            self._stats["errors"] += 1
            return []
    
    async def process_entry(self, entry: Dict[str, str]) -> Optional[GeopoliticalEvent]:
        """Process a single RSS entry and detect if it's a geopolitical event."""
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        
        # Analyze content
        risk_level, categories, regions = self._matcher.analyze(title, summary)
        
        # Only track events with some significance
        if risk_level == "LOW" and not categories:
            return None
        
        event_id = self._generate_event_id(title, entry.get("source", ""))
        
        # Check for duplicates
        if event_id in self._seen_ids:
            return None
        
        self._seen_ids.add(event_id)
        
        # Clean up seen_ids to prevent memory bloat
        if len(self._seen_ids) > self._max_history * 2:
            # Keep only recent IDs
            self._seen_ids = set(list(self._seen_ids)[-self._max_history:])
        
        event = GeopoliticalEvent(
            id=event_id,
            timestamp_ms=int(time.time() * 1000),
            source=entry.get("source", ""),
            title=title,
            summary=summary[:500],  # Limit summary length
            url=entry.get("url", ""),
            risk_level=risk_level,
            categories=categories,
            affected_regions=regions,
        )
        
        self._events.append(event)
        self._stats["events_detected"] += 1
        
        # Trigger risk-off if needed
        if event.requires_risk_off:
            self._trigger_risk_off(event)
        
        return event
    
    def _trigger_risk_off(self, event: GeopoliticalEvent):
        """Activate risk-off mode based on event severity."""
        self._risk_off_active = True
        self._current_risk_level = event.risk_level
        
        # Duration based on risk level
        duration_minutes = {
            "CRITICAL": 120,  # 2 hours
            "HIGH": 60,       # 1 hour
            "MEDIUM": 30,     # 30 minutes
            "LOW": 0,
        }.get(event.risk_level, 0)
        
        self._risk_off_until_ms = int(time.time() * 1000) + (duration_minutes * 60000)
        self._stats["risk_off_triggers"] += 1
    
    def update_risk_status(self) -> Dict[str, Any]:
        """Check and update current risk status."""
        now_ms = int(time.time() * 1000)
        
        if now_ms >= self._risk_off_until_ms:
            self._risk_off_active = False
            self._current_risk_level = "LOW"
        
        return {
            "risk_off_active": self._risk_off_active,
            "risk_level": self._current_risk_level,
            "risk_off_until_ms": self._risk_off_until_ms,
            "seconds_remaining": max(0, (self._risk_off_until_ms - now_ms) // 1000),
            "recent_events": len([e for e in self._events if not e.processed]),
        }
    
    async def check_all_feeds(self) -> List[GeopoliticalEvent]:
        """Check all RSS feeds for new events."""
        self._stats["feeds_checked"] += len(self.RSS_FEEDS)
        
        tasks = [self.parse_rss_feed(feed) for feed in self.RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        new_events = []
        
        for result in results:
            if isinstance(result, list):
                for entry in result:
                    event = await self.process_entry(entry)
                    if event:
                        new_events.append(event)
        
        return new_events
    
    def get_recent_events(
        self,
        limit: int = 10,
        min_risk_level: str = "LOW",
    ) -> List[GeopoliticalEvent]:
        """Get recent events filtered by risk level."""
        risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        min_index = risk_order.index(min_risk_level)
        
        filtered = [
            e for e in reversed(self._events)
            if risk_order.index(e.risk_level) >= min_index
        ]
        
        return filtered[:limit]
    
    def get_stop_loss_multiplier(self) -> float:
        """
        Get dynamic stop-loss multiplier based on geopolitical risk.
        Tightens stops during high-risk periods.
        """
        multipliers = {
            "CRITICAL": 0.5,   # 50% tighter stops
            "HIGH": 0.7,       # 30% tighter
            "MEDIUM": 0.85,    # 15% tighter
            "LOW": 1.0,        # Normal
        }
        return multipliers.get(self._current_risk_level, 1.0)
    
    def get_stats(self) -> Dict[str, int]:
        """Get tracker statistics."""
        return self._stats.copy()
    
    async def close(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def main():
    """Example usage of Geopolitical Tracker."""
    print("Geopolitical Event Tracker")
    print("=" * 50)
    
    tracker = GeopoliticalTracker()
    
    print(f"RSS Feeds monitored: {len(tracker.RSS_FEEDS)}")
    print(f"Check interval: {tracker._check_interval}s")
    print(f"Max history: {tracker._max_history} events")
    
    print("\nRisk Categories:")
    for category in tracker._matcher.CATEGORY_PATTERNS.keys():
        print(f"  - {category}")
    
    print("\nStop-Loss Multipliers:")
    print("  CRITICAL: 0.5x (50% tighter)")
    print("  HIGH: 0.7x (30% tighter)")
    print("  MEDIUM: 0.85x (15% tighter)")
    print("  LOW: 1.0x (normal)")
    
    print("\nTracker initialized successfully!")


if __name__ == "__main__":
    main()
