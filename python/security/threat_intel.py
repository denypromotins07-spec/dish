#!/usr/bin/env python3
"""
Lightweight threat intelligence feed aggregator.
Checks incoming IP addresses and webhook sources against known malicious botnet
and MEV-searcher databases, dropping connections from flagged entities instantly.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of known threats."""
    BOTNET = "botnet"
    MEV_SEARCHER = "mev_searcher"
    SCAMMER = "scammer"
    SANCTIONED = "sanctioned"
    BRUTE_FORCE = "brute_force"
    REPUTATION_LOW = "reputation_low"


class ThreatLevel(Enum):
    """Severity levels for threats."""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25


@dataclass
class ThreatEntry:
    """Represents a single threat entry."""
    indicator: str  # IP address, domain, or hash
    category: ThreatCategory
    level: ThreatLevel
    source: str
    first_seen: datetime
    last_seen: datetime
    confidence: float  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class ThreatIntelligenceFeed:
    """
    Aggregates threat intelligence from multiple sources.
    Provides real-time lookup for IP addresses and other indicators.
    """

    def __init__(
        self,
        update_interval_minutes: int = 30,
        cache_file: Optional[str] = None,
    ):
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self.cache_file = Path(cache_file) if cache_file else None
        
        # In-memory threat database
        self._threats_by_ip: Dict[str, ThreatEntry] = {}
        self._threats_by_hash: Dict[str, ThreatEntry] = {}
        
        # Known MEV searcher addresses (Ethereum examples)
        self._known_mev_searchers: Set[str] = {
            "0x6b75d8af000000e20b7a7ddf000ba900b4009a80",  # Common MEV bot
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap router (can be abused)
        }
        
        # Private IP ranges (always trusted)
        self._trusted_ranges = [
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("127.0.0.0/8"),
        ]
        
        # Last update time
        self._last_update: Optional[datetime] = None
        self._is_running = False

    async def start(self) -> None:
        """Starts the background threat feed updater."""
        self._is_running = True
        await self._load_cache()
        await self._update_feeds()
        
        # Start background updater
        asyncio.create_task(self._updater_loop())

    async def stop(self) -> None:
        """Stops the updater and saves cache."""
        self._is_running = False
        await self._save_cache()

    async def _updater_loop(self) -> None:
        """Background loop to periodically update threat feeds."""
        while self._is_running:
            await asyncio.sleep(self.update_interval.total_seconds())
            await self._update_feeds()

    async def _update_feeds(self) -> None:
        """Fetches updates from configured threat intelligence sources."""
        logger.info("Updating threat intelligence feeds...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch from multiple sources (example URLs - replace with actual feeds)
                tasks = [
                    self._fetch_abuse_ch(session),
                    self._fetch_emerging_threats(session),
                    self._fetch_custom_blocklist(session),
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self._last_update = datetime.now(timezone.utc)
            logger.info(f"Threat feeds updated at {self._last_update}")
            
        except Exception as e:
            logger.error(f"Failed to update threat feeds: {e}")

    async def _fetch_abuse_ch(self, session: aiohttp.ClientSession) -> None:
        """Fetches IP blocklist from Abuse.ch."""
        # Example: https://feodotracker.abuse.ch/downloads/ipblocklist.txt
        # This is a placeholder - implement actual feed parsing
        pass

    async def _fetch_emerging_threats(self, session: aiohttp.ClientSession) -> None:
        """Fetches rules from Emerging Threats."""
        # Example: https://rules.emergingthreats.net/open/suricata/emerging-all.rules
        pass

    async def _fetch_custom_blocklist(self, session: aiohttp.ClientSession) -> None:
        """Fetches custom blocklist from internal source."""
        # Implement custom feed fetching
        pass

    async def _load_cache(self) -> None:
        """Loads cached threat data from disk."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
            
            for entry_data in data.get("threats", []):
                entry = self._parse_threat_entry(entry_data)
                if entry:
                    self._index_threat(entry)
            
            logger.info(f"Loaded {len(self._threats_by_ip)} threats from cache")
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    async def _save_cache(self) -> None:
        """Saves current threat database to disk."""
        if not self.cache_file:
            return
        
        try:
            data = {
                "last_update": self._last_update.isoformat() if self._last_update else None,
                "threats": [self._serialize_threat(e) for e in self._threats_by_ip.values()],
            }
            
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self._threats_by_ip)} threats to cache")
            
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _parse_threat_entry(self, data: Dict) -> Optional[ThreatEntry]:
        """Parses a threat entry from serialized data."""
        try:
            return ThreatEntry(
                indicator=data["indicator"],
                category=ThreatCategory(data["category"]),
                level=ThreatLevel(data["level"]),
                source=data["source"],
                first_seen=datetime.fromisoformat(data["first_seen"]),
                last_seen=datetime.fromisoformat(data["last_seen"]),
                confidence=data["confidence"],
                tags=data.get("tags", []),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to parse threat entry: {e}")
            return None

    def _serialize_threat(self, entry: ThreatEntry) -> Dict:
        """Serializes a threat entry for storage."""
        return {
            "indicator": entry.indicator,
            "category": entry.category.value,
            "level": entry.level.value,
            "source": entry.source,
            "first_seen": entry.first_seen.isoformat(),
            "last_seen": entry.last_seen.isoformat(),
            "confidence": entry.confidence,
            "tags": entry.tags,
            "metadata": entry.metadata,
        }

    def _index_threat(self, entry: ThreatEntry) -> None:
        """Indexes a threat entry for fast lookup."""
        # Index by IP
        self._threats_by_ip[entry.indicator] = entry
        
        # Index by hash (for domains, URLs, etc.)
        indicator_hash = hashlib.sha256(entry.indicator.encode()).hexdigest()[:16]
        self._threats_by_hash[indicator_hash] = entry

    def check_ip(self, ip_address_str: str) -> Tuple[bool, Optional[ThreatEntry]]:
        """
        Checks if an IP address is known to be malicious.
        
        Returns:
            Tuple of (is_threat, threat_entry)
        """
        # Skip private/trusted IPs
        if self._is_trusted_ip(ip_address_str):
            return (False, None)
        
        # Check direct match
        if ip_address_str in self._threats_by_ip:
            entry = self._threats_by_ip[ip_address_str]
            return (True, entry)
        
        # Check if IP is in any CIDR range from threats
        try:
            ip_obj = ipaddress.ip_address(ip_address_str)
            for threat_ip, entry in self._threats_by_ip.items():
                if "/" in threat_ip:  # CIDR notation
                    network = ipaddress.ip_network(threat_ip, strict=False)
                    if ip_obj in network:
                        return (True, entry)
        except ValueError:
            pass
        
        return (False, None)

    def check_mev_searcher(self, address: str) -> bool:
        """Checks if an Ethereum address is a known MEV searcher."""
        return address.lower() in {a.lower() for a in self._known_mev_searchers}

    def _is_trusted_ip(self, ip_str: str) -> bool:
        """Checks if an IP is in a trusted/private range."""
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            return any(ip_obj in network for network in self._trusted_ranges)
        except ValueError:
            return False

    def add_manual_threat(
        self,
        indicator: str,
        category: ThreatCategory,
        level: ThreatLevel,
        confidence: float = 0.9,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Manually adds a threat indicator."""
        now = datetime.now(timezone.utc)
        
        entry = ThreatEntry(
            indicator=indicator,
            category=category,
            level=level,
            source="manual",
            first_seen=now,
            last_seen=now,
            confidence=confidence,
            tags=tags or [],
        )
        
        self._index_threat(entry)
        logger.info(f"Added manual threat: {indicator} ({category.value})")

    def remove_threat(self, indicator: str) -> bool:
        """Removes a threat indicator from the database."""
        if indicator in self._threats_by_ip:
            del self._threats_by_ip[indicator]
            logger.info(f"Removed threat: {indicator}")
            return True
        return False

    def get_stats(self) -> Dict:
        """Returns statistics about the threat database."""
        categories = {}
        for entry in self._threats_by_ip.values():
            cat = entry.category.value
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_threats": len(self._threats_by_ip),
            "by_category": categories,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "mev_searchers_tracked": len(self._known_mev_searchers),
        }


class ThreatProtectionMiddleware:
    """
    Middleware that integrates threat intelligence with request handling.
    Automatically blocks requests from known malicious sources.
    """

    def __init__(self, threat_feed: ThreatIntelligenceFeed):
        self.threat_feed = threat_feed
        self._blocked_count = 0

    async def check_request(
        self,
        client_ip: str,
        source_address: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Checks if a request should be allowed.
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check IP reputation
        is_threat, threat_entry = self.threat_feed.check_ip(client_ip)
        
        if is_threat and threat_entry:
            self._blocked_count += 1
            reason = (
                f"Blocked {client_ip}: {threat_entry.category.value} "
                f"(confidence: {threat_entry.confidence:.2f}, "
                f"source: {threat_entry.source})"
            )
            logger.warning(reason)
            return (False, reason)
        
        # Check MEV searcher if source address provided
        if source_address and self.threat_feed.check_mev_searcher(source_address):
            self._blocked_count += 1
            reason = f"Blocked MEV searcher: {source_address}"
            logger.warning(reason)
            return (False, reason)
        
        return (True, "Allowed")

    @property
    def blocked_count(self) -> int:
        """Returns the number of blocked requests."""
        return self._blocked_count


async def main():
    """Example usage of the threat intelligence system."""
    print("=" * 60)
    print("Threat Intelligence Feed Aggregator")
    print("=" * 60)
    
    feed = ThreatIntelligenceFeed(
        update_interval_minutes=60,
        cache_file="./threat_cache.json",
    )
    
    await feed.start()
    
    # Add some manual test threats
    feed.add_manual_threat(
        indicator="192.0.2.1",
        category=ThreatCategory.BOTNET,
        level=ThreatLevel.HIGH,
        confidence=0.95,
        tags=["test", "documentation"],
    )
    
    # Test IP checks
    test_ips = ["192.0.2.1", "8.8.8.8", "192.168.1.1"]
    
    print("\nIP Reputation Checks:")
    for ip in test_ips:
        is_threat, entry = feed.check_ip(ip)
        status = "THREAT" if is_threat else "CLEAN"
        print(f"  {ip}: {status}")
        if entry:
            print(f"    Category: {entry.category.value}, Level: {entry.level.name}")
    
    # Create middleware and test
    middleware = ThreatProtectionMiddleware(feed)
    
    print("\nMiddleware Tests:")
    for ip in test_ips:
        allowed, reason = await middleware.check_request(ip)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {ip}: {status} - {reason}")
    
    # Print stats
    print("\nThreat Database Stats:")
    stats = feed.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    await feed.stop()


if __name__ == "__main__":
    asyncio.run(main())
