#!/usr/bin/env python3
"""
Pre-flight security script that verifies the bot's current public IP against 
the exchange's whitelisted IPs via REST API before enabling live execution.
Automatically halts trading if the network environment changes.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set, List
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityStatus(Enum):
    """Security status of the IP verification."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CHANGED = "changed"
    ERROR = "error"


@dataclass
class IPVerificationResult:
    """Result of an IP verification check."""
    status: SecurityStatus
    current_ip: Optional[str] = None
    whitelisted_ips: Set[str] = field(default_factory=set)
    message: str = ""


class IPWhitelister:
    """
    Pre-flight IP whitelist verifier for exchange security.
    Ensures the bot only runs from authorized network locations.
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: str,
        api_secret: str,
        whitelisted_ips: Optional[Set[str]] = None,
        check_interval_seconds: int = 60,
        max_failures_before_halt: int = 3,
    ):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.whitelisted_ips = whitelisted_ips or set()
        self.check_interval_seconds = check_interval_seconds
        self.max_failures_before_halt = max_failures_before_halt
        
        self._current_ip: Optional[str] = None
        self._failure_count: int = 0
        self._is_running: bool = False
        self._trading_enabled: bool = False

    async def get_public_ip(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Fetches the current public IP address using multiple providers for redundancy."""
        ip_providers = [
            "https://api.ipify.org?format=text",
            "https://ifconfig.me/ip",
            "https://icanhazip.com",
        ]
        
        for provider in ip_providers:
            try:
                async with session.get(provider, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        ip = (await resp.text()).strip()
                        # Validate IP format
                        if self._is_valid_ip(ip):
                            return ip
            except Exception as e:
                logger.warning(f"Failed to fetch IP from {provider}: {e}")
                continue
        
        return None

    def _is_valid_ip(self, ip: str) -> bool:
        """Validates IPv4 or IPv6 format."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    async def fetch_whitelisted_ips_from_exchange(
        self, session: aiohttp.ClientSession
    ) -> Set[str]:
        """
        Fetches the list of whitelisted IPs directly from the exchange API.
        Implementation varies by exchange; this is a generic template.
        """
        # Generic implementation - customize per exchange
        endpoint = f"https://api.{self.exchange_name.lower()}.com/api/v3/accountIPs"
        
        import hmac
        import time
        timestamp = int(time.time() * 1000)
        signature = hmac.new(
            self.api_secret.encode(),
            f"timestamp={timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-MBX-APIKEY": self.api_key,
        }
        
        try:
            async with session.get(
                endpoint,
                headers=headers,
                params={"timestamp": timestamp, "signature": signature},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Extract IPs from response (format varies by exchange)
                    ips = set(data.get("ips", []))
                    return ips
                else:
                    logger.error(f"Failed to fetch whitelisted IPs: {resp.status}")
        except Exception as e:
            logger.error(f"Error fetching whitelisted IPs: {e}")
        
        # Fallback to manually configured IPs if API fetch fails
        return self.whitelisted_ips

    async def verify_ip(self) -> IPVerificationResult:
        """Performs a single IP verification check."""
        async with aiohttp.ClientSession() as session:
            current_ip = await self.get_public_ip(session)
            
            if not current_ip:
                self._failure_count += 1
                msg = f"Failed to determine public IP (failures: {self._failure_count})"
                logger.error(msg)
                
                if self._failure_count >= self.max_failures_before_halt:
                    self._trading_enabled = False
                    return IPVerificationResult(
                        status=SecurityStatus.ERROR,
                        message=msg + " - Trading HALTED"
                    )
                
                return IPVerificationResult(
                    status=SecurityStatus.ERROR,
                    message=msg
                )
            
            # Reset failure count on success
            self._failure_count = 0
            self._current_ip = current_ip
            
            # Get whitelisted IPs (use cached if available, otherwise fetch)
            whitelisted = self.whitelisted_ips
            if not whitelisted:
                whitelisted = await self.fetch_whitelisted_ips_from_exchange(session)
            
            if current_ip in whitelisted:
                self._trading_enabled = True
                return IPVerificationResult(
                    status=SecurityStatus.VERIFIED,
                    current_ip=current_ip,
                    whitelisted_ips=whitelisted,
                    message=f"IP {current_ip} is whitelisted"
                )
            else:
                self._trading_enabled = False
                return IPVerificationResult(
                    status=SecurityStatus.UNVERIFIED,
                    current_ip=current_ip,
                    whitelisted_ips=whitelisted,
                    message=f"CRITICAL: IP {current_ip} is NOT whitelisted! Trading HALTED."
                )

    async def start_continuous_monitoring(self) -> None:
        """Starts continuous IP monitoring in a background loop."""
        self._is_running = True
        logger.info("Starting continuous IP whitelist monitoring...")
        
        last_verified_ip: Optional[str] = None
        
        while self._is_running:
            result = await self.verify_ip()
            
            if result.status == SecurityStatus.VERIFIED:
                if last_verified_ip and last_verified_ip != result.current_ip:
                    logger.warning(
                        f"IP CHANGE DETECTED: {last_verified_ip} -> {result.current_ip}"
                    )
                    result.status = SecurityStatus.CHANGED
                last_verified_ip = result.current_ip
                logger.debug(result.message)
            else:
                logger.critical(result.message)
                if result.status in (SecurityStatus.UNVERIFIED, SecurityStatus.ERROR):
                    # Emergency halt
                    await self._emergency_halt()
            
            await asyncio.sleep(self.check_interval_seconds)

    async def _emergency_halt(self) -> None:
        """Executes emergency trading halt procedure."""
        logger.critical("EMERGENCY HALT: Disabling all trading operations!")
        self._trading_enabled = False
        # Additional emergency procedures can be added here:
        # - Cancel all open orders
        # - Close all positions
        # - Send alert notifications

    def is_trading_enabled(self) -> bool:
        """Returns whether trading is currently enabled based on IP verification."""
        return self._trading_enabled

    def stop_monitoring(self) -> None:
        """Stops the continuous monitoring loop."""
        self._is_running = False
        logger.info("IP whitelist monitoring stopped.")


async def main():
    """Example usage of the IPWhitelister."""
    whitelister = IPWhitelister(
        exchange_name="Binance",
        api_key="your_api_key",
        api_secret="your_api_secret",
        whitelisted_ips={"203.0.113.1", "198.51.100.5"},  # Example IPs
        check_interval_seconds=30,
    )
    
    # Initial verification
    result = await whitelister.verify_ip()
    print(f"Initial verification: {result.status.value} - {result.message}")
    
    if whitelister.is_trading_enabled():
        print("Trading is ENABLED")
        # Start continuous monitoring
        # await whitelister.start_continuous_monitoring()
    else:
        print("Trading is DISABLED")


if __name__ == "__main__":
    asyncio.run(main())
