#!/usr/bin/env python3
"""
ENVIRONMENT VALIDATOR - PRE-FLIGHT API KEY VERIFICATION
Performs lightweight REST ping to Binance to verify API key permissions
before the heavy Rust core boots. Read-only operations only.
"""

import os
import sys
import time
import hashlib
import hmac
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Optional, Dict, Any


class BinanceValidator:
    """Validates Binance API credentials with read-only endpoints."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.timeout = 5  # seconds
        
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for request."""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Make authenticated or public request to Binance."""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        url = f"{self.base_url}{endpoint}"
        if params:
            url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        
        req = Request(url)
        req.add_header('X-MBX-APIKEY', self.api_key)
        req.add_header('User-Agent', 'HFT-Bot/1.0 (Windows Native)')
        
        try:
            with urlopen(req, timeout=self.timeout) as response:
                import json
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            raise Exception(f"HTTP {e.code}: {error_body}")
        except URLError as e:
            raise Exception(f"Connection failed: {e.reason}")
    
    def validate_connectivity(self) -> bool:
        """Test basic connectivity to Binance."""
        try:
            result = self._make_request("/api/v3/ping")
            return result.get("msg") == "pong" or "pong" in str(result).lower()
        except Exception as e:
            print(f"[WARN] Connectivity check failed: {e}")
            return False
    
    def validate_api_key_permissions(self) -> Dict[str, bool]:
        """Check API key permissions (Read-Only vs Trade)."""
        permissions = {
            "read_only": False,
            "spot_trading": False,
            "futures_trading": False,
            "valid": False
        }
        
        try:
            # Get account info to check permissions
            account_info = self._make_request("/api/v3/account", signed=True)
            
            # Check if we can read account data
            if "accountType" in account_info or "balances" in account_info:
                permissions["read_only"] = True
                permissions["valid"] = True
            
            # Check for trading permissions
            if account_info.get("canTrade", False):
                permissions["spot_trading"] = True
            
            print(f"[INFO] Account Type: {account_info.get('accountType', 'UNKNOWN')}")
            print(f"[INFO] Can Trade: {account_info.get('canTrade', False)}")
            
        except Exception as e:
            error_msg = str(e)
            if "API-key format invalid" in error_msg:
                print(f"[ERROR] Invalid API Key format")
            elif "Signature for this request is not valid" in error_msg:
                print(f"[ERROR] Invalid API Secret or clock skew")
            elif "IP address not authorized" in error_msg:
                print(f"[ERROR] IP address not whitelisted")
            else:
                print(f"[WARN] Permission check failed: {e}")
        
        return permissions
    
    def get_server_time(self) -> Optional[int]:
        """Get server time for clock skew validation."""
        try:
            result = self._make_request("/api/v3/time")
            return result.get("serverTime")
        except Exception:
            return None
    
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        print("=" * 60)
        print("BINANCE API KEY PRE-FLIGHT VALIDATION")
        print("=" * 60)
        
        # Step 1: Check environment variables
        if not self.api_key or not self.api_secret:
            print("[ERROR] BINANCE_API_KEY or BINANCE_API_SECRET not set")
            return False
        
        if len(self.api_key) < 10 or len(self.api_secret) < 10:
            print("[ERROR] API credentials appear truncated or invalid")
            return False
        
        print("[INFO] Environment variables present ✓")
        
        # Step 2: Check connectivity
        if not self.validate_connectivity():
            print("[ERROR] Cannot connect to Binance API")
            return False
        
        print("[INFO] Binance connectivity OK ✓")
        
        # Step 3: Check clock skew
        server_time = self.get_server_time()
        if server_time:
            local_time = int(time.time() * 1000)
            skew_ms = abs(server_time - local_time)
            if skew_ms > 5000:  # 5 second threshold
                print(f"[WARN] Clock skew detected: {skew_ms}ms (>5000ms)")
                print("[WARN] Please sync your system clock")
            else:
                print(f"[INFO] Clock skew: {skew_ms}ms ✓")
        
        # Step 4: Validate permissions
        perms = self.validate_api_key_permissions()
        
        if not perms["valid"]:
            print("[ERROR] API key validation failed")
            return False
        
        print(f"[INFO] Read-Only Access: {'✓' if perms['read_only'] else '✗'}")
        print(f"[INFO] Spot Trading Enabled: {'✓' if perms['spot_trading'] else '✗'}")
        
        if perms["spot_trading"]:
            print("[SUCCESS] API key validated with trading permissions")
        else:
            print("[WARN] API key has read-only access (no trading)")
        
        print("=" * 60)
        return True


def main():
    """Main entry point."""
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")
    testnet = os.environ.get("BINANCE_TESTNET", "true").lower() == "true"
    
    validator = BinanceValidator(api_key, api_secret, testnet)
    success = validator.run_validation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
