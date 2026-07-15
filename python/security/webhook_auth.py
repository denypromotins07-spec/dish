#!/usr/bin/env python3
"""
HMAC-SHA256 signature verification middleware for incoming external trading signals.
Rejects any payload that doesn't possess a valid cryptographic signature,
preventing spoofed signals from triggering trades.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebhookAuthStatus(Enum):
    """Authentication result status."""
    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    EXPIRED_TIMESTAMP = "expired_timestamp"
    REPLAY_DETECTED = "replay_detected"
    MISSING_HEADERS = "missing_headers"
    INVALID_PAYLOAD = "invalid_payload"


@dataclass
class AuthResult:
    """Result of webhook authentication."""
    status: WebhookAuthStatus
    message: str
    is_valid: bool
    signal_data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class WebhookAuthenticator:
    """
    HMAC-SHA256 signature verifier for TradingView and other webhook signals.
    Prevents spoofed signals by requiring cryptographic authentication.
    """

    def __init__(
        self,
        secret_key: bytes,
        max_age_seconds: int = 60,
        enable_replay_protection: bool = True,
        replay_cache_size: int = 10000,
    ):
        self.secret_key = secret_key
        self.max_age_seconds = max_age_seconds
        self.enable_replay_protection = enable_replay_protection
        
        # Replay protection cache (signature -> timestamp)
        self._seen_signatures: Dict[str, float] = {}
        self._replay_cache_size = replay_cache_size

    def verify_webhook(
        self,
        payload: str,
        signature: str,
        timestamp: Optional[int] = None,
    ) -> AuthResult:
        """
        Verifies an incoming webhook request.
        
        Args:
            payload: Raw request body as string
            signature: HMAC-SHA256 signature from headers
            timestamp: Optional Unix timestamp for time-based validation
        
        Returns:
            AuthResult with validation status and parsed signal data
        """
        # Check required components
        if not payload or not signature:
            return AuthResult(
                status=WebhookAuthStatus.MISSING_HEADERS,
                message="Missing payload or signature",
                is_valid=False,
            )
        
        # Verify timestamp freshness
        if timestamp is not None:
            current_time = int(time.time())
            age = abs(current_time - timestamp)
            
            if age > self.max_age_seconds:
                return AuthResult(
                    status=WebhookAuthStatus.EXPIRED_TIMESTAMP,
                    message=f"Request timestamp expired (age: {age}s, max: {self.max_age_seconds}s)",
                    is_valid=False,
                )
        
        # Verify signature
        expected_signature = self._compute_signature(payload, timestamp)
        
        if not self._constant_time_compare(signature, expected_signature):
            logger.warning(f"Invalid signature received. Expected: {expected_signature}, Got: {signature}")
            return AuthResult(
                status=WebhookAuthStatus.INVALID_SIGNATURE,
                message="Signature verification failed",
                is_valid=False,
            )
        
        # Check for replay attacks
        if self.enable_replay_protection:
            if signature in self._seen_signatures:
                return AuthResult(
                    status=WebhookAuthStatus.REPLAY_DETECTED,
                    message="Replay attack detected - signature already used",
                    is_valid=False,
                )
            
            # Add to seen signatures cache
            self._seen_signatures[signature] = time.time()
            
            # Prune old entries if cache is too large
            if len(self._seen_signatures) > self._replay_cache_size:
                cutoff = time.time() - self.max_age_seconds * 2
                self._seen_signatures = {
                    k: v for k, v in self._seen_signatures.items()
                    if v > cutoff
                }
        
        # Parse payload
        try:
            signal_data = json.loads(payload)
        except json.JSONDecodeError as e:
            return AuthResult(
                status=WebhookAuthStatus.INVALID_PAYLOAD,
                message=f"Invalid JSON payload: {e}",
                is_valid=False,
            )
        
        # Validate signal structure
        if not self._validate_signal_structure(signal_data):
            return AuthResult(
                status=WebhookAuthStatus.INVALID_PAYLOAD,
                message="Signal payload missing required fields",
                is_valid=False,
            )
        
        return AuthResult(
            status=WebhookAuthStatus.VALID,
            message="Webhook authenticated successfully",
            is_valid=True,
            signal_data=signal_data,
            timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp else None,
        )

    def _compute_signature(self, payload: str, timestamp: Optional[int]) -> str:
        """
        Computes HMAC-SHA256 signature for the payload.
        
        Signature format: HMAC(secret, payload + timestamp)
        """
        message = payload.encode('utf-8')
        
        if timestamp is not None:
            message += str(timestamp).encode('utf-8')
        
        signature = hmac.new(
            self.secret_key,
            message,
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def _constant_time_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison to prevent timing attacks."""
        return hmac.compare_digest(a.encode(), b.encode())

    def _validate_signal_structure(self, data: Dict[str, Any]) -> bool:
        """
        Validates that the signal has the required structure.
        Expected format varies by source; this is a common TradingView format.
        """
        required_fields = ["action"]  # Minimum required field
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Validate action value
        action = data.get("action", "").upper()
        if action not in ["BUY", "SELL", "LONG", "SHORT", "CLOSE", "EXIT"]:
            logger.warning(f"Invalid action value: {action}")
            return False
        
        return True

    def generate_signature(self, payload: Dict[str, Any], timestamp: Optional[int] = None) -> str:
        """
        Generates a signature for an outgoing webhook payload.
        Useful for testing or generating signed signals.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        return self._compute_signature(payload_str, timestamp)


class TradingViewWebhookHandler(WebhookAuthenticator):
    """
    Specialized handler for TradingView webhook alerts.
    Parses TradingView-specific signal formats.
    """

    def __init__(self, secret_key: bytes, **kwargs):
        super().__init__(secret_key, **kwargs)
        
        # TradingView-specific field mappings
        self.field_mappings = {
            "ticker": "symbol",
            "price": "entry_price",
            "order_type": "type",
        }

    def parse_tradingview_signal(self, auth_result: AuthResult) -> Optional[Dict[str, Any]]:
        """
        Parses a validated TradingView webhook into a standardized signal format.
        """
        if not auth_result.is_valid or not auth_result.signal_data:
            return None
        
        raw_data = auth_result.signal_data
        standardized = {}
        
        # Map TradingView fields to standard format
        for tv_field, std_field in self.field_mappings.items():
            if tv_field in raw_data:
                standardized[std_field] = raw_data[tv_field]
        
        # Handle action mapping
        action = raw_data.get("action", "").upper()
        if action in ["LONG", "BUY"]:
            standardized["side"] = "BUY"
        elif action in ["SHORT", "SELL"]:
            standardized["side"] = "SELL"
        elif action in ["CLOSE", "EXIT"]:
            standardized["side"] = "CLOSE"
        else:
            standardized["side"] = action
        
        # Extract optional fields
        optional_fields = ["stop_loss", "take_profit", "quantity", "leverage"]
        for field in optional_fields:
            if field in raw_data:
                standardized[field] = raw_data[field]
        
        # Add metadata
        standardized["source"] = "tradingview"
        standardized["timestamp"] = auth_result.timestamp
        standardized["original_data"] = raw_data
        
        return standardized


def create_webhook_middleware(secret_key: str) -> TradingViewWebhookHandler:
    """
    Factory function to create webhook middleware.
    
    Usage with FastAPI/Flask:
        middleware = create_webhook_middleware("your-secret-key")
        
        @app.post("/webhook")
        async def handle_webhook(request: Request):
            payload = await request.body()
            signature = request.headers.get("X-Webhook-Signature")
            timestamp = request.headers.get("X-Webhook-Timestamp")
            
            result = middleware.verify_webhook(payload.decode(), signature, int(timestamp))
            
            if not result.is_valid:
                raise HTTPException(status_code=401, detail=result.message)
            
            signal = middleware.parse_tradingview_signal(result)
            return {"status": "accepted", "signal": signal}
    """
    return TradingViewWebhookHandler(secret_key.encode())


async def main():
    """Example usage of the webhook authenticator."""
    print("=" * 60)
    print("Webhook Authentication Middleware")
    print("=" * 60)
    
    # Setup
    secret = b"super_secret_webhook_key_12345"
    handler = TradingViewWebhookHandler(secret)
    
    # Create a test signal
    test_signal = {
        "action": "LONG",
        "ticker": "BTCUSDT",
        "price": 45000.0,
        "stop_loss": 44000.0,
        "take_profit": 47000.0,
        "quantity": 0.1,
    }
    
    # Generate signature
    timestamp = int(time.time())
    payload = json.dumps(test_signal, sort_keys=True)
    signature = handler.generate_signature(test_signal, timestamp)
    
    print(f"\nTest Signal: {test_signal}")
    print(f"Timestamp: {timestamp}")
    print(f"Signature: {signature}")
    
    # Verify the signal
    result = handler.verify_webhook(payload, signature, timestamp)
    
    print(f"\nVerification Result:")
    print(f"  Status: {result.status.value}")
    print(f"  Valid: {result.is_valid}")
    print(f"  Message: {result.message}")
    
    if result.is_valid:
        standardized = handler.parse_tradingview_signal(result)
        print(f"\nStandardized Signal: {standardized}")
    
    # Test invalid signature
    print("\n" + "=" * 60)
    print("Testing invalid signature...")
    invalid_result = handler.verify_webhook(payload, "invalid_signature", timestamp)
    print(f"  Status: {invalid_result.status.value}")
    print(f"  Valid: {invalid_result.is_valid}")
    
    # Test replay attack
    print("\n" + "=" * 60)
    print("Testing replay attack detection...")
    replay_result = handler.verify_webhook(payload, signature, timestamp)
    print(f"  Status: {replay_result.status.value}")
    print(f"  Valid: {replay_result.is_valid}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
