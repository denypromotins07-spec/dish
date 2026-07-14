"""
Python Wrapper for Rust HMAC Signer via PyO3.

Provides a seamless, low-latency interface for the Python execution layer
to sign authenticated Binance REST API requests using the optimized
Rust HMAC-SHA256 implementation.
"""

import logging
from typing import Dict, List, Optional

# Import the Rust module (would be built with maturin or setuptools-rust)
# from rust_auth import BinanceSigner as RustBinanceSigner
# For now, use pure Python fallback that mirrors the Rust implementation

import hmac
import hashlib
import time
import hex

log = logging.getLogger(__name__)


class BinanceSigner:
    """
    Python wrapper for Rust HMAC signer.
    
    Falls back to pure Python implementation if Rust module not available,
    but provides identical API for seamless integration.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        recv_window_ms: int = 5000,
        use_rust: bool = True,
    ):
        self.api_key = api_key
        self.secret_key = secret_key.encode('utf-8')
        self.recv_window_ms = recv_window_ms
        self.use_rust = use_rust
        
        # Try to load Rust implementation
        self._rust_signer = None
        if use_rust:
            try:
                from rust_auth import BinanceSigner as RustBinanceSigner
                self._rust_signer = RustBinanceSigner(api_key, secret_key, recv_window_ms)
                log.info("Using Rust HMAC signer (hardware-accelerated)")
            except ImportError:
                log.warning("Rust signer not available, falling back to Python")
                self.use_rust = False
                
    def _current_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
        
    def sign_query(self, query: str) -> str:
        """
        Sign a query string for GET request.
        
        Args:
            query: URL query parameters (e.g., "symbol=BTCUSDT&side=BUY")
            
        Returns:
            Hex-encoded HMAC-SHA256 signature
        """
        if self._rust_signer:
            return self._rust_signer.sign_query(query)
            
        # Pure Python fallback
        timestamp = self._current_timestamp_ms()
        query_with_ts = f"{query}&timestamp={timestamp}"
        
        signature = hmac.new(
            self.secret_key,
            query_with_ts.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def sign_body(self, body: str) -> str:
        """
        Sign a request body for POST/DELETE request.
        
        Args:
            body: Request body parameters
            
        Returns:
            Hex-encoded HMAC-SHA256 signature
        """
        if self._rust_signer:
            return self._rust_signer.sign_body(body)
            
        # Pure Python fallback
        timestamp = self._current_timestamp_ms()
        body_with_ts = f"{body}&timestamp={timestamp}"
        
        signature = hmac.new(
            self.secret_key,
            body_with_ts.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def build_signed_url(
        self,
        base_url: str,
        endpoint: str,
        params: str,
    ) -> str:
        """
        Build complete signed URL for GET request.
        
        Args:
            base_url: Base API URL (e.g., "https://fapi.binance.com")
            endpoint: API endpoint (e.g., "/fapi/v1/order")
            params: Query parameters without leading "?"
            
        Returns:
            Complete signed URL
        """
        if self._rust_signer:
            return self._rust_signer.build_signed_url(base_url, endpoint, params)
            
        # Pure Python fallback
        signature = self.sign_query(params)
        timestamp = self._current_timestamp_ms()
        
        return (
            f"{base_url}{endpoint}?"
            f"{params}"
            f"&timestamp={timestamp}"
            f"&signature={signature}"
            f"&recvWindow={self.recv_window_ms}"
        )
        
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        
        Returns:
            Dictionary of headers including X-MBX-APIKEY
        """
        if self._rust_signer:
            # Convert Rust HashMap to Python dict
            rust_headers = self._rust_signer.auth_headers()
            return dict(rust_headers)
            
        return {"X-MBX-APIKEY": self.api_key}
        
    def validate_timestamp(self, server_time_ms: int) -> bool:
        """
        Validate that server timestamp is within receive window.
        
        Args:
            server_time_ms: Server timestamp in milliseconds
            
        Returns:
            True if timestamp is valid, False otherwise
        """
        if self._rust_signer:
            return self._rust_signer.validate_timestamp(server_time_ms)
            
        # Pure Python fallback
        client_time = self._current_timestamp_ms()
        diff = abs(server_time_ms - client_time)
        return diff <= self.recv_window_ms
        
    def sign_batch(self, queries: List[str]) -> List[str]:
        """
        Sign multiple queries efficiently (batch operation).
        
        Uses single timestamp for all queries to improve throughput.
        
        Args:
            queries: List of query strings to sign
            
        Returns:
            List of signatures
        """
        if self._rust_signer:
            return list(self._rust_signer.sign_batch(queries))
            
        # Pure Python fallback - batch optimization
        timestamp = self._current_timestamp_ms()
        signatures = []
        
        for query in queries:
            query_with_ts = f"{query}&timestamp={timestamp}"
            signature = hmac.new(
                self.secret_key,
                query_with_ts.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            signatures.append(signature)
            
        return signatures


class AsyncBinanceSigner(BinanceSigner):
    """
    Async-compatible signer for use with aiohttp.
    
    Provides the same API but designed for async contexts.
    """
    
    async def sign_query_async(self, query: str) -> str:
        """Async version of sign_query (non-blocking)."""
        # Signing is CPU-bound but fast, so we can do it directly
        return self.sign_query(query)
        
    async def sign_body_async(self, body: str) -> str:
        """Async version of sign_body."""
        return self.sign_body(body)
        
    async def build_signed_url_async(
        self,
        base_url: str,
        endpoint: str,
        params: str,
    ) -> str:
        """Async version of build_signed_url."""
        return self.build_signed_url(base_url, endpoint, params)


def create_signer(
    api_key: str,
    secret_key: str,
    recv_window_ms: int = 5000,
    prefer_rust: bool = True,
) -> BinanceSigner:
    """
    Factory function to create a Binance signer.
    
    Args:
        api_key: Binance API key
        secret_key: Binance API secret
        recv_window_ms: Receive window in milliseconds
        prefer_rust: Whether to try Rust implementation first
        
    Returns:
        Configured BinanceSigner instance
    """
    return BinanceSigner(
        api_key=api_key,
        secret_key=secret_key,
        recv_window_ms=recv_window_ms,
        use_rust=prefer_rust,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    signer = create_signer(
        api_key="test_api_key",
        secret_key="test_secret_key",
        prefer_rust=True,
    )
    
    # Sign a query
    query = "symbol=BTCUSDT&side=BUY&type=LIMIT"
    signature = signer.sign_query(query)
    print(f"Signature: {signature}")
    
    # Build signed URL
    url = signer.build_signed_url(
        "https://testnet.binancefuture.com",
        "/fapi/v1/order",
        "symbol=BTCUSDT&side=BUY&type=LIMIT&price=50000&quantity=0.001",
    )
    print(f"Signed URL: {url}")
    
    # Get auth headers
    headers = signer.get_auth_headers()
    print(f"Headers: {headers}")
    
    # Batch signing
    queries = [
        "symbol=BTCUSDT",
        "symbol=ETHUSDT",
        "symbol=SOLUSDT",
    ]
    signatures = signer.sign_batch(queries)
    print(f"Batch signatures: {len(signatures)} generated")
