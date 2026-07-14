"""
Redis Cache for Active Order States

Redis configuration script optimized for a low-memory footprint using:
- ziplist/listpack encodings for compact storage
- allkeys-lru eviction policy
- Strict memory limits to prevent RAM bloat

Designed for caching active order states with minimal memory usage.
"""

import os
import json
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[REDIS] Redis client not available, using mock implementation")


# Constants
DEFAULT_MAX_MEMORY_MB = 512
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6379
DEFAULT_DB = 0
KEY_PREFIX = "trading:"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class RedisConfig:
    """Redis configuration for low-memory operation."""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    db: int = DEFAULT_DB
    max_memory_mb: int = DEFAULT_MAX_MEMORY_MB
    password: Optional[str] = None
    socket_timeout_sec: float = 1.0
    socket_connect_timeout_sec: float = 1.0
    
    # Listpack encoding thresholds (for compact storage)
    hash_max_listpack_entries: int = 512
    hash_max_listpack_value: int = 64
    list_max_listpack_entries: int = 256
    list_max_listpack_value: int = 64
    
    # Eviction policy
    maxmemory_policy: str = "allkeys-lru"


class OrderCacheEntry:
    """Compact representation of an order for caching."""
    
    __slots__ = [
        'order_id', 'symbol', 'side', 'order_type', 
        'price', 'quantity', 'filled_quantity', 'status',
        'timestamp_ns', 'exchange'
    ]
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        price: Optional[float],
        quantity: float,
        filled_quantity: float = 0.0,
        status: OrderStatus = OrderStatus.PENDING,
        timestamp_ns: Optional[int] = None,
        exchange: str = "unknown",
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.price = price
        self.quantity = quantity
        self.filled_quantity = filled_quantity
        self.status = status
        self.timestamp_ns = timestamp_ns or time.time_ns()
        self.exchange = exchange
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'order_type': self.order_type,
            'price': self.price,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'status': self.status.value,
            'timestamp_ns': self.timestamp_ns,
            'exchange': self.exchange,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderCacheEntry':
        return cls(
            order_id=data['order_id'],
            symbol=data['symbol'],
            side=data['side'],
            order_type=data['order_type'],
            price=data.get('price'),
            quantity=data['quantity'],
            filled_quantity=data.get('filled_quantity', 0.0),
            status=OrderStatus(data.get('status', 'pending')),
            timestamp_ns=data.get('timestamp_ns'),
            exchange=data.get('exchange', 'unknown'),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to compact JSON bytes."""
        return json.dumps(self.to_dict(), separators=(',', ':')).encode()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'OrderCacheEntry':
        """Deserialize from JSON bytes."""
        return cls.from_dict(json.loads(data.decode()))


class LowMemoryRedisClient:
    """
    Redis client wrapper optimized for low-memory operation.
    
    Features:
    - Automatic memory limit enforcement
    - Compact serialization
    - LRU-friendly access patterns
    """
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[Any] = None
        self._lock = threading.Lock()
        
        if not REDIS_AVAILABLE:
            return
        
        self._connect()
        self._configure_memory_limits()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout_sec,
                socket_connect_timeout=self.config.socket_connect_timeout_sec,
                decode_responses=False,  # We handle encoding ourselves
            )
            # Test connection
            self._client.ping()
        except Exception as e:
            print(f"[REDIS] Connection error: {e}")
            self._client = None
    
    def _configure_memory_limits(self):
        """Configure Redis memory limits and eviction policy."""
        if self._client is None:
            return
        
        try:
            # Set max memory
            max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
            self._client.config_set('maxmemory', str(max_memory_bytes))
            
            # Set eviction policy
            self._client.config_set('maxmemory-policy', self.config.maxmemory_policy)
            
            # Configure listpack thresholds for compact storage
            self._client.config_set(
                'hash-max-listpack-entries',
                str(self.config.hash_max_listpack_entries)
            )
            self._client.config_set(
                'hash-max-listpack-value',
                str(self.config.hash_max_listpack_value)
            )
            self._client.config_set(
                'list-max-listpack-entries',
                str(self.config.list_max_listpack_entries)
            )
            self._client.config_set(
                'list-max-listpack-value',
                str(self.config.list_max_listpack_value)
            )
            
            print(f"[REDIS] Memory configured: {self.config.max_memory_mb}MB limit, "
                  f"{self.config.maxmemory_policy} eviction")
            
        except Exception as e:
            print(f"[REDIS] Configuration error: {e}")
    
    def _key(self, key: str) -> str:
        """Generate prefixed key."""
        return f"{KEY_PREFIX}{key}"
    
    def set_order(self, order: OrderCacheEntry, ttl_sec: Optional[int] = None) -> bool:
        """Store an order in the cache."""
        if self._client is None:
            return False
        
        with self._lock:
            try:
                key = self._key(f"order:{order.order_id}")
                self._client.set(key, order.to_bytes(), ex=ttl_sec)
                
                # Also index by symbol for quick lookup
                symbol_key = self._key(f"symbol:{order.symbol}:orders")
                self._client.sadd(symbol_key, order.order_id)
                
                return True
            except Exception as e:
                print(f"[REDIS] Set order error: {e}")
                return False
    
    def get_order(self, order_id: str) -> Optional[OrderCacheEntry]:
        """Retrieve an order from the cache."""
        if self._client is None:
            return None
        
        with self._lock:
            try:
                key = self._key(f"order:{order_id}")
                data = self._client.get(key)
                if data:
                    return OrderCacheEntry.from_bytes(data)
                return None
            except Exception as e:
                print(f"[REDIS] Get order error: {e}")
                return None
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[float] = None,
    ) -> bool:
        """Update order status atomically."""
        order = self.get_order(order_id)
        if order is None:
            return False
        
        order.status = status
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
        
        return self.set_order(order)
    
    def delete_order(self, order_id: str) -> bool:
        """Delete an order from the cache."""
        if self._client is None:
            return False
        
        with self._lock:
            try:
                order = self.get_order(order_id)
                if order:
                    key = self._key(f"order:{order_id}")
                    symbol_key = self._key(f"symbol:{order.symbol}:orders")
                    
                    self._client.delete(key)
                    self._client.srem(symbol_key, order_id)
                
                return True
            except Exception as e:
                print(f"[REDIS] Delete order error: {e}")
                return False
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderCacheEntry]:
        """Get all orders for a symbol."""
        if self._client is None:
            return []
        
        with self._lock:
            try:
                symbol_key = self._key(f"symbol:{symbol}:orders")
                order_ids = self._client.smembers(symbol_key)
                
                orders = []
                for order_id_bytes in order_ids:
                    order_id = order_id_bytes.decode() if isinstance(order_id_bytes, bytes) else order_id_bytes
                    order = self.get_order(order_id)
                    if order:
                        orders.append(order)
                
                return orders
            except Exception as e:
                print(f"[REDIS] Get orders by symbol error: {e}")
                return []
    
    def get_active_orders(self) -> List[OrderCacheEntry]:
        """Get all active (non-terminal) orders."""
        active_statuses = {
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED,
        }
        
        # Scan all order keys
        orders = []
        if self._client:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._client.scan(
                        cursor=cursor,
                        match=self._key("order:*"),
                        count=100
                    )
                    
                    for key in keys:
                        data = self._client.get(key)
                        if data:
                            order = OrderCacheEntry.from_bytes(data)
                            if order.status in active_statuses:
                                orders.append(order)
                    
                    if cursor == 0:
                        break
                        
            except Exception as e:
                print(f"[REDIS] Get active orders error: {e}")
        
        return orders
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics."""
        if self._client is None:
            return {}
        
        try:
            info = self._client.info('memory')
            return {
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'used_memory_peak_mb': info.get('used_memory_peak', 0) / (1024 * 1024),
                'max_memory_mb': info.get('maxmemory', 0) / (1024 * 1024),
                'mem_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0),
                'evicted_keys': info.get('evicted_keys', 0),
            }
        except Exception as e:
            print(f"[REDIS] Get memory stats error: {e}")
            return {}
    
    def flush_all(self):
        """Flush all trading keys (use with caution)."""
        if self._client is None:
            return
        
        with self._lock:
            try:
                # Delete only our prefixed keys
                cursor = 0
                while True:
                    cursor, keys = self._client.scan(
                        cursor=cursor,
                        match=self._key("*"),
                        count=100
                    )
                    if keys:
                        self._client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                print(f"[REDIS] Flush error: {e}")


class RedisCacheManager:
    """
    High-level manager for Redis cache operations.
    Provides singleton access and automatic initialization.
    """
    
    _instance: Optional['RedisCacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'RedisCacheManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, config: Optional[RedisConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.config = config or RedisConfig()
        self._client: Optional[LowMemoryRedisClient] = None
    
    def initialize(self):
        """Initialize the Redis client."""
        self._client = LowMemoryRedisClient(self.config)
        print("[REDIS] Cache manager initialized")
    
    @property
    def client(self) -> Optional[LowMemoryRedisClient]:
        return self._client
    
    def cache_order(self, order: OrderCacheEntry) -> bool:
        """Cache an order."""
        if self._client:
            return self._client.set_order(order)
        return False
    
    def get_cached_order(self, order_id: str) -> Optional[OrderCacheEntry]:
        """Get a cached order."""
        if self._client:
            return self._client.get_order(order_id)
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._client:
            return self._client.get_memory_stats()
        return {}


# Convenience functions
_cache_manager_instance: Optional[RedisCacheManager] = None


def get_cache_manager() -> RedisCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = RedisCacheManager()
    return _cache_manager_instance


def init_redis_cache(max_memory_mb: int = 512) -> RedisCacheManager:
    """Initialize Redis cache with custom settings."""
    global _cache_manager_instance
    config = RedisConfig(max_memory_mb=max_memory_mb)
    _cache_manager_instance = RedisCacheManager(config)
    _cache_manager_instance.initialize()
    return _cache_manager_instance


if __name__ == "__main__":
    # Demo/test code
    print("[DEMO] Redis Cache Manager Demo")
    
    if not REDIS_AVAILABLE:
        print("[DEMO] Redis client not installed, skipping demo")
    else:
        manager = init_redis_cache(max_memory_mb=256)
        
        # Create test order
        order = OrderCacheEntry(
            order_id="test-order-001",
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            price=50000.0,
            quantity=0.1,
            status=OrderStatus.SUBMITTED,
        )
        
        # Cache the order
        success = manager.cache_order(order)
        print(f"[DEMO] Order cached: {success}")
        
        # Retrieve the order
        retrieved = manager.get_cached_order("test-order-001")
        if retrieved:
            print(f"[DEMO] Retrieved order: {retrieved.order_id}, status: {retrieved.status.value}")
        
        # Stats
        print("[DEMO] Stats:", manager.get_stats())
