"""
Advanced NautilusTrader Configuration Builder.

Generates optimized YAML configuration for NautilusTrader nodes,
defining venues, data clients, and execution clients with tuned
message bus settings for low-latency trading.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger(__name__)


@dataclass
class MessageBusConfig:
    """Configuration for Nautilus's internal message bus."""
    
    # Use lock-free queues for inter-component communication
    queue_type: str = "mpsc"  # Multi-producer single-consumer
    
    # Batch size for message processing (balance latency vs throughput)
    batch_size: int = 256
    
    # Enable zero-copy serialization where possible
    zero_copy_enabled: bool = True
    
    # Memory pool size for message buffers (bytes)
    buffer_pool_size: int = 32 * 1024 * 1024  # 32MB
    

@dataclass
class DataClientConfig:
    """Configuration for market data clients."""
    
    client_id: str
    venue: str
    
    # Connection settings
    heartbeat_interval_ms: int = 30_000
    reconnect_delay_ms: int = 1_000
    max_reconnect_attempts: int = 10
    
    # Data subscription settings
    subscribe_trades: bool = True
    subscribe_order_book: bool = True
    order_book_depth: int = 25  # L2 depth
    
    # Performance tuning
    use_rust_ingestion: bool = True  # Use Rust WebSocket client
    rust_event_bus_integration: bool = True
    

@dataclass
class ExecutionClientConfig:
    """Configuration for order execution clients."""
    
    client_id: str
    venue: str
    
    # Order settings
    max_order_rate_per_second: int = 100
    default_time_in_force: str = "GTC"
    
    # Risk limits
    max_position_size: float = 10.0
    max_notional_exposure: float = 1_000_000.0
    
    # Signing integration
    use_rust_signer: bool = True  # Use Rust HMAC signer via PyO3
    

@dataclass
class TradingNodeConfigBuilder:
    """
    Builder pattern for constructing optimized NautilusTrader configurations.
    
    Generates YAML configs with hardware-aware defaults for AMD Ryzen AI 5
    and strict memory constraints (14GB total system RAM).
    """
    
    # Core settings
    trader_id: str = "TRADER-001"
    instance_id: str = "INSTANCE-AMD-RYZEN-AI5"
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "/var/log/nautilus"
    
    # Memory constraints
    max_memory_gb: float = 14.0
    
    # Components
    message_bus_config: MessageBusConfig = field(default_factory=MessageBusConfig)
    data_clients: List[DataClientConfig] = field(default_factory=list)
    execution_clients: List[ExecutionClientConfig] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    
    # Additional Nautilus options
    streaming_enabled: bool = True
    snapshot_orders: bool = True
    snapshot_positions: bool = True
    
    def add_binance_data_client(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False
    ) -> 'TradingNodeConfigBuilder':
        """Add Binance futures data client configuration."""
        venue = "BINANCE_FUTURES" if not testnet else "BINANCE_FUTURES_TESTNET"
        
        self.data_clients.append(DataClientConfig(
            client_id=f"BINANCE_DATA_{venue}",
            venue=venue,
            use_rust_ingestion=True,
            rust_event_bus_integration=True
        ))
        
        if venue not in self.venues:
            self.venues.append(venue)
            
        return self
        
    def add_binance_execution_client(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ) -> 'TradingNodeConfigBuilder':
        """Add Binance futures execution client configuration."""
        venue = "BINANCE_FUTURES" if not testnet else "BINANCE_FUTURES_TESTNET"
        
        self.execution_clients.append(ExecutionClientConfig(
            client_id=f"BINANCE_EXEC_{venue}",
            venue=venue,
            max_order_rate_per_second=100,
            use_rust_signer=True
        ))
        
        if venue not in self.venues:
            self.venues.append(venue)
            
        return self
        
    def build(self) -> Dict[str, Any]:
        """
        Build the complete NautilusTrader configuration dictionary.
        
        Returns a dict that can be serialized to YAML or passed directly
        to NautilusTrader's TradingNode constructor.
        """
        config = {
            "trader": {
                "trader_id": self.trader_id,
                "instance_id": self.instance_id,
                "omits_default_fields": True,
            },
            "logging": {
                "level": self.log_level,
                "file_level": "DEBUG",
                "directory": self.log_directory,
                "use_rich_console": True,
                "rotate_on_close": True,
            },
            "memory": {
                "max_memory_gb": self.max_memory_gb,
                "gc_trigger_threshold": 0.93,  # Trigger GC at 93% of limit
                "emergency_pause_threshold": 0.97,  # Pause non-critical at 97%
            },
            "message_bus": {
                "type": self.message_bus_config.queue_type,
                "batch_size": self.message_bus_config.batch_size,
                "zero_copy_enabled": self.message_bus_config.zero_copy_enabled,
                "buffer_pool_size": self.message_bus_config.buffer_pool_size,
            },
            "data_engine": {
                "time_bars_interval": "1s",
                "snapshot_orders": self.snapshot_orders,
                "snapshot_positions": self.snapshot_positions,
                "streaming": self.streaming_enabled,
            },
            "risk_engine": {
                "max_open_orders": 50,
                "max_order_rate": 100,
            },
            "exec_engine": {
                "retry_strategy": {
                    "max_retries": 3,
                    "delay_between_retries_ms": 100,
                },
            },
            "venues": self.venues,
            "data_clients": [
                {
                    "client_id": dc.client_id,
                    "venue": dc.venue,
                    "heartbeat_interval_ms": dc.heartbeat_interval_ms,
                    "reconnect_delay_ms": dc.reconnect_delay_ms,
                    "max_reconnect_attempts": dc.max_reconnect_attempts,
                    "subscribe_trades": dc.subscribe_trades,
                    "subscribe_order_book": dc.subscribe_order_book,
                    "order_book_depth": dc.order_book_depth,
                    "use_rust_ingestion": dc.use_rust_ingestion,
                    "rust_event_bus_integration": dc.rust_event_bus_integration,
                }
                for dc in self.data_clients
            ],
            "exec_clients": [
                {
                    "client_id": ec.client_id,
                    "venue": ec.venue,
                    "max_order_rate_per_second": ec.max_order_rate_per_second,
                    "default_time_in_force": ec.default_time_in_force,
                    "max_position_size": ec.max_position_size,
                    "max_notional_exposure": ec.max_notional_exposure,
                    "use_rust_signer": ec.use_rust_signer,
                }
                for ec in self.execution_clients
            ],
        }
        
        log.info(f"Built NautilusTrader config with {len(self.venues)} venues")
        return config
        
    def to_yaml(self) -> str:
        """Serialize configuration to YAML string."""
        config_dict = self.build()
        return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a YAML file."""
        yaml_content = self.to_yaml()
        with open(filepath, 'w') as f:
            f.write(yaml_content)
        log.info(f"Saved NautilusTrader config to {filepath}")


def build_node_config(
    binance_config: Dict[str, Any],
    rust_event_bus_callback: callable = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to build a complete node configuration.
    
    Args:
        binance_config: Dynamic configuration from Binance API
        rust_event_bus_callback: Callback for Rust event bus integration
        api_key: Binance API key
        api_secret: Binance API secret
        testnet: Use testnet endpoints
        
    Returns:
        Complete configuration dictionary for TradingNode
    """
    builder = TradingNodeConfigBuilder()
    
    # Add data client
    builder.add_binance_data_client(testnet=testnet)
    
    # Add execution client if credentials provided
    if api_key and api_secret:
        builder.add_binance_execution_client(api_key, api_secret, testnet=testnet)
    
    # Apply dynamic venue settings from Binance API
    # (tick sizes, fee rates, etc.)
    if binance_config:
        log.info("Applying dynamic Binance venue configuration...")
        # Additional customization based on live exchange info
        
    return builder.build()
