#!/usr/bin/env python3
"""
Asynchronous, append-only audit logger that records every authenticated API request,
order cancellation, and configuration change. Uses cryptographic hashing to ensure
logs cannot be tampered with.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    API_REQUEST = "api_request"
    ORDER_SUBMIT = "order_submit"
    ORDER_CANCEL = "order_cancel"
    ORDER_MODIFY = "order_modify"
    CONFIG_CHANGE = "config_change"
    AUTH_FAILURE = "auth_failure"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    KEY_ROTATION = "key_rotation"
    SECURITY_ALERT = "security_alert"


@dataclass
class AuditEvent:
    """Represents a single audit event."""
    timestamp: str
    event_type: str
    exchange: str
    action: str
    details: Dict[str, Any]
    sequence_number: int
    previous_hash: str
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Computes the SHA-256 hash of this event."""
        data = {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "exchange": self.exchange,
            "action": self.action,
            "details": self.details,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self.compute_hash()


class AuditLogger:
    """
    Cryptographically secure, append-only audit logger.
    Implements hash chaining to detect tampering.
    """

    def __init__(
        self,
        log_dir: str = "./audit_logs",
        max_memory_events: int = 1000,
        flush_interval_seconds: int = 5,
        hmac_secret: Optional[bytes] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_events = max_memory_events
        self.flush_interval_seconds = flush_interval_seconds
        self.hmac_secret = hmac_secret or os.urandom(32)
        
        # In-memory event buffer
        self._event_buffer: deque[AuditEvent] = deque(maxlen=max_memory_events)
        self._sequence_number: int = 0
        self._last_hash: str = "0" * 64  # Genesis hash
        
        # File handles per day
        self._current_file: Optional[Any] = None
        self._current_date: Optional[str] = None
        
        # Background flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._running: bool = False

    async def start(self) -> None:
        """Starts the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        await self.log_event(
            event_type=AuditEventType.SYSTEM_START,
            exchange="SYSTEM",
            action="audit_logger_started",
            details={"version": "1.0.0", "hostname": os.uname().nodename},
        )

    async def stop(self) -> None:
        """Stops the audit logger and flushes remaining events."""
        await self.log_event(
            event_type=AuditEventType.SYSTEM_STOP,
            exchange="SYSTEM",
            action="audit_logger_stopped",
            details={"events_logged": self._sequence_number},
        )
        
        self._running = False
        if self._flush_task:
            await self._flush_task
        await self._flush_now()
        if self._current_file:
            self._current_file.close()

    async def _flush_loop(self) -> None:
        """Background loop to periodically flush events to disk."""
        while self._running:
            await asyncio.sleep(self.flush_interval_seconds)
            await self._flush_now()

    async def _flush_now(self) -> None:
        """Flushes all buffered events to disk immediately."""
        if not self._event_buffer:
            return
        
        # Get today's date for file naming
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Rotate file if date changed
        if today != self._current_date:
            if self._current_file:
                self._current_file.close()
            self._current_date = today
            log_path = self.log_dir / f"audit_{today}.jsonl"
            self._current_file = open(log_path, "a", encoding="utf-8")
        
        # Write events
        for event in list(self._event_buffer):
            line = json.dumps(asdict(event), sort_keys=True)
            
            # Add HMAC signature for integrity verification
            signature = hmac.new(
                self.hmac_secret,
                line.encode(),
                hashlib.sha256
            ).hexdigest()
            
            self._current_file.write(f"{line}|{signature}\n")
        
        self._current_file.flush()
        os.fsync(self._current_file.fileno())
        
        # Clear buffer after successful flush
        self._event_buffer.clear()

    async def log_event(
        self,
        event_type: AuditEventType,
        exchange: str,
        action: str,
        details: Dict[str, Any],
    ) -> AuditEvent:
        """Logs a new audit event with hash chaining."""
        self._sequence_number += 1
        
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type.value,
            exchange=exchange,
            action=action,
            details=details,
            sequence_number=self._sequence_number,
            previous_hash=self._last_hash,
        )
        
        # Compute hash including previous hash (hash chain)
        event.event_hash = event.compute_hash()
        
        # Update last hash for next event
        self._last_hash = event.event_hash
        
        # Add to buffer
        self._event_buffer.append(event)
        
        logger.debug(f"Audit event logged: {event.event_type} - {event.action}")
        return event

    async def log_api_request(
        self,
        exchange: str,
        endpoint: str,
        method: str,
        response_code: int,
        latency_ms: float,
        request_id: Optional[str] = None,
    ) -> AuditEvent:
        """Logs an API request event."""
        return await self.log_event(
            event_type=AuditEventType.API_REQUEST,
            exchange=exchange,
            action=f"{method} {endpoint}",
            details={
                "endpoint": endpoint,
                "method": method,
                "response_code": response_code,
                "latency_ms": latency_ms,
                "request_id": request_id,
            },
        )

    async def log_order_submit(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
    ) -> AuditEvent:
        """Logs an order submission event."""
        return await self.log_event(
            event_type=AuditEventType.ORDER_SUBMIT,
            exchange=exchange,
            action="order_submit",
            details={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
            },
        )

    async def log_order_cancel(
        self,
        exchange: str,
        symbol: str,
        order_id: str,
        reason: str = "user_requested",
    ) -> AuditEvent:
        """Logs an order cancellation event."""
        return await self.log_event(
            event_type=AuditEventType.ORDER_CANCEL,
            exchange=exchange,
            action="order_cancel",
            details={
                "symbol": symbol,
                "order_id": order_id,
                "reason": reason,
            },
        )

    async def log_config_change(
        self,
        exchange: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        reason: str = "",
    ) -> AuditEvent:
        """Logs a configuration change event."""
        return await self.log_event(
            event_type=AuditEventType.CONFIG_CHANGE,
            exchange=exchange,
            action="config_change",
            details={
                "config_key": config_key,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "reason": reason,
            },
        )

    async def log_security_alert(
        self,
        exchange: str,
        alert_type: str,
        severity: str,
        message: str,
    ) -> AuditEvent:
        """Logs a security alert event."""
        return await self.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            exchange=exchange,
            action="security_alert",
            details={
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
            },
        )

    def verify_integrity(self, log_file_path: str) -> bool:
        """
        Verifies the integrity of an audit log file by checking hash chains and HMAC signatures.
        Returns True if the log is intact, False if tampering is detected.
        """
        path = Path(log_file_path)
        if not path.exists():
            logger.error(f"Log file not found: {log_file_path}")
            return False
        
        previous_hash = "0" * 64
        line_count = 0
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split line and signature
                try:
                    json_part, signature = line.rsplit("|", 1)
                except ValueError:
                    logger.error(f"Invalid line format at line {line_count + 1}")
                    return False
                
                # Verify HMAC signature
                expected_sig = hmac.new(
                    self.hmac_secret,
                    json_part.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(signature, expected_sig):
                    logger.error(f"HMAC verification failed at line {line_count + 1}")
                    return False
                
                # Parse and verify hash chain
                try:
                    event_data = json.loads(json_part)
                    event = AuditEvent(**event_data)
                    
                    computed_hash = event.compute_hash()
                    if computed_hash != event.event_hash:
                        logger.error(f"Hash mismatch at line {line_count + 1}")
                        return False
                    
                    if event.previous_hash != previous_hash:
                        logger.error(f"Hash chain broken at line {line_count + 1}")
                        return False
                    
                    previous_hash = event.event_hash
                    line_count += 1
                    
                except Exception as e:
                    logger.error(f"Error parsing line {line_count + 1}: {e}")
                    return False
        
        logger.info(f"Integrity verification passed: {line_count} events verified")
        return True


async def main():
    """Example usage of the AuditLogger."""
    logger_instance = AuditLogger(
        log_dir="./audit_logs",
        flush_interval_seconds=2,
    )
    
    await logger_instance.start()
    
    # Log some example events
    await logger_instance.log_api_request(
        exchange="binance",
        endpoint="/api/v3/order",
        method="POST",
        response_code=200,
        latency_ms=15.3,
    )
    
    await logger_instance.log_order_submit(
        exchange="binance",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        price=45000.0,
    )
    
    await logger_instance.log_config_change(
        exchange="binance",
        config_key="max_position_size",
        old_value=1.0,
        new_value=2.0,
        reason="strategy_update",
    )
    
    # Wait for flush
    await asyncio.sleep(3)
    
    await logger_instance.stop()
    
    # Verify integrity
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = f"./audit_logs/audit_{today}.jsonl"
    is_valid = logger_instance.verify_integrity(log_file)
    print(f"Log integrity: {'VALID' if is_valid else 'TAMPERED'}")


if __name__ == "__main__":
    asyncio.run(main())
