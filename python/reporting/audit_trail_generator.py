"""
Cryptographically signed audit trail generator.
Generates immutable audit trails of all automated risk overrides,
manual UI interventions, and system halts for institutional compliance.
"""

from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, BinaryIO
from pathlib import Path
from enum import Enum


logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    RISK_OVERRIDE = "risk_override"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_HALT = "system_halt"
    POSITION_FLATTEN = "position_flatten"
    EMERGENCY_CANCEL = "emergency_cancel"
    CONFIG_CHANGE = "config_change"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_KEY_ROTATION = "api_key_rotation"
    WITHDRAWAL_APPROVAL = "withdrawal_approval"


@dataclass
class AuditEvent:
    """Single audit event entry."""
    event_id: str
    timestamp: str  # ISO format UTC
    event_type: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    previous_state_hash: str
    current_state_hash: str
    signature: str = ""  # Cryptographic signature


@dataclass
class AuditChain:
    """Complete audit chain with hash linking."""
    genesis_hash: str
    events: List[AuditEvent] = field(default_factory=list)
    latest_hash: str = ""


class AuditTrailGenerator:
    """
    Generates cryptographically signed, immutable audit trails.
    
    Uses hash chaining to ensure tamper-evidence and provides
    digital signatures for regulatory compliance.
    """
    
    def __init__(
        self,
        output_directory: str,
        signing_key: Optional[bytes] = None,
        chain_id: str = "default",
    ):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate or use provided signing key
        if signing_key is None:
            self.signing_key = self._generate_signing_key()
        else:
            self.signing_key = signing_key
        
        self.chain_id = chain_id
        self.genesis_hash = self._compute_hash(f"genesis:{chain_id}")
        
        # Current chain state
        self._current_chain = AuditChain(
            genesis_hash=self.genesis_hash,
            latest_hash=self.genesis_hash,
        )
        
        # Buffer for batch writing
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = 100  # Write every 100 events
        
        # Statistics
        self._total_events = 0
        self._total_files_written = 0
    
    def _generate_signing_key(self) -> bytes:
        """Generate a secure signing key (in production, use HSM or KMS)."""
        import secrets
        return secrets.token_bytes(32)
    
    def _compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Sign an event using HMAC-SHA256."""
        import hmac
        
        event_data = json.dumps({
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "previous_state_hash": event.previous_state_hash,
            "current_state_hash": event.current_state_hash,
        }, sort_keys=True)
        
        signature = hmac.new(
            self.signing_key,
            event_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def record_event(
        self,
        event_type: AuditEventType,
        description: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        state_before: Optional[Dict[str, Any]] = None,
        state_after: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        Record an auditable event.
        
        Args:
            event_type: Type of event
            description: Human-readable description
            details: Event-specific details
            user_id: Optional user who triggered the event
            state_before: State before the event (for hashing)
            state_after: State after the event (for hashing)
        
        Returns:
            The recorded AuditEvent
        """
        import uuid
        
        # Compute state hashes
        prev_hash = self._current_chain.latest_hash
        
        state_before_str = json.dumps(state_before or {}, sort_keys=True)
        state_after_str = json.dumps(state_after or {}, sort_keys=True)
        
        current_hash = self._compute_hash(
            f"{prev_hash}:{state_before_str}:{state_after_str}"
        )
        
        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type.value,
            user_id=user_id,
            description=description,
            details=details,
            previous_state_hash=prev_hash,
            current_state_hash=current_hash,
        )
        
        # Sign the event
        event.signature = self._sign_event(event)
        
        # Add to chain
        self._current_chain.events.append(event)
        self._current_chain.latest_hash = current_hash
        self._event_buffer.append(event)
        self._total_events += 1
        
        # Flush buffer if full
        if len(self._event_buffer) >= self._buffer_size:
            self.flush_to_disk()
        
        logger.info(f"Audit event recorded: {event.event_id} ({event_type.value})")
        
        return event
    
    def record_risk_override(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        user_id: Optional[str] = None,
    ) -> AuditEvent:
        """Record a risk parameter override."""
        return self.record_event(
            event_type=AuditEventType.RISK_OVERRIDE,
            description=f"Risk parameter override: {parameter}",
            details={
                "parameter": parameter,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
            },
            user_id=user_id,
            state_before={"parameter": parameter, "value": old_value},
            state_after={"parameter": parameter, "value": new_value},
        )
    
    def record_manual_intervention(
        self,
        action: str,
        affected_positions: List[str],
        reason: str,
        user_id: Optional[str] = None,
    ) -> AuditEvent:
        """Record a manual trading intervention."""
        return self.record_event(
            event_type=AuditEventType.MANUAL_INTERVENTION,
            description=f"Manual intervention: {action}",
            details={
                "action": action,
                "affected_positions": affected_positions,
                "reason": reason,
            },
            user_id=user_id,
        )
    
    def record_system_halt(
        self,
        reason: str,
        triggered_by: str,  # AUTO or USER
        affected_strategies: List[str],
        user_id: Optional[str] = None,
    ) -> AuditEvent:
        """Record a system halt event."""
        return self.record_event(
            event_type=AuditEventType.SYSTEM_HALT,
            description=f"System halt: {reason}",
            details={
                "reason": reason,
                "triggered_by": triggered_by,
                "affected_strategies": affected_strategies,
            },
            user_id=user_id,
        )
    
    def flush_to_disk(self) -> Optional[Path]:
        """Flush buffered events to disk as JSONL."""
        if not self._event_buffer:
            return None
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"audit_trail_{self.chain_id}_{timestamp}.jsonl"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                for event in self._event_buffer:
                    # Convert to dict and write as JSON line
                    event_dict = asdict(event)
                    f.write(json.dumps(event_dict) + "\n")
            
            self._event_buffer.clear()
            self._total_files_written += 1
            
            logger.info(f"Audit trail flushed to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to flush audit trail: {e}")
            return None
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire audit chain."""
        expected_hash = self.genesis_hash
        
        for event in self._current_chain.events:
            # Verify hash chain
            if event.previous_state_hash != expected_hash:
                logger.error(f"Chain broken at event {event.event_id}")
                return False
            
            # Verify signature
            expected_sig = self._sign_event(event)
            if event.signature != expected_sig:
                logger.error(f"Signature invalid for event {event.event_id}")
                return False
            
            # Update expected hash
            expected_hash = event.current_state_hash
        
        return True
    
    def export_for_audit(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
    ) -> Dict[str, Any]:
        """
        Export audit trail for external audit.
        
        Returns a complete, verifiable audit package.
        """
        filtered_events = []
        
        for event in self._current_chain.events:
            event_time = datetime.fromisoformat(event.timestamp)
            
            # Apply filters
            if start_date and event_time < start_date:
                continue
            if end_date and event_time > end_date:
                continue
            if event_types and AuditEventType(event.event_type) not in event_types:
                continue
            
            filtered_events.append(asdict(event))
        
        return {
            "chain_id": self.chain_id,
            "genesis_hash": self.genesis_hash,
            "latest_hash": self._current_chain.latest_hash,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_events": len(filtered_events),
            "events": filtered_events,
            "integrity_verified": self.verify_chain_integrity(),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        event_type_counts: Dict[str, int] = {}
        for event in self._current_chain.events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        return {
            "total_events": self._total_events,
            "buffered_events": len(self._event_buffer),
            "files_written": self._total_files_written,
            "chain_length": len(self._current_chain.events),
            "genesis_hash": self.genesis_hash,
            "latest_hash": self._current_chain.latest_hash,
            "events_by_type": event_type_counts,
            "integrity_verified": self.verify_chain_integrity(),
        }
    
    def force_flush(self) -> Optional[Path]:
        """Force flush all buffered events to disk."""
        return self.flush_to_disk()


# Example usage
if __name__ == "__main__":
    generator = AuditTrailGenerator(output_directory="/data/audit_trails")
    
    # Record various events
    generator.record_risk_override(
        parameter="max_position_size",
        old_value=1000000,
        new_value=500000,
        reason="Increased market volatility",
        user_id="risk_manager_001",
    )
    
    generator.record_manual_intervention(
        action="Close all ETH positions",
        affected_positions=["ETH-PERP", "ETH-SPOT"],
        reason="Exchange maintenance window",
        user_id="trader_002",
    )
    
    generator.record_system_halt(
        reason="Circuit breaker triggered - 5% drawdown",
        triggered_by="AUTO",
        affected_strategies=["statarb", "momentum"],
    )
    
    # Flush to disk
    filepath = generator.force_flush()
    
    # Get statistics
    stats = generator.get_statistics()
    print(f"Audit trail stats: {stats}")
    
    # Verify integrity
    is_valid = generator.verify_chain_integrity()
    print(f"Chain integrity: {'VALID' if is_valid else 'INVALID'}")
