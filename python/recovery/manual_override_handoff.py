"""
Manual Override Handoff - Safely transitions control from automated Rust engine to human operator.
Gracefully pauses ML inference and locks order gateways during crisis situations.
"""

import time
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum, auto
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ControlState(Enum):
    """Control state of the trading system."""
    AUTOMATED = auto()      # Full automated trading
    TRANSITIONING = auto()  # In process of handoff
    MANUAL = auto()         # Human operator in control
    PAUSED = auto()         # All trading paused
    EMERGENCY = auto()      # Emergency halt active


class HandoffReason(Enum):
    """Reason for control handoff."""
    OPERATOR_REQUEST = "operator_request"
    SYSTEM_ANOMALY = "system_anomaly"
    CIRCUIT_BREAKER = "circuit_breaker"
    MAINTENANCE = "maintenance"
    EXCHANGE_OUTAGE = "exchange_outage"
    RISK_LIMIT = "risk_limit"


@dataclass
class HandoffRequest:
    """Request for control handoff."""
    request_id: str
    reason: HandoffReason
    requested_by: str
    timestamp: float = field(default_factory=time.time)
    target_state: ControlState = ControlState.MANUAL
    notes: str = ""
    expiry_seconds: float = 300.0  # Request expires after this
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.expiry_seconds


@dataclass
class HandoffResponse:
    """Response to handoff request."""
    request_id: str
    accepted: bool
    current_state: ControlState
    pending_orders_cancelled: int = 0
    positions_locked: bool = False
    ml_inference_paused: bool = False
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""


@dataclass
class OperatorSession:
    """Active operator session."""
    operator_id: str
    session_start: float
    last_heartbeat: float
    permissions: List[str]
    active_symbols: List[str]


class ManualOverrideHandoff:
    """
    Manages safe transition between automated and manual control.
    Implements proper locking, state verification, and audit logging.
    """
    
    def __init__(self):
        self._state = ControlState.AUTOMATED
        self._state_lock = threading.RLock()
        self._pending_request: Optional[HandoffRequest] = None
        self._active_session: Optional[OperatorSession] = None
        
        # Callbacks for system components
        self._order_gateway_callback: Optional[Callable] = None
        self._ml_inference_callback: Optional[Callable] = None
        self._position_manager_callback: Optional[Callable] = None
        self._risk_manager_callback: Optional[Callable] = None
        
        # Audit log
        self._audit_log: List[Dict] = []
        self._max_audit_entries = 10000
        
        # State tracking
        self._handoff_count = 0
        self._last_handoff_time = 0.0
        self._emergency_active = False
        
        # Authorized operators
        self._authorized_operators: Dict[str, List[str]] = {}
        
    def register_authorized_operator(
        self, 
        operator_id: str, 
        permissions: List[str]
    ) -> None:
        """Register an authorized operator with specific permissions."""
        self._authorized_operators[operator_id] = permissions
        self._log_audit("operator_registered", {
            "operator_id": operator_id,
            "permissions": permissions,
        })
    
    def set_component_callbacks(
        self,
        order_gateway: Optional[Callable[[ControlState], bool]] = None,
        ml_inference: Optional[Callable[[ControlState], bool]] = None,
        position_manager: Optional[Callable[[ControlState], bool]] = None,
        risk_manager: Optional[Callable[[ControlState], bool]] = None,
    ) -> None:
        """Set callbacks for transitioning system components."""
        self._order_gateway_callback = order_gateway
        self._ml_inference_callback = ml_inference
        self._position_manager_callback = position_manager
        self._risk_manager_callback = risk_manager
    
    def request_handoff(self, request: HandoffRequest) -> HandoffResponse:
        """
        Request a handoff from automated to manual control.
        Returns response indicating success or failure.
        """
        with self._state_lock:
            # Check if request is expired
            if request.is_expired():
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=False,
                    current_state=self._state,
                    error_message="Request expired",
                )
            
            # Check if operator is authorized
            if request.requested_by not in self._authorized_operators:
                self._log_audit("handoff_denied", {
                    "request_id": request.request_id,
                    "reason": "unauthorized_operator",
                })
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=False,
                    current_state=self._state,
                    error_message="Operator not authorized",
                )
            
            # Check if already in target state
            if self._state == request.target_state:
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=True,
                    current_state=self._state,
                )
            
            # Check rate limiting (max 1 handoff per minute)
            if time.time() - self._last_handoff_time < 60.0:
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=False,
                    current_state=self._state,
                    error_message="Rate limit exceeded",
                )
            
            # Begin transition
            self._pending_request = request
            self._state = ControlState.TRANSITIONING
            
            self._log_audit("handoff_started", {
                "request_id": request.request_id,
                "reason": request.reason.value,
                "requested_by": request.requested_by,
                "target_state": request.target_state.name,
            })
            
            # Execute transition
            try:
                result = self._execute_handoff(request)
                
                if result.accepted:
                    self._state = request.target_state
                    self._last_handoff_time = time.time()
                    self._handoff_count += 1
                    
                    # Create operator session for manual control
                    if request.target_state == ControlState.MANUAL:
                        self._active_session = OperatorSession(
                            operator_id=request.requested_by,
                            session_start=time.time(),
                            last_heartbeat=time.time(),
                            permissions=self._authorized_operators.get(
                                request.requested_by, []
                            ),
                            active_symbols=[],
                        )
                    
                    self._log_audit("handoff_completed", {
                        "request_id": request.request_id,
                        "new_state": self._state.name,
                    })
                else:
                    self._state = ControlState.AUTOMATED
                    self._log_audit("handoff_failed", {
                        "request_id": request.request_id,
                        "error": result.error_message,
                    })
                
                self._pending_request = None
                return result
                
            except Exception as e:
                self._state = ControlState.AUTOMATED
                self._pending_request = None
                self._log_audit("handoff_error", {
                    "request_id": request.request_id,
                    "error": str(e),
                })
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=False,
                    current_state=ControlState.AUTOMATED,
                    error_message=str(e),
                )
    
    def _execute_handoff(self, request: HandoffRequest) -> HandoffResponse:
        """Execute the actual handoff transition."""
        pending_orders_cancelled = 0
        positions_locked = False
        ml_inference_paused = False
        
        # Step 1: Pause ML inference first
        if self._ml_inference_callback:
            try:
                ml_inference_paused = self._ml_inference_callback(request.target_state)
            except Exception as e:
                logger.error(f"ML inference pause failed: {e}")
        
        # Step 2: Lock order gateway
        if self._order_gateway_callback:
            try:
                orders_cancelled = self._order_gateway_callback(request.target_state)
                pending_orders_cancelled = orders_cancelled if isinstance(orders_cancelled, int) else 0
            except Exception as e:
                logger.error(f"Order gateway lock failed: {e}")
                return HandoffResponse(
                    request_id=request.request_id,
                    accepted=False,
                    current_state=self._state,
                    error_message=f"Order gateway failed: {e}",
                )
        
        # Step 3: Lock positions if going to manual/emergency
        if self._position_manager_callback and request.target_state in (
            ControlState.MANUAL, ControlState.EMERGENCY, ControlState.PAUSED
        ):
            try:
                positions_locked = self._position_manager_callback(request.target_state)
            except Exception as e:
                logger.error(f"Position lock failed: {e}")
        
        # Step 4: Notify risk manager
        if self._risk_manager_callback:
            try:
                self._risk_manager_callback(request.target_state)
            except Exception as e:
                logger.error(f"Risk manager notification failed: {e}")
        
        return HandoffResponse(
            request_id=request.request_id,
            accepted=True,
            current_state=request.target_state,
            pending_orders_cancelled=pending_orders_cancelled,
            positions_locked=positions_locked,
            ml_inference_paused=ml_inference_paused,
        )
    
    def return_to_automated(self, operator_id: str) -> bool:
        """Return control to automated system."""
        with self._state_lock:
            if self._state != ControlState.MANUAL:
                return False
            
            if operator_id not in self._authorized_operators:
                return False
            
            # Verify no pending manual orders
            if self._active_session and self._active_session.active_symbols:
                logger.warning("Returning to automated with active manual symbols")
            
            # Execute reverse handoff
            request = HandoffRequest(
                request_id=f"return_{time.time_ns()}",
                reason=HandoffReason.OPERATOR_REQUEST,
                requested_by=operator_id,
                target_state=ControlState.AUTOMATED,
            )
            
            response = self._execute_handoff(request)
            
            if response.accepted:
                self._state = ControlState.AUTOMATED
                self._active_session = None
                self._last_handoff_time = time.time()
                self._handoff_count += 1
                
                self._log_audit("returned_to_automated", {
                    "operator_id": operator_id,
                })
                return True
            
            return False
    
    def emergency_halt(self, reason: str) -> bool:
        """Trigger immediate emergency halt."""
        with self._state_lock:
            self._emergency_active = True
            previous_state = self._state
            self._state = ControlState.EMERGENCY
            
            self._log_audit("emergency_halt", {
                "reason": reason,
                "previous_state": previous_state.name,
            })
            
            # Trigger all callbacks immediately
            for callback in [
                self._ml_inference_callback,
                self._order_gateway_callback,
                self._position_manager_callback,
                self._risk_manager_callback,
            ]:
                if callback:
                    try:
                        callback(ControlState.EMERGENCY)
                    except Exception as e:
                        logger.error(f"Emergency callback failed: {e}")
            
            return True
    
    def resume_from_emergency(self, authorized_by: str) -> bool:
        """Resume from emergency halt."""
        with self._state_lock:
            if not self._emergency_active:
                return False
            
            if authorized_by not in self._authorized_operators:
                return False
            
            self._emergency_active = False
            self._state = ControlState.PAUSED  # Go to paused first, not automated
            
            self._log_audit("resume_from_emergency", {
                "authorized_by": authorized_by,
            })
            
            return True
    
    def get_current_state(self) -> ControlState:
        """Get current control state."""
        return self._state
    
    def get_session_info(self) -> Optional[Dict]:
        """Get current operator session info."""
        if self._active_session:
            return asdict(self._active_session)
        return None
    
    def get_stats(self) -> Dict:
        """Get handoff statistics."""
        return {
            "current_state": self._state.name,
            "handoff_count": self._handoff_count,
            "last_handoff_time": self._last_handoff_time,
            "emergency_active": self._emergency_active,
            "pending_request": self._pending_request.request_id if self._pending_request else None,
            "active_operator": self._active_session.operator_id if self._active_session else None,
        }
    
    def _log_audit(self, event: str, details: Dict) -> None:
        """Log audit entry."""
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event": event,
            "details": details,
            "state": self._state.name,
        }
        
        self._audit_log.append(entry)
        
        # Trim old entries
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]


# Example usage and testing
if __name__ == "__main__":
    handoff = ManualOverrideHandoff()
    
    # Register authorized operator
    handoff.register_authorized_operator("trader_001", ["trade", "cancel", "view"])
    
    # Set mock callbacks
    def mock_order_gateway(state):
        print(f"Order gateway transitioning to {state.name}")
        return 5  # Cancelled 5 orders
    
    def mock_ml_inference(state):
        print(f"ML inference transitioning to {state.name}")
        return True
    
    handoff.set_component_callbacks(
        order_gateway=mock_order_gateway,
        ml_inference=mock_ml_inference,
    )
    
    # Request handoff
    request = HandoffRequest(
        request_id="req_001",
        reason=HandoffReason.OPERATOR_REQUEST,
        requested_by="trader_001",
        target_state=ControlState.MANUAL,
        notes="Testing manual override",
    )
    
    response = handoff.request_handoff(request)
    print(f"\nHandoff Response:")
    print(f"  Accepted: {response.accepted}")
    print(f"  Current State: {response.current_state.name}")
    print(f"  Orders Cancelled: {response.pending_orders_cancelled}")
    print(f"  ML Paused: {response.ml_inference_paused}")
    
    # Get stats
    print(f"\nStats: {handoff.get_stats()}")
    
    # Return to automated
    success = handoff.return_to_automated("trader_001")
    print(f"\nReturn to automated: {success}")
    print(f"Final state: {handoff.get_current_state().name}")
