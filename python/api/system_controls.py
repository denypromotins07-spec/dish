"""
Safe Python handlers for UI manual overrides.
Processes frontend commands like "Pause Trading", "Flatten All Positions", etc.
via secure message queues with strict validation.
"""

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import uuid


class ControlCommand(Enum):
    """Valid control commands from UI."""
    PAUSE_TRADING = "pause_trading"
    RESUME_TRADING = "resume_trading"
    FLATTEN_ALL = "flatten_all_positions"
    HALT_ML_INFERENCE = "halt_ml_inference"
    RESUME_ML_INFERENCE = "resume_ml_inference"
    TRIGGER_REBALANCE = "trigger_manual_rebalance"
    EMERGENCY_STOP = "emergency_stop"
    SET_RISK_LEVEL = "set_risk_level"
    ADJUST_POSITION_LIMIT = "adjust_position_limit"


@dataclass
class ControlRequest:
    """Control request from UI."""
    command: str
    request_id: str
    timestamp_ms: int
    parameters: Dict[str, Any]
    source: str = "ui"


@dataclass
class ControlResponse:
    """Response to control request."""
    request_id: str
    success: bool
    message: str
    timestamp_ms: int
    result: Optional[Dict] = None


class SystemControls:
    """
    Safe handler for UI manual override commands.
    Validates and routes commands to appropriate handlers.
    """

    def __init__(self):
        # Registered command handlers
        self._handlers: Dict[ControlCommand, Callable] = {}
        
        # Command queue (bounded)
        self._command_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Running state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # Response callbacks
        self._response_callbacks: Dict[str, asyncio.Future] = {}
        
        # Audit log (bounded)
        self._audit_log: List[Dict] = []
        self._max_audit_entries = 1000
        
        # Safety flags
        self._is_paused = False
        self._ml_halted = False
        self._emergency_stopped = False

    def register_handler(self, command: ControlCommand, handler: Callable):
        """Register a handler for a specific command."""
        self._handlers[command] = handler

    async def submit_command(
        self,
        command: str,
        parameters: Optional[Dict] = None,
        source: str = "ui",
    ) -> ControlResponse:
        """Submit a control command for processing."""
        request_id = str(uuid.uuid4())
        timestamp_ms = int(time.time() * 1000)
        
        # Validate command
        try:
            cmd_enum = ControlCommand(command)
        except ValueError:
            return ControlResponse(
                request_id=request_id,
                success=False,
                message=f"Unknown command: {command}",
                timestamp_ms=timestamp_ms,
            )
        
        # Create request
        request = ControlRequest(
            command=command,
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            parameters=parameters or {},
            source=source,
        )
        
        # Check for emergency stop state
        if self._emergency_stopped and command != ControlCommand.EMERGENCY_STOP.value:
            return ControlResponse(
                request_id=request_id,
                success=False,
                message="System is in emergency stop state. Only reset allowed.",
                timestamp_ms=timestamp_ms,
            )
        
        # Queue the command
        try:
            self._command_queue.put_nowait((cmd_enum, request))
        except asyncio.QueueFull:
            return ControlResponse(
                request_id=request_id,
                success=False,
                message="Command queue full. Try again later.",
                timestamp_ms=timestamp_ms,
            )
        
        # Wait for response
        future = asyncio.Future()
        self._response_callbacks[request_id] = future
        
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            return ControlResponse(
                request_id=request_id,
                success=False,
                message="Command processing timed out",
                timestamp_ms=int(time.time() * 1000),
            )
        finally:
            self._response_callbacks.pop(request_id, None)

    async def start_processor(self):
        """Start the command processor background task."""
        self._running = True
        
        async def process_loop():
            while self._running:
                try:
                    cmd, request = await asyncio.wait_for(
                        self._command_queue.get(),
                        timeout=1.0
                    )
                    await self._process_command(cmd, request)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error processing command: {e}")
        
        self._processor_task = asyncio.create_task(process_loop())

    async def stop_processor(self):
        """Stop the command processor."""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

    async def _process_command(self, cmd: ControlCommand, request: ControlRequest):
        """Process a single command."""
        timestamp_ms = int(time.time() * 1000)
        
        # Log the command
        self._log_command(cmd, request)
        
        # Check if handler exists
        handler = self._handlers.get(cmd)
        
        if handler is None:
            response = ControlResponse(
                request_id=request.request_id,
                success=False,
                message=f"No handler registered for command: {cmd.value}",
                timestamp_ms=timestamp_ms,
            )
        else:
            try:
                # Execute handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(request.parameters)
                else:
                    result = handler(request.parameters)
                
                response = ControlResponse(
                    request_id=request.request_id,
                    success=True,
                    message=f"Command {cmd.value} executed successfully",
                    timestamp_ms=timestamp_ms,
                    result=result,
                )
            except Exception as e:
                response = ControlResponse(
                    request_id=request.request_id,
                    success=False,
                    message=f"Command failed: {str(e)}",
                    timestamp_ms=timestamp_ms,
                )
        
        # Send response to waiter
        if request.request_id in self._response_callbacks:
            self._response_callbacks[request.request_id].set_result(response)

    def _log_command(self, cmd: ControlCommand, request: ControlRequest):
        """Log command for audit trail."""
        entry = {
            "timestamp_ms": request.timestamp_ms,
            "command": cmd.value,
            "request_id": request.request_id,
            "source": request.source,
            "parameters": request.parameters,
        }
        
        self._audit_log.append(entry)
        
        # Enforce memory bound
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    # Built-in command handlers
    
    async def handle_pause_trading(self, params: Dict) -> Dict:
        """Handle pause trading command."""
        self._is_paused = True
        return {"state": "paused"}

    async def handle_resume_trading(self, params: Dict) -> Dict:
        """Handle resume trading command."""
        if self._emergency_stopped:
            raise ValueError("Cannot resume - system is in emergency stop")
        self._is_paused = False
        return {"state": "running"}

    async def handle_flatten_all(self, params: Dict) -> Dict:
        """Handle flatten all positions command."""
        # This would trigger actual position flattening logic
        return {
            "action": "flatten_initiated",
            "positions_to_close": params.get("count", "all"),
        }

    async def handle_halt_ml(self, params: Dict) -> Dict:
        """Handle halt ML inference command."""
        self._ml_halted = True
        return {"state": "ml_halted"}

    async def handle_resume_ml(self, params: Dict) -> Dict:
        """Handle resume ML inference command."""
        self._ml_halted = False
        return {"state": "ml_running"}

    async def handle_emergency_stop(self, params: Dict) -> Dict:
        """Handle emergency stop command."""
        self._emergency_stopped = True
        self._is_paused = True
        self._ml_halted = True
        return {"state": "emergency_stopped"}

    def get_status(self) -> Dict:
        """Get current system control status."""
        return {
            "is_paused": self._is_paused,
            "ml_halted": self._ml_halted,
            "emergency_stopped": self._emergency_stopped,
            "pending_commands": self._command_queue.qsize(),
        }

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]


def create_system_controls_with_handlers() -> SystemControls:
    """Create SystemControls with default handlers registered."""
    controls = SystemControls()
    
    # Register built-in handlers
    controls.register_handler(ControlCommand.PAUSE_TRADING, controls.handle_pause_trading)
    controls.register_handler(ControlCommand.RESUME_TRADING, controls.handle_resume_trading)
    controls.register_handler(ControlCommand.FLATTEN_ALL, controls.handle_flatten_all)
    controls.register_handler(ControlCommand.HALT_ML_INFERENCE, controls.handle_halt_ml)
    controls.register_handler(ControlCommand.RESUME_ML_INFERENCE, controls.handle_resume_ml)
    controls.register_handler(ControlCommand.EMERGENCY_STOP, controls.handle_emergency_stop)
    
    return controls


# Example usage with WebSocket bridge
class WebSocketControlBridge:
    """Bridge WebSocket commands to SystemControls."""

    def __init__(self, controls: SystemControls, ws_send_func: Callable):
        self.controls = controls
        self.ws_send = ws_send_func

    async def handle_ui_message(self, message: str):
        """Handle incoming WebSocket message from UI."""
        try:
            data = json.loads(message)
            
            if data.get("type") != "control_command":
                return
            
            command = data.get("command")
            parameters = data.get("parameters", {})
            
            # Submit command
            response = await self.controls.submit_command(
                command=command,
                parameters=parameters,
                source="websocket",
            )
            
            # Send response back to UI
            await self.ws_send(json.dumps({
                "type": "control_response",
                "data": asdict(response),
            }))
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            await self.ws_send(json.dumps({
                "type": "error",
                "message": str(e),
            }))


if __name__ == '__main__':
    async def test_controls():
        controls = create_system_controls_with_handlers()
        await controls.start_processor()
        
        # Test pause command
        print("Testing pause command...")
        response = await controls.submit_command("pause_trading")
        print(f"Response: {asdict(response)}")
        
        # Test status
        print(f"Status: {controls.get_status()}")
        
        # Test emergency stop
        print("\nTesting emergency stop...")
        response = await controls.submit_command("emergency_stop")
        print(f"Response: {asdict(response)}")
        print(f"Status: {controls.get_status()}")
        
        # Test command during emergency stop
        print("\nTesting resume during emergency stop (should fail)...")
        response = await controls.submit_command("resume_trading")
        print(f"Response: {asdict(response)}")
        
        # Get audit log
        print(f"\nAudit log: {controls.get_audit_log()}")
        
        await controls.stop_processor()

    asyncio.run(test_controls())
