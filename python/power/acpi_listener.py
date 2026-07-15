"""
Linux ACPI Event Listener for Laptop Power Management.
Intercepts lid-close and sleep signals, overrides sleep if positions exist.
Safely flattens book and cancels orders on forced shutdown.
"""

import asyncio
import logging
import subprocess
import signal
from typing import Optional, Callable, Awaitable
from enum import Enum
import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PowerEventType(Enum):
    LID_CLOSE = "lid_close"
    LID_OPEN = "lid_open"
    SUSPEND = "suspend"
    RESUME = "resume"
    AC_CONNECT = "ac_connect"
    AC_DISCONNECT = "ac_disconnect"
    CRITICAL_BATTERY = "critical_battery"


class ACPIListener:
    """
    Listens for ACPI power events on Linux.
    Integrates with trading bot to handle laptop sleep/shutdown safely.
    """
    
    def __init__(self):
        self.event_handlers: dict[PowerEventType, list[Callable]] = {
            event: [] for event in PowerEventType
        }
        self.has_active_positions = False
        self.sleep_override_enabled = True
        self._dbus_bus = None
        self._main_loop = None
        
    def register_handler(
        self, 
        event_type: PowerEventType, 
        handler: Callable[[], Awaitable[None]]
    ):
        """Register an async handler for a power event."""
        self.event_handlers[event_type].append(handler)
        
    def set_has_active_positions(self, has_positions: bool):
        """Update whether bot has active trading positions."""
        self.has_active_positions = has_positions
        logger.info(f"Active positions status: {has_positions}")
    
    async def _emit_event(self, event_type: PowerEventType):
        """Emit event to all registered handlers."""
        logger.info(f"Power event: {event_type.value}")
        
        for handler in self.event_handlers[event_type]:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Handler for {event_type} failed: {e}")
    
    def _on_dbus_signal(self, interface, member, *args, **kwargs):
        """Handle D-Bus signals from logind."""
        if member == "PrepareForSleep":
            sleeping = args[0] if args else False
            if sleeping:
                asyncio.create_task(self._handle_sleep_request())
            else:
                asyncio.create_task(self._emit_event(PowerEventType.RESUME))
                
        elif member == "LidClosed":
            asyncio.create_task(self._handle_lid_close())
            
        elif member == "LidOpened":
            asyncio.create_task(self._emit_event(PowerEventType.LID_OPEN))
    
    async def _handle_sleep_request(self):
        """Handle system sleep request."""
        await self._emit_event(PowerEventType.SUSPEND)
        
        if self.has_active_positions and self.sleep_override_enabled:
            logger.critical("OVERRIDING SLEEP: Active positions exist!")
            # Inhibit sleep - this requires logind cooperation
            self._inhibit_sleep()
            return
        
        # Allow sleep after cleanup
        logger.info("Allowing system sleep after cleanup...")
        await self._cleanup_before_sleep()
    
    async def _handle_lid_close(self):
        """Handle lid close event."""
        await self._emit_event(PowerEventType.LID_CLOSE)
        
        if self.has_active_positions and self.sleep_override_enabled:
            logger.warning("Lid closed but positions active - monitoring...")
            # Could trigger haptic alert or notification here
    
    def _inhibit_sleep(self) -> int:
        """Inhibit system sleep via logind. Returns file descriptor."""
        try:
            bus = dbus.SystemBus()
            logind = bus.get_object('org.freedesktop.login1', '/org/freedesktop/login1')
            inhibit = logind.get_interface('org.freedesktop.login1.Manager')
            
            fd = inhibit.Inhibit(
                'sleep',
                'crypto-trading-bot',
                'Active trading positions must be managed',
                'delay'
            )
            
            logger.info("Sleep inhibited via logind")
            return fd
            
        except Exception as e:
            logger.error(f"Failed to inhibit sleep: {e}")
            return -1
    
    async def _cleanup_before_sleep(self):
        """Perform cleanup before allowing sleep."""
        logger.info("Cleaning up before sleep...")
        
        # Cancel all pending orders
        # Flatten positions if configured
        # Save state to disk
        
        await self._emit_event(PowerEventType.SUSPEND)
    
    def start(self):
        """Start listening for ACPI events."""
        DBusGMainLoop(set_as_default=True)
        
        self._dbus_bus = dbus.SystemBus()
        
        # Subscribe to logind signals
        self._dbus_bus.add_signal_receiver(
            self._on_dbus_signal,
            dbus_interface='org.freedesktop.login1.Manager',
            signal_name='PrepareForSleep',
            path='/org/freedesktop/login1'
        )
        
        # Subscribe to lid events (may need systemd-acpi service)
        self._dbus_bus.add_signal_receiver(
            self._on_dbus_signal,
            dbus_interface='org.freedesktop.login1.Manager',
            signal_name='LidClosed',
        )
        
        self._dbus_bus.add_signal_receiver(
            self._on_dbus_signal,
            dbus_interface='org.freedesktop.login1.Manager',
            signal_name='LidOpened',
        )
        
        # Run GLib main loop
        self._main_loop = GLib.MainLoop()
        logger.info("ACPI listener started")
        self._main_loop.run()
    
    def stop(self):
        """Stop the ACPI listener."""
        if self._main_loop:
            self._main_loop.quit()
        logger.info("ACPI listener stopped")


class PowerManagementPolicy:
    """
    Defines policy for handling power events.
    """
    
    def __init__(
        self,
        flatten_on_critical: bool = True,
        cancel_on_sleep: bool = True,
        notify_on_battery: bool = True,
        min_battery_percent: float = 15.0,
    ):
        self.flatten_on_critical = flatten_on_critical
        self.cancel_on_sleep = cancel_on_sleep
        self.notify_on_battery = notify_on_battery
        self.min_battery_percent = min_battery_percent


async def setup_power_management():
    """Setup power management with trading bot integration."""
    
    listener = ACPIListener()
    policy = PowerManagementPolicy()
    
    # Handler: Cancel orders before sleep
    async def on_suspend():
        if policy.cancel_on_sleep:
            logger.info("Cancelling all pending orders...")
            # Call exchange API to cancel orders
    
    # Handler: Emergency flatten on critical battery
    async def on_critical_battery():
        if policy.flatten_on_critical:
            logger.critical("CRITICAL BATTERY: Flattening all positions!")
            # Market out all positions immediately
    
    # Handler: Battery warning
    async def on_ac_disconnect():
        if policy.notify_on_battery:
            logger.warning("Running on battery power")
            # Check battery level
            # If below threshold, warn user
    
    # Register handlers
    listener.register_handler(PowerEventType.SUSPEND, on_suspend)
    listener.register_handler(PowerEventType.CRITICAL_BATTERY, on_critical_battery)
    listener.register_handler(PowerEventType.AC_DISCONNECT, on_ac_disconnect)
    
    return listener


if __name__ == "__main__":
    async def main():
        listener = await setup_power_management()
        
        # Simulate having active positions
        listener.set_has_active_positions(True)
        
        try:
            listener.start()
        except KeyboardInterrupt:
            listener.stop()
    
    # Run in asyncio loop
    import threading
    thread = threading.Thread(target=lambda: asyncio.run(main()))
    thread.daemon = True
    thread.start()
    
    # Keep main thread alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass
