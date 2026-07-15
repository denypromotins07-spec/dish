import React, { memo, useMemo, useEffect } from 'react';
import { useUIStore } from '@/core/store';
import { wsClient } from '@/core/wsClient';

/**
 * Real-time network status indicator.
 * Displays WebSocket ping, packet loss, and backend sync state
 * using glowing neon status dots and lightweight CSS animations.
 */
export const ConnectionStatus = memo(() => {
  const wsConnected = useUIStore((state) => state.wsConnected);
  const wsPing = useUIStore((state) => state.wsPing);
  const packetLoss = useUIStore((state) => state.packetLoss);

  // Update connection status periodically
  useEffect(() => {
    const updateStatus = () => {
      const connected = wsClient.isConnected();
      const ping = wsClient.getPing();
      
      useUIStore.getState().updateConnectionStatus(
        connected,
        ping,
        packetLoss
      );
    };

    updateStatus();
    const interval = setInterval(updateStatus, 1000);

    return () => clearInterval(interval);
  }, [packetLoss]);

  // Memoize status calculations
  const statusInfo = useMemo(() => {
    if (!wsConnected) {
      return {
        label: 'DISCONNECTED',
        class: 'disconnected',
        colorClass: 'text-error',
        bgColor: 'bg-error/10',
        borderColor: 'border-error/30',
      };
    }

    if (wsPing > 500) {
      return {
        label: 'HIGH LATENCY',
        class: 'connecting',
        colorClass: 'text-warning',
        bgColor: 'bg-warning/10',
        borderColor: 'border-warning/30',
      };
    }

    if (wsPing > 100) {
      return {
        label: 'DEGRADED',
        class: 'connecting',
        colorClass: 'text-warning',
        bgColor: 'bg-warning/10',
        borderColor: 'border-warning/30',
      };
    }

    return {
      label: 'CONNECTED',
      class: 'connected',
      colorClass: 'text-success',
      bgColor: 'bg-success/10',
      borderColor: 'border-success/30',
    };
  }, [wsConnected, wsPing]);

  // Format ping display
  const formatPing = (ping: number): string => {
    if (ping === 0 && !wsConnected) return '--';
    return `${ping}ms`;
  };

  // Packet loss indicator
  const packetLossIndicator = useMemo(() => {
    if (packetLoss === 0) return null;
    
    let colorClass = 'text-success';
    if (packetLoss > 5) colorClass = 'text-warning';
    if (packetLoss > 10) colorClass = 'text-error';

    return (
      <div className={`text-xs text-mono-tight ${colorClass}`}>
        LOSS: {packetLoss.toFixed(1)}%
      </div>
    );
  }, [packetLoss]);

  return (
    <div className={`flex items-center gap-3 px-4 py-2 rounded-lg 
                     ${statusInfo.bgColor} border ${statusInfo.borderColor}
                     gpu-accelerated layout-stable transition-all duration-200`}>
      
      {/* Status dot with glow animation */}
      <div className="relative">
        <span className={`status-dot ${statusInfo.class}`} />
        {/* Outer glow ring */}
        <span className={`absolute inset-0 rounded-full animate-ping opacity-75
                          ${wsConnected ? 'bg-success' : 'bg-error'}`} 
              style={{ animationDuration: '2s' }} />
      </div>

      {/* Status text */}
      <div className="flex flex-col">
        <span className={`text-xs font-bold uppercase tracking-wider ${statusInfo.colorClass}`}>
          {statusInfo.label}
        </span>
        <span className="text-xs text-gray-500 text-mono-tight">
          PING: {formatPing(wsPing)}
        </span>
      </div>

      {/* Packet loss indicator */}
      {packetLossIndicator}

      {/* Backend sync indicator */}
      {wsConnected && (
        <div className="ml-auto flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-accent-cyan animate-pulse" />
          <span className="text-xs text-accent-cyan text-mono-tight">SYNC</span>
        </div>
      )}

      {/* Reconnect button (only when disconnected) */}
      {!wsConnected && (
        <button
          onClick={() => wsClient.connect()}
          className="ml-2 px-3 py-1 text-xs font-medium uppercase tracking-wide
                     bg-surface-active hover:bg-surface-hover border border-surface-active
                     rounded transition-all duration-150 text-white"
        >
          RECONNECT
        </button>
      )}
    </div>
  );
});

ConnectionStatus.displayName = 'ConnectionStatus';

export default ConnectionStatus;
