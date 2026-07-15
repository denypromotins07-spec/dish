import React, { memo, useMemo } from 'react';
import { useUIStore, useHFStore } from '@/core/store';

/**
 * Sleek, low-overhead UI panel displaying real-time RAM usage,
 * CPU temperature, and active thread counts.
 * Pulls from Zustand store with strict memoization to prevent unnecessary re-renders.
 */
export const SystemHealthPanel = memo(() => {
  // Select only necessary state slices to minimize re-renders
  const cpuTemp = useUIStore((state) => state.cpuTemp);
  const ramUsage = useUIStore((state) => state.ramUsage);
  const activeThreads = useUIStore((state) => state.activeThreads);
  const gpuLoad = useUIStore((state) => state.gpuLoad);
  const fpsThrottler = useUIStore((state) => state.fpsThrottler);
  
  const tickCount = useHFStore((state) => state.tickCount);

  // Memoize status color calculations
  const tempStatusColor = useMemo(() => {
    if (cpuTemp < 50) return 'text-success';
    if (cpuTemp < 70) return 'text-warning';
    if (cpuTemp < 85) return 'text-error';
    return 'text-error animate-pulse-fast';
  }, [cpuTemp]);

  const ramStatusColor = useMemo(() => {
    if (ramUsage < 60) return 'text-success';
    if (ramUsage < 80) return 'text-warning';
    return 'text-error';
  }, [ramUsage]);

  const gpuStatusColor = useMemo(() => {
    if (gpuLoad < 50) return 'text-success';
    if (gpuLoad < 80) return 'text-warning';
    return 'text-error';
  }, [gpuLoad]);

  // Format numbers with fixed width for monospace alignment
  const formatTemp = (temp: number) => `${temp.toFixed(1)}°C`;
  const formatPercent = (pct: number) => `${pct.toFixed(1)}%`;
  const formatThreads = (threads: number) => threads.toString().padStart(3, '0');
  const formatFPS = (fps: number) => fps.toString().padStart(2, '0');

  return (
    <div className="w-full h-full grid grid-cols-5 gap-4 p-2">
      {/* CPU Temperature */}
      <div className="flex flex-col items-center justify-center p-3 rounded-lg 
                      bg-surface-primary/50 border border-surface-active/30 
                      gpu-accelerated layout-stable">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">CPU Temp</div>
        <div className={`text-lg font-bold text-mono-tight ${tempStatusColor}`}>
          {formatTemp(cpuTemp)}
        </div>
        {/* Temperature bar */}
        <div className="w-full h-1 mt-2 bg-background-tertiary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-300 ${
              cpuTemp < 50 ? 'bg-success' : cpuTemp < 70 ? 'bg-warning' : 'bg-error'
            }`}
            style={{ width: `${Math.min(100, (cpuTemp / 100) * 100)}%` }}
          />
        </div>
      </div>

      {/* RAM Usage */}
      <div className="flex flex-col items-center justify-center p-3 rounded-lg 
                      bg-surface-primary/50 border border-surface-active/30 
                      gpu-accelerated layout-stable">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">RAM Usage</div>
        <div className={`text-lg font-bold text-mono-tight ${ramStatusColor}`}>
          {formatPercent(ramUsage)}
        </div>
        {/* RAM bar */}
        <div className="w-full h-1 mt-2 bg-background-tertiary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-300 ${
              ramUsage < 60 ? 'bg-success' : ramUsage < 80 ? 'bg-warning' : 'bg-error'
            }`}
            style={{ width: `${Math.min(100, ramUsage)}%` }}
          />
        </div>
      </div>

      {/* GPU Load */}
      <div className="flex flex-col items-center justify-center p-3 rounded-lg 
                      bg-surface-primary/50 border border-surface-active/30 
                      gpu-accelerated layout-stable">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">GPU Load</div>
        <div className={`text-lg font-bold text-mono-tight ${gpuStatusColor}`}>
          {formatPercent(gpuLoad)}
        </div>
        {/* GPU bar */}
        <div className="w-full h-1 mt-2 bg-background-tertiary rounded-full overflow-hidden">
          <div 
            className={`h-full transition-all duration-300 ${
              gpuLoad < 50 ? 'bg-success' : gpuLoad < 80 ? 'bg-warning' : 'bg-error'
            }`}
            style={{ width: `${Math.min(100, gpuLoad)}%` }}
          />
        </div>
      </div>

      {/* Active Threads */}
      <div className="flex flex-col items-center justify-center p-3 rounded-lg 
                      bg-surface-primary/50 border border-surface-active/30 
                      gpu-accelerated layout-stable">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Threads</div>
        <div className="text-lg font-bold text-mono-tight text-accent-cyan">
          {formatThreads(activeThreads)}
        </div>
        <div className="text-xs text-gray-600 mt-1">active</div>
      </div>

      {/* FPS Throttler */}
      <div className="flex flex-col items-center justify-center p-3 rounded-lg 
                      bg-surface-primary/50 border border-surface-active/30 
                      gpu-accelerated layout-stable">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Render FPS</div>
        <div className="text-lg font-bold text-mono-tight text-accent-magenta">
          {formatFPS(fpsThrottler)}
        </div>
        <div className="text-xs text-gray-600 mt-1">target</div>
      </div>

      {/* Tick Counter (hidden debug info) */}
      <div className="col-span-5 flex items-center justify-center gap-4 text-xs text-gray-600">
        <span>TICKS: {tickCount.toLocaleString()}</span>
        <span>|</span>
        <span>MEMORY: {(performance as any).memory?.usedJSHeapSize 
          ? `${((performance as any).memory.usedJSHeapSize / 1024 / 1024).toFixed(1)} MB`
          : 'N/A'}</span>
      </div>
    </div>
  );
});

SystemHealthPanel.displayName = 'SystemHealthPanel';

export default SystemHealthPanel;
