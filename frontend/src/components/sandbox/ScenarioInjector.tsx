import { useState, useCallback } from 'react';
import { wsClient } from '../../core/wsClient';

type ShockType = 
  | 'flash_crash'
  | 'liquidity_evaporation'
  | 'spread_widening'
  | 'funding_spike'
  | 'volatility_surge'
  | 'correlation_breakdown';

interface ShockConfig {
  type: ShockType;
  magnitude: number;
  duration: number; // seconds
  delay: number; // seconds until injection
}

interface ScenarioInjectorProps {
  activeShocks?: ShockConfig[];
  onShockInjected?: (shock: ShockConfig) => void;
  onShockRemoved?: (shockId: string) => void;
}

export const ScenarioInjector: React.FC<ScenarioInjectorProps> = ({
  activeShocks = [],
  onShockInjected,
  onShockRemoved,
}) => {
  const [selectedType, setSelectedType] = useState<ShockType>('flash_crash');
  const [magnitude, setMagnitude] = useState<number>(5);
  const [duration, setDuration] = useState<number>(60);
  const [delay, setDelay] = useState<number>(0);

  const shockPresets: Record<ShockType, { label: string; color: string; defaultMag: number }> = {
    flash_crash: { label: 'Flash Crash', color: 'red', defaultMag: 10 },
    liquidity_evaporation: { label: 'Liquidity Evaporation', color: 'orange', defaultMag: 50 },
    spread_widening: { label: 'Spread Widening', color: 'yellow', defaultMag: 20 },
    funding_spike: { label: 'Funding Rate Spike', color: 'fuchsia', defaultMag: 0.1 },
    volatility_surge: { label: 'Volatility Surge', color: 'cyan', defaultMag: 100 },
    correlation_breakdown: { label: 'Correlation Breakdown', color: 'purple', defaultMag: 0.8 },
  };

  const injectShock = useCallback(() => {
    const shock: ShockConfig = {
      type: selectedType,
      magnitude,
      duration,
      delay,
    };

    // Send to Rust backend scenario_injector
    wsClient.sendBinary({
      type: 'SCENARIO_INJECT',
      payload: shock,
    });

    onShockInjected?.(shock);
  }, [selectedType, magnitude, duration, delay, onShockInjected]);

  const removeShock = useCallback((index: number) => {
    wsClient.sendBinary({
      type: 'SCENARIO_REMOVE',
      payload: { index },
    });
    onShockRemoved?.(index.toString());
  }, [onShockRemoved]);

  const clearAllShocks = useCallback(() => {
    wsClient.sendBinary({
      type: 'SCENARIO_CLEAR_ALL',
      payload: {},
    });
  }, []);

  const getColorClass = (color: string) => {
    return {
      red: 'text-red-400 bg-red-500/10 border-red-500/30',
      orange: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
      yellow: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
      fuchsia: 'text-fuchsia-400 bg-fuchsia-500/10 border-fuchsia-500/30',
      cyan: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
      purple: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
    }[color] || 'text-gray-400 bg-gray-500/10 border-gray-500/30';
  };

  return (
    <div className="p-4 bg-gray-900/80 backdrop-blur-md rounded-lg border border-orange-500/20 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-orange-400 font-mono text-sm uppercase tracking-wider flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          Synthetic Shock Injector
        </h3>
        {activeShocks.length > 0 && (
          <button
            onClick={clearAllShocks}
            className="text-xs px-2 py-1 text-red-400 hover:bg-red-500/20 rounded transition-colors"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Shock Type Selection */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
        {(Object.entries(shockPresets) as [ShockType, typeof shockPresets[keyof typeof shockPresets]][]).map(([type, preset]) => (
          <button
            key={type}
            onClick={() => {
              setSelectedType(type);
              setMagnitude(preset.defaultMag);
            }}
            className={`px-3 py-2 text-xs font-mono rounded border transition-all ${
              selectedType === type
                ? getColorClass(preset.color)
                : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* Configuration Sliders */}
      <div className="space-y-4 mb-4">
        {/* Magnitude */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-2">
            <span className="text-gray-400">Magnitude</span>
            <span className="text-orange-400">{magnitude.toFixed(1)}{selectedType === 'funding_spike' ? '%' : selectedType === 'volatility_surge' ? '%' : '%'}</span>
          </div>
          <input
            type="range"
            min={selectedType === 'funding_spike' ? 0.01 : 1}
            max={selectedType === 'funding_spike' ? 1 : 100}
            step={selectedType === 'funding_spike' ? 0.01 : 1}
            value={magnitude}
            onChange={(e) => setMagnitude(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-orange-500"
          />
        </div>

        {/* Duration */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-2">
            <span className="text-gray-400">Duration</span>
            <span className="text-cyan-400">{duration}s</span>
          </div>
          <input
            type="range"
            min="10"
            max="600"
            step="10"
            value={duration}
            onChange={(e) => setDuration(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
        </div>

        {/* Delay */}
        <div>
          <div className="flex justify-between text-xs font-mono mb-2">
            <span className="text-gray-400">Delay Before Injection</span>
            <span className="text-fuchsia-400">{delay}s</span>
          </div>
          <input
            type="range"
            min="0"
            max="120"
            step="5"
            value={delay}
            onChange={(e) => setDelay(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
          />
        </div>
      </div>

      {/* Inject Button */}
      <button
        onClick={injectShock}
        className="w-full px-4 py-3 bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white font-mono text-sm rounded-lg transition-all shadow-lg shadow-orange-500/20 flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Inject Shock
      </button>

      {/* Active Shocks List */}
      {activeShocks.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-2">
            Active Shocks ({activeShocks.length})
          </h4>
          {activeShocks.map((shock, idx) => {
            const preset = shockPresets[shock.type];
            return (
              <div
                key={idx}
                className={`flex items-center justify-between p-2 rounded border ${getColorClass(preset.color)}`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-current rounded-full animate-pulse" />
                  <span className="text-xs font-mono">{preset.label}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs opacity-70">
                    {shock.magnitude.toFixed(1)} × {shock.duration}s
                  </span>
                  <button
                    onClick={() => removeShock(idx)}
                    className="p-1 hover:bg-current/20 rounded transition-colors"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
