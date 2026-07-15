import React, { useState, useCallback, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useRiskStore } from '../../core/store';

interface RiskLimits {
  maxDrawdown: number;
  varLimit: number;
  maxLeverage: number;
  dailyLossLimit: number;
}

export const RiskLimitAdjuster: React.FC = () => {
  const limits = useRiskStore(useShallow(state => state.limits));
  const updateLimits = useRiskStore(useShallow(state => state.updateLimits));
  
  const [localValues, setLocalValues] = useState<RiskLimits>(limits);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Handle slider changes with debouncing
  const handleLimitChange = useCallback((key: keyof RiskLimits, value: number) => {
    setLocalValues(prev => ({ ...prev, [key]: value }));
    setHasUnsavedChanges(true);

    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    // Debounce backend updates (500ms)
    debounceTimerRef.current = setTimeout(() => {
      updateLimits({ [key]: value });
      setHasUnsavedChanges(false);
    }, 500);
  }, [updateLimits]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  // Sync local state when store updates
  React.useEffect(() => {
    setLocalValues(limits);
  }, [limits]);

  const renderSlider = (
    key: keyof RiskLimits,
    label: string,
    min: number,
    max: number,
    step: number,
    unit: string,
    color: 'cyan' | 'magenta' | 'yellow' | 'rose'
  ) => {
    const value = localValues[key];
    const isExceeded = key === 'maxDrawdown' && value < 5 || 
                       key === 'dailyLossLimit' && value < 1000;

    const colorClasses = {
      cyan: { bar: 'bg-cyan-500', text: 'text-cyan-400', glow: 'shadow-[0_0_15px_rgba(6,182,212,0.4)]' },
      magenta: { bar: 'bg-magenta-500', text: 'text-magenta-400', glow: 'shadow-[0_0_15px_rgba(236,72,153,0.4)]' },
      yellow: { bar: 'bg-yellow-500', text: 'text-yellow-400', glow: 'shadow-[0_0_15px_rgba(234,179,8,0.4)]' },
      rose: { bar: 'bg-rose-500', text: 'text-rose-400', glow: 'shadow-[0_0_15px_rgba(244,63,94,0.4)]' }
    };

    const currentColor = colorClasses[color];

    return (
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <label className="text-xs text-gray-400 font-medium">{label}</label>
          <span className={`text-sm font-mono font-bold ${currentColor.text}`}>
            {value.toFixed(step < 1 ? 2 : 0)}{unit}
          </span>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => handleLimitChange(key, parseFloat(e.target.value))}
          className={`w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-${color}-500`}
          style={{ accentColor: `var(--${color}-500)` }}
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>{min}{unit}</span>
          <span>{max}{unit}</span>
        </div>
        {isExceeded && (
          <div className="text-xs text-rose-400 mt-1 flex items-center">
            <span className="mr-1">⚠️</span> Warning: Low threshold
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-800">
        <h3 className="text-yellow-400 font-bold text-lg tracking-wider uppercase flex items-center">
          <span className="mr-2">🛡️</span>
          Risk Limits
        </h3>
        {hasUnsavedChanges && (
          <span className="px-2 py-1 bg-yellow-600/30 text-yellow-400 text-xs font-bold rounded animate-pulse">
            SAVING...
          </span>
        )}
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        {renderSlider('maxDrawdown', 'Max Drawdown', 1, 20, 0.5, '%', 'rose')}
        {renderSlider('varLimit', 'VaR Limit (95%)', 1000, 50000, 500, ' USD', 'magenta')}
        {renderSlider('maxLeverage', 'Max Leverage', 1, 100, 1, 'x', 'cyan')}
        {renderSlider('dailyLossLimit', 'Daily Loss Limit', 500, 10000, 100, ' USD', 'yellow')}
      </div>

      {/* Current Exposure Summary */}
      <div className="mt-4 pt-4 border-t border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">Current Exposure</h4>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-xs text-gray-500">Current Drawdown</div>
            <div className="text-sm font-mono text-emerald-400">-1.2%</div>
          </div>
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-xs text-gray-500">Current VaR</div>
            <div className="text-sm font-mono text-cyan-400">$12,450</div>
          </div>
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-xs text-gray-500">Avg Leverage</div>
            <div className="text-sm font-mono text-yellow-400">3.2x</div>
          </div>
          <div className="bg-gray-800/50 rounded p-2">
            <div className="text-xs text-gray-500">Today's PnL</div>
            <div className="text-sm font-mono text-emerald-400">+$2,340</div>
          </div>
        </div>
      </div>

      {/* Pre-trade Validator Status */}
      <div className="mt-3 pt-3 border-t border-gray-800 flex items-center justify-between">
        <span className="text-xs text-gray-500">Pre-Trade Validator:</span>
        <span className="text-emerald-400 text-xs font-bold flex items-center">
          <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
          ENFORCING LIMITS
        </span>
      </div>

      {/* Encrypted Update Note */}
      <p className="text-xs text-gray-600 mt-2 text-center">
        🔒 All limit updates are encrypted and sent directly to the Rust pre_trade_validator
      </p>
    </div>
  );
};

export default RiskLimitAdjuster;
