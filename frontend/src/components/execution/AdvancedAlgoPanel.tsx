import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAlgoStore } from '../../core/store';

type AlgoType = 'TWAP' | 'VWAP' | 'ICEBERG' | 'SNIPER';

interface AlgoParams {
  duration?: number; // minutes
  participationRate?: number; // percentage
  maxDisplaySize?: number;
  triggerPrice?: number;
  aggressionLevel?: number; // 1-10
}

export const AdvancedAlgoPanel: React.FC = () => {
  const [activeAlgo, setActiveAlgo] = useState<AlgoType | null>(null);
  const [params, setParams] = useState<AlgoParams>({});
  
  const updateAlgoConfig = useAlgoStore(useShallow(state => state.updateAlgoConfig));
  const isRunning = useAlgoStore(useShallow(state => state.isAlgoRunning));
  
  // Debounce timer ref to prevent WebSocket spam
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Optimized param update with debouncing
  const handleParamChange = useCallback((key: keyof AlgoParams, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }));
    
    // Clear existing timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    // Debounce updates to backend (300ms)
    debounceTimerRef.current = setTimeout(() => {
      if (activeAlgo) {
        updateAlgoConfig(activeAlgo, { [key]: value });
      }
    }, 300);
  }, [activeAlgo, updateAlgoConfig]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  const startAlgo = useCallback((type: AlgoType) => {
    setActiveAlgo(type);
    updateAlgoConfig(type, { ...params, enabled: true });
  }, [params, updateAlgoConfig]);

  const stopAlgo = useCallback(() => {
    if (activeAlgo) {
      updateAlgoConfig(activeAlgo, { enabled: false });
      setActiveAlgo(null);
    }
  }, [activeAlgo, updateAlgoConfig]);

  const renderParams = () => {
    switch (activeAlgo) {
      case 'TWAP':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Duration (minutes)</label>
              <input
                type="range"
                min="1"
                max="240"
                value={params.duration || 60}
                onChange={(e) => handleParamChange('duration', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-cyan-500"
              />
              <div className="text-right text-cyan-400 text-sm font-mono">{params.duration || 60}m</div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Slices per Minute</label>
              <input
                type="range"
                min="1"
                max="60"
                value={params.participationRate || 10}
                onChange={(e) => handleParamChange('participationRate', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-cyan-500"
              />
              <div className="text-right text-cyan-400 text-sm font-mono">{params.participationRate || 10}</div>
            </div>
          </div>
        );
      case 'VWAP':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Participation Rate (%)</label>
              <input
                type="range"
                min="1"
                max="100"
                value={params.participationRate || 20}
                onChange={(e) => handleParamChange('participationRate', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-magenta-500"
              />
              <div className="text-right text-magenta-400 text-sm font-mono">{params.participationRate || 20}%</div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Aggression Level</label>
              <input
                type="range"
                min="1"
                max="10"
                value={params.aggressionLevel || 5}
                onChange={(e) => handleParamChange('aggressionLevel', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-magenta-500"
              />
              <div className="text-right text-magenta-400 text-sm font-mono">{params.aggressionLevel || 5}/10</div>
            </div>
          </div>
        );
      case 'ICEBERG':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Max Display Size (%)</label>
              <input
                type="range"
                min="1"
                max="50"
                value={params.maxDisplaySize || 10}
                onChange={(e) => handleParamChange('maxDisplaySize', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-yellow-500"
              />
              <div className="text-right text-yellow-400 text-sm font-mono">{params.maxDisplaySize || 10}%</div>
            </div>
          </div>
        );
      case 'SNIPER':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Trigger Price Offset (bps)</label>
              <input
                type="range"
                min="1"
                max="100"
                value={params.triggerPrice || 10}
                onChange={(e) => handleParamChange('triggerPrice', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-red-500"
              />
              <div className="text-right text-red-400 text-sm font-mono">{params.triggerPrice || 10} bps</div>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Aggression</label>
              <input
                type="range"
                min="1"
                max="10"
                value={params.aggressionLevel || 8}
                onChange={(e) => handleParamChange('aggressionLevel', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg accent-red-500"
              />
              <div className="text-right text-red-400 text-sm font-mono">{params.aggressionLevel || 8}/10</div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl">
      <h3 className="text-magenta-400 font-bold text-lg mb-4 tracking-wider uppercase flex items-center justify-between">
        <span>Algo Execution</span>
        {isRunning && <span className="animate-pulse w-2 h-2 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e]"></span>}
      </h3>

      {/* Algo Selector */}
      {!activeAlgo ? (
        <div className="grid grid-cols-2 gap-3">
          {(['TWAP', 'VWAP', 'ICEBERG', 'SNIPER'] as AlgoType[]).map(algo => (
            <button
              key={algo}
              onClick={() => startAlgo(algo)}
              className={`p-4 rounded border transition-all duration-200 group ${
                algo === 'TWAP' ? 'border-cyan-900 hover:border-cyan-500 hover:bg-cyan-900/20' :
                algo === 'VWAP' ? 'border-magenta-900 hover:border-magenta-500 hover:bg-magenta-900/20' :
                algo === 'ICEBERG' ? 'border-yellow-900 hover:border-yellow-500 hover:bg-yellow-900/20' :
                'border-red-900 hover:border-red-500 hover:bg-red-900/20'
              }`}
            >
              <div className={`font-bold text-sm ${
                algo === 'TWAP' ? 'text-cyan-400' :
                algo === 'VWAP' ? 'text-magenta-400' :
                algo === 'ICEBERG' ? 'text-yellow-400' :
                'text-red-400'
              }`}>{algo}</div>
              <div className="text-xs text-gray-500 mt-1 group-hover:text-gray-300">
                {algo === 'TWAP' && 'Time Weighted Avg'}
                {algo === 'VWAP' && 'Volume Weighted Avg'}
                {algo === 'ICEBERG' && 'Hidden Liquidity'}
                {algo === 'SNIPER' && 'Liquidity Grab'}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-gray-800">
            <span className={`font-bold ${
              activeAlgo === 'TWAP' ? 'text-cyan-400' :
              activeAlgo === 'VWAP' ? 'text-magenta-400' :
              activeAlgo === 'ICEBERG' ? 'text-yellow-400' :
              'text-red-400'
            }`}>{activeAlgo} Active</span>
            <button
              onClick={stopAlgo}
              className="px-4 py-1 text-xs font-bold bg-red-900/50 text-red-400 border border-red-700 rounded hover:bg-red-800 transition-colors"
            >
              STOP
            </button>
          </div>
          
          {renderParams()}
          
          <div className="pt-3 border-t border-gray-800">
            <div className="flex justify-between text-xs text-gray-500">
              <span>Status:</span>
              <span className="text-green-400">{isRunning ? 'RUNNING' : 'READY'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvancedAlgoPanel;
