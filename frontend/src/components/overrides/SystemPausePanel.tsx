import React, { useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useSystemStore } from '../../core/store';

type PauseModule = 'ML_INFERENCE' | 'STRATEGY_EXEC' | 'MARKET_MAKING' | 'ORDER_ROUTING';

export const SystemPausePanel: React.FC = () => {
  const pausedModules = useSystemStore(useShallow(state => state.pausedModules));
  const toggleModulePause = useSystemStore(useShallow(state => state.toggleModulePause));

  const modules: { id: PauseModule; label: string; description: string; icon: string }[] = [
    {
      id: 'ML_INFERENCE',
      label: 'ML Inference',
      description: 'Pause neural network predictions & regime detection',
      icon: '🧠'
    },
    {
      id: 'STRATEGY_EXEC',
      label: 'Strategy Execution',
      description: 'Freeze all alpha strategy signals',
      icon: '📈'
    },
    {
      id: 'MARKET_MAKING',
      label: 'Market Making',
      description: 'Stop automated quoting & spread management',
      icon: '🔄'
    },
    {
      id: 'ORDER_ROUTING',
      label: 'Smart Order Routing',
      description: 'Halt cross-venue order splitting',
      icon: '🔀'
    }
  ];

  const handleToggle = useCallback((moduleId: PauseModule) => {
    toggleModulePause(moduleId);
  }, [toggleModulePause]);

  const isPaused = (moduleId: PauseModule) => pausedModules.includes(moduleId);

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl">
      <h3 className="text-yellow-400 font-bold text-lg mb-4 tracking-wider uppercase flex items-center">
        <span className="mr-2">⏸️</span>
        System Pause Controls
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {modules.map((module) => (
          <button
            key={module.id}
            onClick={() => handleToggle(module.id)}
            className={`relative p-4 rounded-lg border transition-all duration-200 text-left group ${
              isPaused(module.id)
                ? 'bg-yellow-900/20 border-yellow-600 shadow-[0_0_20px_rgba(234,179,8,0.3)]'
                : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
            }`}
          >
            {/* Status Indicator */}
            <div className="absolute top-3 right-3">
              <div className={`w-3 h-3 rounded-full transition-all ${
                isPaused(module.id)
                  ? 'bg-yellow-500 shadow-[0_0_10px_#eab308] animate-pulse'
                  : 'bg-emerald-500 shadow-[0_0_10px_#10b981]'
              }`} />
            </div>

            {/* Icon */}
            <div className="text-2xl mb-2">{module.icon}</div>

            {/* Label */}
            <div className={`font-bold text-sm mb-1 ${
              isPaused(module.id) ? 'text-yellow-400' : 'text-white'
            }`}>
              {module.label}
            </div>

            {/* Description */}
            <div className="text-xs text-gray-500 group-hover:text-gray-400 transition-colors">
              {module.description}
            </div>

            {/* Status Badge */}
            <div className={`mt-2 inline-block px-2 py-1 text-xs font-bold rounded ${
              isPaused(module.id)
                ? 'bg-yellow-600/30 text-yellow-400'
                : 'bg-emerald-600/30 text-emerald-400'
            }`}>
              {isPaused(module.id) ? 'PAUSED' : 'ACTIVE'}
            </div>
          </button>
        ))}
      </div>

      {/* Global Controls */}
      <div className="mt-4 pt-4 border-t border-gray-800 flex items-center justify-between">
        <div className="text-xs text-gray-500">
          <span className="text-emerald-400 font-mono">
            {4 - pausedModules.length}
          </span>
          {' '}modules active
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => {
              // Pause all modules
              ['ML_INFERENCE', 'STRATEGY_EXEC', 'MARKET_MAKING', 'ORDER_ROUTING'].forEach(m => {
                if (!pausedModules.includes(m as PauseModule)) {
                  toggleModulePause(m as PauseModule);
                }
              });
            }}
            className="px-4 py-2 bg-yellow-900/50 hover:bg-yellow-800 border border-yellow-700 text-yellow-400 text-xs font-bold rounded transition-all"
          >
            PAUSE ALL
          </button>
          <button
            onClick={() => {
              // Resume all modules
              pausedModules.forEach(m => toggleModulePause(m));
            }}
            disabled={pausedModules.length === 0}
            className="px-4 py-2 bg-emerald-900/50 hover:bg-emerald-800 border border-emerald-700 text-emerald-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            RESUME ALL
          </button>
        </div>
      </div>

      {/* Data Ingestion Status */}
      <div className="mt-3 pt-3 border-t border-gray-800">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Data Ingestion:</span>
          <span className="text-emerald-400 font-bold flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            ALWAYS ACTIVE
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-500">Risk Monitors:</span>
          <span className="text-emerald-400 font-bold flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            ALWAYS ACTIVE
          </span>
        </div>
      </div>
    </div>
  );
};

export default SystemPausePanel;
