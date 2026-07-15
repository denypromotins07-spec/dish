import React, { useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useStrategyStore } from '../../core/store';

type StrategyId = 'STATARB' | 'MARKETMAKING' | 'LATENCYARB' | 'SMC';

interface StrategyCardProps {
  id: StrategyId;
  name: string;
  description: string;
  allocatedCapital: number;
  dailyPnl: number;
  isActive: boolean;
  onToggle: (id: StrategyId) => void;
  onReallocate: (id: StrategyId, newAllocation: number) => void;
}

const StrategyCard = React.memo(({ 
  id, 
  name, 
  description, 
  allocatedCapital, 
  dailyPnl, 
  isActive, 
  onToggle,
  onReallocate 
}: StrategyCardProps) => {
  const pnlColor = dailyPnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const pnlBg = dailyPnl >= 0 ? 'bg-emerald-900/30' : 'bg-rose-900/30';

  return (
    <div className={`relative p-4 rounded-lg border transition-all duration-200 ${
      isActive 
        ? 'bg-gray-800/80 border-cyan-700 shadow-[0_0_20px_rgba(6,182,212,0.2)]' 
        : 'bg-gray-900/50 border-gray-800 opacity-70'
    }`}>
      {/* Toggle Switch */}
      <button
        onClick={() => onToggle(id)}
        className={`absolute top-3 right-3 w-10 h-5 rounded-full transition-colors ${
          isActive ? 'bg-cyan-600' : 'bg-gray-700'
        }`}
      >
        <div className={`w-4 h-4 bg-white rounded-full transform transition-transform ${
          isActive ? 'translate-x-5' : 'translate-x-0.5'
        }`} />
      </button>

      {/* Header */}
      <div className="mb-3">
        <h4 className={`font-bold text-sm ${isActive ? 'text-white' : 'text-gray-500'}`}>
          {name}
        </h4>
        <p className="text-xs text-gray-500 mt-1">{description}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div>
          <div className="text-xs text-gray-500">Allocated</div>
          <div className="text-sm font-mono text-white">${allocatedCapital.toLocaleString()}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Daily PnL</div>
          <div className={`text-sm font-mono font-bold ${pnlColor} ${pnlBg} px-2 py-0.5 rounded inline-block`}>
            {dailyPnl >= 0 ? '+' : ''}{dailyPnl.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Quick Reallocate Buttons */}
      {isActive && (
        <div className="flex space-x-1">
          {[0.5, 1, 2].map((mult) => (
            <button
              key={mult}
              onClick={() => onReallocate(id, allocatedCapital * mult)}
              className="flex-1 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
            >
              {mult === 0.5 ? '½' : mult === 1 ? '1x' : '2x'}
            </button>
          ))}
        </div>
      )}

      {/* Status Indicator */}
      <div className={`absolute bottom-3 right-3 w-2 h-2 rounded-full ${
        isActive ? 'bg-cyan-500 animate-pulse' : 'bg-gray-600'
      }`} />
    </div>
  );
}, (prev, next) => {
  return prev.dailyPnl === next.dailyPnl && 
         prev.isActive === next.isActive &&
         prev.allocatedCapital === next.allocatedCapital;
});

export const StrategyToggleGrid: React.FC = () => {
  const strategies = useStrategyStore(useShallow(state => state.strategies));
  const toggleStrategy = useStrategyStore(useShallow(state => state.toggleStrategy));
  const reallocateCapital = useStrategyStore(useShallow(state => state.reallocateCapital));

  const totalAllocated = useMemo(() => 
    strategies.reduce((acc, s) => acc + s.allocatedCapital, 0),
    [strategies]
  );

  const totalDailyPnl = useMemo(() => 
    strategies.reduce((acc, s) => acc + s.dailyPnl, 0),
    [strategies]
  );

  const handleReallocate = useCallback((id: StrategyId, newAllocation: number) => {
    reallocateCapital(id, newAllocation);
  }, [reallocateCapital]);

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-400 font-bold text-lg tracking-wider uppercase flex items-center">
          <span className="mr-2">🧠</span>
          Alpha Strategies
        </h3>
        <div className="text-right">
          <div className="text-xs text-gray-500">Total Allocated</div>
          <div className="text-sm font-mono font-bold text-white">${totalAllocated.toLocaleString()}</div>
        </div>
      </div>

      {/* Summary Bar */}
      <div className={`mb-4 p-3 rounded-lg ${
        totalDailyPnl >= 0 ? 'bg-emerald-900/20 border border-emerald-800' : 'bg-rose-900/20 border border-rose-800'
      }`}>
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-400">Combined Daily PnL</span>
          <span className={`text-lg font-mono font-bold ${totalDailyPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {totalDailyPnl >= 0 ? '+' : ''}${totalDailyPnl.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Strategy Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {strategies.map((strategy) => (
          <StrategyCard
            key={strategy.id}
            id={strategy.id as StrategyId}
            name={strategy.name}
            description={strategy.description}
            allocatedCapital={strategy.allocatedCapital}
            dailyPnl={strategy.dailyPnl}
            isActive={strategy.isActive}
            onToggle={toggleStrategy}
            onReallocate={handleReallocate}
          />
        ))}
      </div>

      {/* Risk Parity Note */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Risk Parity Engine:</span>
          <span className="text-cyan-400 font-bold flex items-center">
            <span className="w-2 h-2 bg-cyan-500 rounded-full mr-2 animate-pulse"></span>
            ACTIVE
          </span>
        </div>
        <p className="text-xs text-gray-600 mt-1">
          Capital automatically rebalanced based on Sharpe ratio & volatility targeting
        </p>
      </div>
    </div>
  );
};

export default StrategyToggleGrid;
