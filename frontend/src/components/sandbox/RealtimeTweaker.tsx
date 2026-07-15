import { useState, useCallback, useEffect } from 'react';
import { wsClient } from '../../core/wsClient';

interface StrategyParams {
  [key: string]: number | boolean | string;
}

interface RealtimeTweakerProps {
  strategyId: string;
  currentParams: StrategyParams;
  onParamsChanged?: (params: StrategyParams) => void;
}

export const RealtimeTweaker: React.FC<RealtimeTweakerProps> = ({
  strategyId,
  currentParams,
  onParamsChanged,
}) => {
  const [localParams, setLocalParams] = useState<StrategyParams>(currentParams);
  const [pendingChanges, setPendingChanges] = useState<Set<string>>(new Set());
  const [lastApplied, setLastApplied] = useState<number>(0);

  // Debounce parameter updates (500ms)
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (pendingChanges.size > 0) {
        applyChanges();
      }
    }, 500);
    return () => clearTimeout(timeout);
  }, [localParams, pendingChanges]);

  const applyChanges = useCallback(() => {
    wsClient.sendBinary({
      type: 'STRATEGY_PARAMS_UPDATE',
      payload: { strategyId, params: localParams, timestamp: Date.now() },
    });
    setPendingChanges(new Set());
    setLastApplied(Date.now());
    onParamsChanged?.(localParams);
  }, [strategyId, localParams, onParamsChanged]);

  const updateParam = useCallback(<K extends keyof StrategyParams>(
    key: K,
    value: StrategyParams[K]
  ) => {
    setLocalParams(prev => ({ ...prev, [key]: value }));
    setPendingChanges(prev => new Set(prev).add(key as string));
  }, []);

  const resetToDefaults = useCallback(() => {
    setLocalParams(currentParams);
    setPendingChanges(new Set());
  }, [currentParams]);

  const getParamType = (value: number | boolean | string): 'number' | 'boolean' => {
    return typeof value === 'boolean' ? 'boolean' : 'number';
  };

  return (
    <div className="p-4 bg-gray-900/80 backdrop-blur-md rounded-lg border border-cyan-500/20 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider">
          Strategy Parameter Tweaker
        </h3>
        <div className="flex gap-2">
          {pendingChanges.size > 0 && (
            <span className="text-xs text-yellow-400 flex items-center gap-1">
              <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
              Pending: {pendingChanges.size}
            </span>
          )}
          <button
            onClick={resetToDefaults}
            className="text-xs px-2 py-1 text-gray-400 hover:text-cyan-400 transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {Object.entries(localParams).map(([key, value]) => {
          const type = getParamType(value);
          const isPending = pendingChanges.has(key);

          if (type === 'boolean') {
            return (
              <div key={key} className="flex items-center justify-between p-2 bg-gray-800/50 rounded">
                <span className={`text-xs font-mono ${isPending ? 'text-yellow-400' : 'text-gray-400'}`}>
                  {key}
                </span>
                <button
                  onClick={() => updateParam(key, !value as boolean)}
                  className={`relative w-10 h-5 rounded-full transition-colors ${
                    value ? 'bg-cyan-500/50' : 'bg-gray-700'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      value ? 'left-5' : 'left-0.5'
                    }`}
                  />
                </button>
              </div>
            );
          }

          return (
            <div key={key} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className={`font-mono ${isPending ? 'text-yellow-400' : 'text-gray-400'}`}>
                  {key}
                </span>
                <span className="text-cyan-400 font-mono">{String(value)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={Number(value)}
                onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          );
        })}
      </div>

      {lastApplied > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-800 text-xs text-gray-500 font-mono text-center">
          Last applied: {new Date(lastApplied).toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};
