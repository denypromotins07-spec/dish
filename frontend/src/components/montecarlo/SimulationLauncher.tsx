import { useState, useCallback } from 'react';
import { wsClient } from '../../core/wsClient';

interface SimulationConfig {
  pathCount: number;
  timeHorizon: number; // days
  slippageShock: number;
  feeVariation: number;
  permutationCount: number;
  includeBlackSwans: boolean;
}

interface SimulationLauncherProps {
  onSimulationStart?: (config: SimulationConfig) => void;
  onSimulationComplete?: () => void;
  isRunning?: boolean;
}

export const SimulationLauncher: React.FC<SimulationLauncherProps> = ({
  onSimulationStart,
  onSimulationComplete,
  isRunning = false,
}) => {
  const [config, setConfig] = useState<SimulationConfig>({
    pathCount: 10000,
    timeHorizon: 252, // 1 trading year
    slippageShock: 0.001,
    feeVariation: 0.0002,
    permutationCount: 1000,
    includeBlackSwans: false,
  });

  const [progress, setProgress] = useState<number>(0);
  const [estimatedTime, setEstimatedTime] = useState<string>('--');

  // Debounced config updates
  const updateConfig = useCallback(<K extends keyof SimulationConfig>(
    key: K,
    value: SimulationConfig[K]
  ) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  }, []);

  const launchSimulation = useCallback(() => {
    // Send configuration to Rust backend
    wsClient.sendBinary({
      type: 'MONTE_CARLO_START',
      payload: config,
    });

    onSimulationStart?.(config);

    // Simulate progress updates (in real app, this comes from WebSocket)
    let prog = 0;
    const interval = setInterval(() => {
      prog += Math.random() * 5;
      if (prog >= 100) {
        prog = 100;
        clearInterval(interval);
        onSimulationComplete?.();
      }
      setProgress(Math.min(prog, 100));
    }, 200);
  }, [config, onSimulationStart, onSimulationComplete]);

  const cancelSimulation = useCallback(() => {
    wsClient.sendBinary({
      type: 'MONTE_CARLO_CANCEL',
      payload: {},
    });
    setProgress(0);
  }, []);

  const presets = [
    { name: 'Quick', paths: 1000, horizon: 30 },
    { name: 'Standard', paths: 10000, horizon: 252 },
    { name: 'Deep', paths: 50000, horizon: 756 },
  ];

  return (
    <div className="p-4 bg-gray-900/80 backdrop-blur-md rounded-lg border border-fuchsia-500/20 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-fuchsia-400 font-mono text-sm uppercase tracking-wider">
          Monte Carlo Simulation Launcher
        </h3>
        {isRunning && (
          <span className="flex items-center gap-2 text-xs text-cyan-400">
            <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
            Running
          </span>
        )}
      </div>

      {/* Preset Buttons */}
      {!isRunning && (
        <div className="flex gap-2 mb-4">
          {presets.map(preset => (
            <button
              key={preset.name}
              onClick={() => {
                updateConfig('pathCount', preset.paths);
                updateConfig('timeHorizon', preset.horizon);
              }}
              className={`px-3 py-1.5 text-xs font-mono rounded border transition-all ${
                config.pathCount === preset.paths
                  ? 'border-fuchsia-500 bg-fuchsia-500/20 text-fuchsia-400'
                  : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-fuchsia-500/50'
              }`}
            >
              {preset.name}
            </button>
          ))}
        </div>
      )}

      {/* Configuration Grid */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Path Count */}
        <div className="space-y-2">
          <label className="text-xs font-mono text-gray-400">
            Simulation Paths: <span className="text-fuchsia-400">{config.pathCount.toLocaleString()}</span>
          </label>
          <input
            type="range"
            min="1000"
            max="100000"
            step="1000"
            value={config.pathCount}
            onChange={(e) => updateConfig('pathCount', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500 disabled:opacity-50"
          />
        </div>

        {/* Time Horizon */}
        <div className="space-y-2">
          <label className="text-xs font-mono text-gray-400">
            Time Horizon: <span className="text-cyan-400">{config.timeHorizon} days</span>
          </label>
          <input
            type="range"
            min="30"
            max="756"
            step="30"
            value={config.timeHorizon}
            onChange={(e) => updateConfig('timeHorizon', parseInt(e.target.value))}
            disabled={isRunning}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 disabled:opacity-50"
          />
        </div>

        {/* Slippage Shock */}
        <div className="space-y-2">
          <label className="text-xs font-mono text-gray-400">
            Slippage Shock: <span className="text-yellow-400">{(config.slippageShock * 100).toFixed(2)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="0.01"
            step="0.0001"
            value={config.slippageShock}
            onChange={(e) => updateConfig('slippageShock', parseFloat(e.target.value))}
            disabled={isRunning}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-yellow-500 disabled:opacity-50"
          />
        </div>

        {/* Fee Variation */}
        <div className="space-y-2">
          <label className="text-xs font-mono text-gray-400">
            Fee Variation: <span className="text-green-400">{(config.feeVariation * 10000).toFixed(1)} bps</span>
          </label>
          <input
            type="range"
            min="0"
            max="0.001"
            step="0.00005"
            value={config.feeVariation}
            onChange={(e) => updateConfig('feeVariation', parseFloat(e.target.value))}
            disabled={isRunning}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-green-500 disabled:opacity-50"
          />
        </div>
      </div>

      {/* Black Swan Toggle */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-800/50 rounded-lg border border-gray-700">
        <span className="text-xs font-mono text-gray-400">Include Black Swan Events</span>
        <button
          onClick={() => updateConfig('includeBlackSwans', !config.includeBlackSwans)}
          disabled={isRunning}
          className={`relative w-12 h-6 rounded-full transition-colors ${
            config.includeBlackSwans ? 'bg-red-500/50' : 'bg-gray-700'
          } disabled:opacity-50`}
        >
          <span
            className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${
              config.includeBlackSwans ? 'left-7' : 'left-1'
            }`}
          />
        </button>
      </div>

      {/* Progress Bar */}
      {isRunning && (
        <div className="mb-4 space-y-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-fuchsia-400">Progress</span>
            <span className="text-gray-400">{progress.toFixed(0)}%</span>
          </div>
          <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-fuchsia-500 to-cyan-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 font-mono text-center">
            Estimated time remaining: {estimatedTime}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-3">
        {!isRunning ? (
          <button
            onClick={launchSimulation}
            className="flex-1 px-4 py-3 bg-gradient-to-r from-fuchsia-600 to-cyan-600 hover:from-fuchsia-500 hover:to-cyan-500 text-white font-mono text-sm rounded-lg transition-all shadow-lg shadow-fuchsia-500/20"
          >
            Launch Simulation
          </button>
        ) : (
          <button
            onClick={cancelSimulation}
            className="flex-1 px-4 py-3 bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/50 font-mono text-sm rounded-lg transition-all"
          >
            Cancel Simulation
          </button>
        )}
      </div>
    </div>
  );
};
