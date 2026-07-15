import { useMemo } from 'react';

interface RuinMetrics {
  riskOfRuin: number;
  probabilityMaxDrawdown: number;
  confidenceInterval95: [number, number];
  confidenceInterval99: [number, number];
  expectedReturn: number;
  volatility: number;
  kellyFraction: number;
  maxConsecutiveLosses: number;
}

interface RuinMetricsPanelProps {
  metrics: RuinMetrics;
  simulationCount?: number;
}

export const RuinMetricsPanel: React.FC<RuinMetricsPanelProps> = ({
  metrics,
  simulationCount = 10000,
}) => {
  // Memoized calculations to prevent re-renders
  const formattedMetrics = useMemo(() => ({
    riskOfRuin: (metrics.riskOfRuin * 100).toFixed(2),
    probMaxDD: (metrics.probabilityMaxDrawdown * 100).toFixed(2),
    ci95Lower: (metrics.confidenceInterval95[0] * 100).toFixed(2),
    ci95Upper: (metrics.confidenceInterval95[1] * 100).toFixed(2),
    ci99Lower: (metrics.confidenceInterval99[0] * 100).toFixed(2),
    ci99Upper: (metrics.confidenceInterval99[1] * 100).toFixed(2),
    expectedReturn: (metrics.expectedReturn * 100).toFixed(2),
    volatility: (metrics.volatility * 100).toFixed(2),
    kellyFraction: (metrics.kellyFraction * 100).toFixed(2),
    maxConsecLosses: metrics.maxConsecutiveLosses,
  }), [metrics]);

  const getRiskColor = (risk: number) => {
    if (risk < 5) return 'text-green-400 bg-green-500/10 border-green-500/30';
    if (risk < 15) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
    if (risk < 30) return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
    return 'text-red-400 bg-red-500/10 border-red-500/30';
  };

  const getRiskLabel = (risk: number) => {
    if (risk < 5) return 'Low Risk';
    if (risk < 15) return 'Moderate';
    if (risk < 30) return 'Elevated';
    return 'Critical';
  };

  const riskLevel = parseFloat(formattedMetrics.riskOfRuin);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 bg-gray-900/80 backdrop-blur-md rounded-lg border border-cyan-500/20 shadow-lg">
      {/* Risk of Ruin - Main Metric */}
      <div className={`col-span-2 p-4 rounded-lg border ${getRiskColor(riskLevel)} transition-all duration-300`}>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-mono uppercase tracking-wider opacity-80">Risk of Ruin</h4>
          <span className="text-xs px-2 py-0.5 rounded-full border border-current opacity-70">
            {getRiskLabel(riskLevel)}
          </span>
        </div>
        <div className="text-3xl font-mono font-bold">{formattedMetrics.riskOfRuin}%</div>
        <div className="mt-2 h-2 bg-current/20 rounded-full overflow-hidden">
          <div
            className="h-full bg-current transition-all duration-500"
            style={{ width: `${Math.min(riskLevel, 100)}%` }}
          />
        </div>
        <div className="mt-1 text-xs opacity-60">
          Based on {simulationCount.toLocaleString()} simulations
        </div>
      </div>

      {/* Probability of Max Drawdown */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Prob. Max DD
        </div>
        <div className="text-xl font-mono text-fuchsia-400">{formattedMetrics.probMaxDD}%</div>
      </div>

      {/* Kelly Fraction */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Kelly Fraction
        </div>
        <div className="text-xl font-mono text-cyan-400">{formattedMetrics.kellyFraction}%</div>
      </div>

      {/* 95% Confidence Interval */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          95% CI Returns
        </div>
        <div className="text-sm font-mono text-green-400">
          {formattedMetrics.ci95Lower}% → {formattedMetrics.ci95Upper}%
        </div>
      </div>

      {/* 99% Confidence Interval */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          99% CI Returns
        </div>
        <div className="text-sm font-mono text-blue-400">
          {formattedMetrics.ci99Lower}% → {formattedMetrics.ci99Upper}%
        </div>
      </div>

      {/* Expected Return */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Exp. Return
        </div>
        <div className="text-xl font-mono text-green-400">{formattedMetrics.expectedReturn}%</div>
      </div>

      {/* Volatility */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Volatility
        </div>
        <div className="text-xl font-mono text-yellow-400">{formattedMetrics.volatility}%</div>
      </div>

      {/* Max Consecutive Losses */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Max Consec. Losses
        </div>
        <div className="text-xl font-mono text-red-400">{formattedMetrics.maxConsecLosses}</div>
      </div>

      {/* Sharpe Proxy (Return/Volatility) */}
      <div className="p-3 rounded-lg bg-gray-800/50 border border-gray-700/50">
        <div className="text-xs font-mono text-gray-500 uppercase tracking-wider mb-1">
          Return/Risk Ratio
        </div>
        <div className="text-xl font-mono text-purple-400">
          {(metrics.expectedReturn / metrics.volatility).toFixed(2)}
        </div>
      </div>
    </div>
  );
};
