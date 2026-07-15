import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useBrainStore } from '../../core/store';

interface SignalItem {
  id: string;
  timestamp: number;
  regime: string;
  factors: string[];
  confidence: number;
}

export const AlphaSignalFeed: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const signals = useBrainStore(useShallow(state => state.signals));
  const currentRegime = useBrainStore(useShallow(state => state.currentRegime));
  const modelConfidence = useBrainStore(useShallow(state => state.modelConfidence));

  // Virtualization state
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 15 });
  const ITEM_HEIGHT = 64; // pixels per signal item

  // Handle scroll for virtualization
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    const scrollTop = containerRef.current.scrollTop;
    const visibleHeight = containerRef.current.clientHeight;
    const totalItems = signals.length;
    
    const start = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - 3);
    const end = Math.min(
      totalItems,
      Math.ceil((scrollTop + visibleHeight) / ITEM_HEIGHT) + 3
    );
    
    setVisibleRange({ start, end });
  }, [signals.length]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      handleScroll();
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll, signals.length]);

  // Get visible signals
  const visibleSignals = useMemo(() => {
    return signals.slice(visibleRange.start, visibleRange.end);
  }, [signals, visibleRange]);

  // Regime color mapping
  const getRegimeColor = (regime: string) => {
    if (regime.includes('High Volatility')) return 'text-rose-400 bg-rose-900/20 border-rose-800';
    if (regime.includes('Mean Reversion')) return 'text-cyan-400 bg-cyan-900/20 border-cyan-800';
    if (regime.includes('Trending')) return 'text-emerald-400 bg-emerald-900/20 border-emerald-800';
    if (regime.includes('Sideways')) return 'text-yellow-400 bg-yellow-900/20 border-yellow-800';
    return 'text-gray-400 bg-gray-900/20 border-gray-800';
  };

  // Confidence level indicator
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-emerald-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-rose-400';
  };

  const totalHeight = signals.length * ITEM_HEIGHT;

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg shadow-xl flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h3 className="text-magenta-400 font-bold text-lg tracking-wider uppercase flex items-center">
          <span className="mr-2">🧠</span>
          Alpha Signal Feed
        </h3>
        <div className="flex items-center space-x-4">
          {/* Current Regime Badge */}
          <div className={`px-3 py-1 rounded border text-xs font-bold ${getRegimeColor(currentRegime)}`}>
            {currentRegime}
          </div>
          {/* Model Confidence */}
          <div className="text-right">
            <div className="text-xs text-gray-500">ML Confidence</div>
            <div className={`text-sm font-mono font-bold ${getConfidenceColor(modelConfidence)}`}>
              {(modelConfidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Signal List (Virtualized) */}
      <div 
        ref={containerRef} 
        className="flex-1 overflow-auto"
        style={{ maxHeight: '300px' }}
      >
        <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
          {visibleSignals.map((signal, index) => (
            <div
              key={signal.id}
              style={{
                position: 'absolute',
                top: (visibleRange.start + index) * ITEM_HEIGHT,
                left: 0,
                right: 0,
                height: ITEM_HEIGHT
              }}
              className="px-4 py-2 border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
            >
              <div className="flex items-center justify-between">
                {/* Timestamp & ID */}
                <div className="flex items-center space-x-3">
                  <span className="text-xs text-gray-600 font-mono">
                    {new Date(signal.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="text-xs text-gray-700 font-mono">#{signal.id.slice(0, 6)}</span>
                </div>

                {/* Confidence Meter */}
                <div className="flex items-center space-x-2">
                  <div className="w-20 h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        signal.confidence >= 0.8 ? 'bg-emerald-500' :
                        signal.confidence >= 0.6 ? 'bg-yellow-500' :
                        'bg-rose-500'
                      }`}
                      style={{ width: `${signal.confidence * 100}%` }}
                    />
                  </div>
                  <span className={`text-xs font-mono font-bold ${getConfidenceColor(signal.confidence)}`}>
                    {(signal.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Factors */}
              <div className="mt-2 flex flex-wrap gap-1">
                {signal.factors.slice(0, 5).map((factor, i) => (
                  <span 
                    key={i}
                    className="px-2 py-0.5 text-xs bg-gray-800 text-gray-400 rounded border border-gray-700"
                  >
                    {factor}
                  </span>
                ))}
                {signal.factors.length > 5 && (
                  <span className="px-2 py-0.5 text-xs text-gray-600">+{signal.factors.length - 5} more</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="flex justify-between text-xs text-gray-500">
          <span>{signals.length} signals processed</span>
          <span className="flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2 animate-pulse"></span>
            Real-time
          </span>
        </div>
      </div>

      {/* Active Factors Legend */}
      <div className="px-4 py-3 border-t border-gray-800">
        <h4 className="text-xs text-gray-500 uppercase font-bold mb-2">Active Alpha Factors</h4>
        <div className="flex flex-wrap gap-2">
          {[
            { name: 'Order Flow Imbalance', strength: 'HIGH' },
            { name: 'Liquidity Gradient', strength: 'MED' },
            { name: 'Volatility Surface', strength: 'LOW' },
            { name: 'Cross-Exchange Arb', strength: 'HIGH' },
            { name: 'Momentum Decay', strength: 'MED' }
          ].map((factor) => (
            <div 
              key={factor.name}
              className={`px-3 py-1 rounded text-xs border ${
                factor.strength === 'HIGH' ? 'bg-emerald-900/20 border-emerald-800 text-emerald-400' :
                factor.strength === 'MED' ? 'bg-yellow-900/20 border-yellow-800 text-yellow-400' :
                'bg-gray-800 border-gray-700 text-gray-400'
              }`}
            >
              {factor.name}
              <span className="ml-2 opacity-60">[{factor.strength}]</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AlphaSignalFeed;
