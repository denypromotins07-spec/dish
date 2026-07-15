import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useOrderStore } from '../../core/store';

interface Fill {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  fee: number;
  venue: string;
  // TCA metrics
  arrivalPrice: number;
  slippageBps: number;
  feeImpact: number;
}

export const OrderHistoryFeed: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const fills = useOrderStore(useShallow(state => state.fills));
  
  // Virtualization state
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 25 });
  const ITEM_HEIGHT = 56; // pixels per fill item

  // Handle scroll for virtualization
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    const scrollTop = containerRef.current.scrollTop;
    const visibleHeight = containerRef.current.clientHeight;
    const totalItems = fills.length;
    
    const start = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - 3);
    const end = Math.min(totalItems, Math.ceil((scrollTop + visibleHeight) / ITEM_HEIGHT) + 3);
    
    setVisibleRange({ start, end });
  }, [fills.length]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      handleScroll();
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll, fills.length]);

  // Get visible fills
  const visibleFills = useMemo(() => {
    return fills.slice(visibleRange.start, visibleRange.end);
  }, [fills, visibleRange]);

  const totalHeight = fills.length * ITEM_HEIGHT;

  // Calculate aggregate TCA metrics
  const tcaMetrics = useMemo(() => {
    if (fills.length === 0) return null;
    
    const totalSlippage = fills.reduce((acc, f) => acc + f.slippageBps, 0);
    const totalFees = fills.reduce((acc, f) => acc + f.feeImpact, 0);
    const avgSlippage = totalSlippage / fills.length;
    const avgFeeImpact = totalFees / fills.length;
    
    return {
      totalFills: fills.length,
      avgSlippageBps: avgSlippage.toFixed(2),
      avgFeeImpactBps: avgFeeImpact.toFixed(2),
      totalVolume: fills.reduce((acc, f) => acc + (f.price * f.quantity), 0)
    };
  }, [fills]);

  // Slippage color coding
  const getSlippageColor = (bps: number) => {
    if (bps <= 1) return 'text-emerald-400 bg-emerald-900/20';
    if (bps <= 3) return 'text-yellow-400 bg-yellow-900/20';
    return 'text-rose-400 bg-rose-900/20';
  };

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg shadow-xl flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h3 className="text-green-400 font-bold text-lg tracking-wider uppercase flex items-center">
          <span className="mr-2">✅</span>
          Order History & TCA
        </h3>
        {tcaMetrics && (
          <div className="text-xs text-gray-500">
            {tcaMetrics.totalFills} fills | ${tcaMetrics.totalVolume.toLocaleString()} vol
          </div>
        )}
      </div>

      {/* TCA Summary Bar */}
      {tcaMetrics && (
        <div className="px-4 py-2 bg-gray-800/30 border-b border-gray-800 grid grid-cols-3 gap-4">
          <div>
            <div className="text-xs text-gray-500">Avg Slippage</div>
            <div className={`text-sm font-mono font-bold ${
              parseFloat(tcaMetrics.avgSlippageBps) <= 1 ? 'text-emerald-400' :
              parseFloat(tcaMetrics.avgSlippageBps) <= 3 ? 'text-yellow-400' :
              'text-rose-400'
            }`}>
              {tcaMetrics.avgSlippageBps} bps
            </div>
          </div>
          <div>
            <div className="text-xs text-gray-500">Avg Fee Impact</div>
            <div className="text-sm font-mono font-bold text-cyan-400">
              {tcaMetrics.avgFeeImpactBps} bps
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-500">Execution Quality</div>
            <div className="text-sm font-bold text-emerald-400">
              {parseFloat(tcaMetrics.avgSlippageBps) <= 1 ? 'EXCELLENT' :
               parseFloat(tcaMetrics.avgSlippageBps) <= 3 ? 'GOOD' : 'NEEDS IMPROVEMENT'}
            </div>
          </div>
        </div>
      )}

      {/* Column Headers */}
      <div className="px-4 py-2 bg-gray-900/50 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wider">
        <div className="grid grid-cols-12 gap-2">
          <div className="col-span-2">Time</div>
          <div className="col-span-2">Symbol</div>
          <div className="col-span-1">Side</div>
          <div className="col-span-2 text-right">Qty</div>
          <div className="col-span-2 text-right">Price</div>
          <div className="col-span-2 text-right">Slippage</div>
          <div className="col-span-1 text-right">Fee</div>
        </div>
      </div>

      {/* Fill List (Virtualized) */}
      <div 
        ref={containerRef} 
        className="flex-1 overflow-auto"
        style={{ maxHeight: '300px' }}
      >
        <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
          {visibleFills.map((fill, index) => (
            <div
              key={fill.id}
              style={{
                position: 'absolute',
                top: (visibleRange.start + index) * ITEM_HEIGHT,
                left: 0,
                right: 0,
                height: ITEM_HEIGHT
              }}
              className="px-4 py-2 border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
            >
              <div className="grid grid-cols-12 gap-2 items-center">
                {/* Time */}
                <div className="col-span-2 text-xs text-gray-500 font-mono">
                  {new Date(fill.timestamp).toLocaleTimeString()}
                </div>
                
                {/* Symbol */}
                <div className="col-span-2 font-bold text-white text-sm">
                  {fill.symbol}
                </div>
                
                {/* Side */}
                <div className="col-span-1">
                  <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                    fill.side === 'BUY' 
                      ? 'bg-emerald-900/30 text-emerald-400' 
                      : 'bg-rose-900/30 text-rose-400'
                  }`}>
                    {fill.side}
                  </span>
                </div>
                
                {/* Quantity */}
                <div className="col-span-2 text-right text-sm font-mono text-gray-300">
                  {fill.quantity.toFixed(4)}
                </div>
                
                {/* Price */}
                <div className="col-span-2 text-right text-sm font-mono text-cyan-400">
                  {fill.price.toFixed(2)}
                </div>
                
                {/* Slippage (TCA) */}
                <div className="col-span-2 text-right">
                  <span className={`text-xs px-2 py-0.5 rounded font-mono font-bold ${getSlippageColor(fill.slippageBps)}`}>
                    {fill.slippageBps.toFixed(1)} bps
                  </span>
                </div>
                
                {/* Fee Impact */}
                <div className="col-span-1 text-right text-xs font-mono text-gray-500">
                  {fill.feeImpact.toFixed(1)}
                </div>
              </div>
              
              {/* Expanded Details (on hover) */}
              <div className="hidden group-hover:block mt-2 pt-2 border-t border-gray-800/50">
                <div className="grid grid-cols-4 gap-2 text-xs text-gray-500">
                  <div>Venue: <span className="text-gray-300">{fill.venue}</span></div>
                  <div>Arrival: <span className="text-gray-300 font-mono">{fill.arrivalPrice.toFixed(2)}</span></div>
                  <div>Exec ID: <span className="text-gray-300 font-mono">{fill.id.slice(0, 8)}</span></div>
                  <div className="text-right">Fee: <span className="text-gray-300">${fill.fee.toFixed(4)}</span></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Empty State */}
      {fills.length === 0 && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">📜</div>
            <div>No order history</div>
            <div className="text-xs mt-1">Filled orders will appear here with TCA metrics</div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Session: {fills.length} fills</span>
          <span className="flex items-center">
            <span className="w-2 h-2 bg-emerald-500 rounded-full mr-2"></span>
            Real-time TCA enabled
          </span>
        </div>
      </div>
    </div>
  );
};

export default OrderHistoryFeed;
