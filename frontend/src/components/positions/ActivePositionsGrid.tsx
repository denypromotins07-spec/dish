import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { usePositionStore, useMarketStore } from '../../core/store';

interface PositionRowProps {
  positionId: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  quantity: number;
  currentPrice: number;
  unrealizedPnl: number;
  leverage: number;
  marginUsed: number;
  liquidationPrice: number;
}

// Memoized row component to prevent unnecessary re-renders
const PositionRow = React.memo(({ 
  positionId, 
  symbol, 
  side, 
  entryPrice, 
  quantity, 
  currentPrice, 
  unrealizedPnl,
  leverage,
  marginUsed,
  liquidationPrice
}: PositionRowProps) => {
  // Direct DOM refs for PnL updates without React re-renders
  const pnlCellRef = useRef<HTMLTableCellElement>(null);
  const priceCellRef = useRef<HTMLTableCellElement>(null);

  // Update PnL color based on value
  useEffect(() => {
    if (pnlCellRef.current) {
      pnlCellRef.current.textContent = `${unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}`;
      pnlCellRef.current.className = `font-mono font-bold ${
        unrealizedPnl > 0 ? 'text-emerald-400' : 
        unrealizedPnl < 0 ? 'text-rose-400' : 'text-gray-400'
      }`;
    }
  }, [unrealizedPnl]);

  // Update current price display
  useEffect(() => {
    if (priceCellRef.current) {
      priceCellRef.current.textContent = currentPrice.toFixed(2);
    }
  }, [currentPrice]);

  const pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100 * leverage;
  const isLong = side === 'LONG';
  const pnlColor = isLong 
    ? (currentPrice >= entryPrice ? 'text-emerald-400' : 'text-rose-400')
    : (currentPrice <= entryPrice ? 'text-emerald-400' : 'text-rose-400');

  return (
    <tr className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors group">
      <td className="py-3 px-4">
        <div className="flex items-center space-x-2">
          <span className={`w-1 h-4 rounded-sm ${isLong ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
          <span className="font-bold text-white">{symbol}</span>
          <span className={`text-xs px-2 py-0.5 rounded ${
            isLong ? 'bg-emerald-900/30 text-emerald-400' : 'bg-rose-900/30 text-rose-400'
          }`}>{side}</span>
        </div>
      </td>
      <td className="py-3 px-4 font-mono text-gray-300">{quantity.toFixed(4)}</td>
      <td className="py-3 px-4 font-mono text-gray-400">{entryPrice.toFixed(2)}</td>
      <td ref={priceCellRef} className="py-3 px-4 font-mono text-cyan-400">0.00</td>
      <td className="py-3 px-4 font-mono text-gray-400">{leverage}x</td>
      <td className="py-3 px-4 font-mono text-gray-400">${marginUsed.toFixed(2)}</td>
      <td ref={pnlCellRef} className="py-3 px-4 font-mono font-bold">0.00</td>
      <td className="py-3 px-4 font-mono text-gray-500 text-xs">{liquidationPrice.toFixed(2)}</td>
      <td className="py-3 px-4">
        <button className="opacity-0 group-hover:opacity-100 px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded transition-all">
          Manage
        </button>
      </td>
    </tr>
  );
}, (prev, next) => {
  // Custom comparison to only re-render when essential data changes
  return prev.unrealizedPnl === next.unrealizedPnl && 
         prev.currentPrice === next.currentPrice &&
         prev.quantity === next.quantity;
});

export const ActivePositionsGrid: React.FC = () => {
  const positions = usePositionStore(useShallow(state => state.positions));
  const prices = useMarketStore(useShallow(state => state.prices));
  
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLTableSectionElement>(null);

  // Virtualization: Only render visible rows + buffer
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 });
  const ROW_HEIGHT = 52; // pixels

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    const scrollTop = containerRef.current.scrollTop;
    const visibleHeight = containerRef.current.clientHeight;
    const totalRows = positions.length;
    
    const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 5);
    const end = Math.min(
      totalRows,
      Math.ceil((scrollTop + visibleHeight) / ROW_HEIGHT) + 5
    );
    
    setVisibleRange({ start, end });
  }, [positions.length]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      handleScroll(); // Initial calculation
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll, positions.length]);

  // Calculate total height for scrollbar
  const totalHeight = positions.length * ROW_HEIGHT;

  // Get visible positions
  const visiblePositions = useMemo(() => {
    return positions.slice(visibleRange.start, visibleRange.end).map(pos => {
      const currentPrice = prices[pos.symbol] || pos.entryPrice;
      const diff = currentPrice - pos.entryPrice;
      const unrealizedPnl = pos.side === 'LONG' 
        ? diff * pos.quantity 
        : -diff * pos.quantity;
      
      return {
        ...pos,
        currentPrice,
        unrealizedPnl
      };
    });
  }, [positions, prices, visibleRange]);

  // Total PnL calculation
  const totalPnl = useMemo(() => {
    return positions.reduce((acc, pos) => {
      const currentPrice = prices[pos.symbol] || pos.entryPrice;
      const diff = currentPrice - pos.entryPrice;
      const pnl = pos.side === 'LONG' ? diff * pos.quantity : -diff * pos.quantity;
      return acc + pnl;
    }, 0);
  }, [positions, prices]);

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg shadow-xl flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h3 className="text-emerald-400 font-bold text-lg tracking-wider uppercase">Active Positions</h3>
        <div className={`text-sm font-mono font-bold ${totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          Total PnL: {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)} USD
        </div>
      </div>

      {/* Table Container with Virtualization */}
      <div ref={containerRef} className="flex-1 overflow-auto" style={{ maxHeight: '400px' }}>
        <table className="w-full min-w-[800px]">
          <thead className="sticky top-0 bg-gray-900 z-10">
            <tr className="text-xs text-gray-500 uppercase tracking-wider border-b border-gray-800">
              <th className="py-2 px-4 text-left">Symbol</th>
              <th className="py-2 px-4 text-right">Size</th>
              <th className="py-2 px-4 text-right">Entry</th>
              <th className="py-2 px-4 text-right">Mark</th>
              <th className="py-2 px-4 text-right">Lev</th>
              <th className="py-2 px-4 text-right">Margin</th>
              <th className="py-2 px-4 text-right">Unrealized PnL</th>
              <th className="py-2 px-4 text-right">Liq. Price</th>
              <th className="py-2 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody ref={scrollRef} style={{ height: `${totalHeight}px`, position: 'relative' }}>
            {visiblePositions.map((pos, index) => (
              <div
                key={pos.positionId}
                style={{
                  position: 'absolute',
                  top: (visibleRange.start + index) * ROW_HEIGHT,
                  width: '100%',
                  height: ROW_HEIGHT
                }}
              >
                <PositionRow {...pos} />
              </div>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer Stats */}
      <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50">
        <div className="flex justify-between text-xs text-gray-500">
          <span>{positions.length} Active Positions</span>
          <span>Margin Used: ${positions.reduce((a, b) => a + b.marginUsed, 0).toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

export default ActivePositionsGrid;
