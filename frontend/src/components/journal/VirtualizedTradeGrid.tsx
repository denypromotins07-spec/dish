import { useState, useMemo, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

interface Trade {
  id: string;
  timestamp: number;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl?: number;
  fees: number;
  strategy: string;
  tags: string[];
}

interface VirtualizedTradeGridProps {
  trades: Trade[];
  onTradeSelect?: (trade: Trade) => void;
}

export const VirtualizedTradeGrid: React.FC<VirtualizedTradeGridProps> = ({
  trades,
  onTradeSelect,
}) => {
  const [filterSide, setFilterSide] = useState<string>('all');
  const [filterStrategy, setFilterStrategy] = useState<string>('all');
  const [sortBy, setSortBy] = useState<keyof Trade>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Memoized filtering and sorting
  const filteredTrades = useMemo(() => {
    let result = [...trades];

    if (filterSide !== 'all') {
      result = result.filter(t => t.side === filterSide);
    }

    if (filterStrategy !== 'all') {
      result = result.filter(t => t.strategy === filterStrategy);
    }

    result.sort((a, b) => {
      const aVal = a[sortBy];
      const bVal = b[sortBy];
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
      }
      return sortOrder === 'desc' 
        ? String(bVal).localeCompare(String(aVal))
        : String(aVal).localeCompare(String(bVal));
    });

    return result;
  }, [trades, filterSide, filterStrategy, sortBy, sortOrder]);

  // Get unique strategies for filter dropdown
  const strategies = useMemo(() => {
    return Array.from(new Set(trades.map(t => t.strategy)));
  }, [trades]);

  // Virtual scrolling
  const parentRef = useState<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: filteredTrades.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 40,
    overscan: 5,
  });

  const handleSort = useCallback((key: keyof Trade) => {
    if (sortBy === key) {
      setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(key);
      setSortOrder('desc');
    }
  }, [sortBy]);

  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  const formatTime = (ts: number) => {
    return new Date(ts).toLocaleTimeString();
  };

  // Calculate aggregate stats
  const stats = useMemo(() => {
    const totalPnl = filteredTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const totalFees = filteredTrades.reduce((sum, t) => sum + t.fees, 0);
    const wins = filteredTrades.filter(t => (t.pnl || 0) > 0).length;
    const winRate = filteredTrades.length > 0 ? wins / filteredTrades.length : 0;

    return { totalPnl, totalFees, winRate, count: filteredTrades.length };
  }, [filteredTrades]);

  return (
    <div className="flex flex-col h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-gray-800 bg-gray-900/80">
        <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider">
          Trade Journal ({stats.count})
        </h3>
        
        <div className="flex gap-2">
          <select
            value={filterSide}
            onChange={(e) => setFilterSide(e.target.value)}
            className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700 focus:border-cyan-500 outline-none"
          >
            <option value="all">All Sides</option>
            <option value="buy">Long</option>
            <option value="sell">Short</option>
          </select>

          <select
            value={filterStrategy}
            onChange={(e) => setFilterStrategy(e.target.value)}
            className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700 focus:border-cyan-500 outline-none"
          >
            <option value="all">All Strategies</option>
            {strategies.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Column Headers */}
      <div className="grid grid-cols-12 gap-2 px-3 py-2 bg-gray-800/50 text-xs font-mono text-gray-400 border-b border-gray-800">
        <button onClick={() => handleSort('timestamp')} className="col-span-2 text-left hover:text-cyan-400">
          Time {sortBy === 'timestamp' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <div className="col-span-1">Symbol</div>
        <div className="col-span-1">Side</div>
        <button onClick={() => handleSort('quantity')} className="col-span-1 text-left hover:text-cyan-400">
          Qty {sortBy === 'quantity' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button onClick={() => handleSort('price')} className="col-span-1 text-left hover:text-cyan-400">
          Price {sortBy === 'price' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button onClick={() => handleSort('pnl')} className="col-span-2 text-left hover:text-cyan-400">
          PnL {sortBy === 'pnl' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <div className="col-span-2">Strategy</div>
        <div className="col-span-2">Tags</div>
      </div>

      {/* Virtualized List */}
      <div ref={parentRef} className="flex-1 overflow-auto" style={{ minHeight: '200px' }}>
        <div style={{ height: `${virtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const trade = filteredTrades[virtualRow.index];
            const pnlColor = trade.pnl 
              ? trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
              : 'text-gray-500';

            return (
              <div
                key={trade.id}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
                onClick={() => onTradeSelect?.(trade)}
                className="absolute left-0 right-0 grid grid-cols-12 gap-2 px-3 py-2 border-b border-gray-800/50 hover:bg-cyan-500/5 cursor-pointer transition-colors"
                style={{ transform: `translateY(${virtualRow.start}px)` }}
              >
                <div className="col-span-2 text-xs font-mono text-gray-500">
                  {formatTime(trade.timestamp)}
                </div>
                <div className="col-span-1 text-xs font-mono text-cyan-400">{trade.symbol}</div>
                <div className={`col-span-1 text-xs font-mono ${trade.side === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                  {trade.side.toUpperCase()}
                </div>
                <div className="col-span-1 text-xs font-mono text-gray-400">{formatNumber(trade.quantity, 4)}</div>
                <div className="col-span-1 text-xs font-mono text-gray-400">{formatNumber(trade.price)}</div>
                <div className={`col-span-2 text-xs font-mono ${pnlColor}`}>
                  {trade.pnl !== undefined ? formatNumber(trade.pnl) : '--'}
                </div>
                <div className="col-span-2 text-xs font-mono text-fuchsia-400">{trade.strategy}</div>
                <div className="col-span-2 flex gap-1 flex-wrap">
                  {trade.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="text-xs px-1.5 py-0.5 bg-gray-800 text-gray-400 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900/80 border-t border-gray-800 text-xs font-mono">
        <span className="text-gray-500">Total Trades: {stats.count}</span>
        <span className={stats.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}>
          Net PnL: {formatNumber(stats.totalPnl)}
        </span>
        <span className="text-yellow-400">Fees: {formatNumber(stats.totalFees)}</span>
        <span className={stats.winRate >= 0.5 ? 'text-green-400' : 'text-yellow-400'}>
          Win Rate: {(stats.winRate * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};
