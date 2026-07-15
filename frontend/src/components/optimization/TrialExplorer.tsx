import { useState, useMemo, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

interface OptimizationTrial {
  id: string;
  trialNumber: number;
  parameters: Record<string, number>;
  sharpe: number;
  sortino: number;
  maxDrawdown: number;
  totalReturn: number;
  status: 'completed' | 'running' | 'failed';
}

interface TrialExplorerProps {
  trials: OptimizationTrial[];
  onTrialSelect?: (trial: OptimizationTrial) => void;
}

export const TrialExplorer: React.FC<TrialExplorerProps> = ({ trials, onTrialSelect }) => {
  const [sortBy, setSortBy] = useState<keyof OptimizationTrial>('sharpe');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Filter and sort trials without causing re-renders of visible items
  const processedTrials = useMemo(() => {
    let filtered = trials;
    
    if (filterStatus !== 'all') {
      filtered = trials.filter(t => t.status === filterStatus);
    }

    return [...filtered].sort((a, b) => {
      const aVal = a[sortBy] as number;
      const bVal = b[sortBy] as number;
      return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
    });
  }, [trials, sortBy, sortOrder, filterStatus]);

  // Virtual scrolling for thousands of trials
  const parentRef = useState<HTMLDivElement>(null);
  
  const virtualizer = useVirtualizer({
    count: processedTrials.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 48,
    overscan: 5,
  });

  const handleSort = useCallback((key: keyof OptimizationTrial) => {
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400 bg-green-500/10';
      case 'running': return 'text-cyan-400 bg-cyan-500/10';
      case 'failed': return 'text-red-400 bg-red-500/10';
      default: return 'text-gray-400 bg-gray-500/10';
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-900/50 rounded-lg border border-cyan-500/20 overflow-hidden">
      {/* Header Controls */}
      <div className="flex items-center justify-between p-3 border-b border-gray-800 bg-gray-900/80">
        <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider">
          Optimization Trials ({processedTrials.length})
        </h3>
        
        <div className="flex gap-2">
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="bg-gray-800 text-gray-300 text-xs px-2 py-1 rounded border border-gray-700 focus:border-cyan-500 outline-none"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Column Headers */}
      <div className="grid grid-cols-12 gap-2 px-3 py-2 bg-gray-800/50 text-xs font-mono text-gray-400 border-b border-gray-800">
        <button 
          onClick={() => handleSort('trialNumber')}
          className="col-span-1 text-left hover:text-cyan-400 transition-colors"
        >
          # {sortBy === 'trialNumber' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button 
          onClick={() => handleSort('sharpe')}
          className="col-span-2 text-left hover:text-cyan-400 transition-colors"
        >
          Sharpe {sortBy === 'sharpe' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button 
          onClick={() => handleSort('sortino')}
          className="col-span-2 text-left hover:text-cyan-400 transition-colors"
        >
          Sortino {sortBy === 'sortino' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button 
          onClick={() => handleSort('maxDrawdown')}
          className="col-span-2 text-left hover:text-cyan-400 transition-colors"
        >
          MaxDD {sortBy === 'maxDrawdown' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <button 
          onClick={() => handleSort('totalReturn')}
          className="col-span-2 text-left hover:text-cyan-400 transition-colors"
        >
          Return {sortBy === 'totalReturn' && (sortOrder === 'desc' ? '↓' : '↑')}
        </button>
        <div className="col-span-2 text-left">Status</div>
        <div className="col-span-1 text-left">Actions</div>
      </div>

      {/* Virtualized List */}
      <div
        ref={parentRef}
        className="flex-1 overflow-auto"
        style={{ minHeight: '200px' }}
      >
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: '100%',
            position: 'relative',
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const trial = processedTrials[virtualRow.index];
            
            return (
              <div
                key={trial.id}
                data-index={virtualRow.index}
                ref={virtualizer.measureElement}
                onClick={() => onTrialSelect?.(trial)}
                className="absolute left-0 right-0 grid grid-cols-12 gap-2 px-3 py-2 border-b border-gray-800/50 hover:bg-cyan-500/5 cursor-pointer transition-colors"
                style={{
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <div className="col-span-1 text-xs font-mono text-gray-500">
                  {trial.trialNumber}
                </div>
                <div className="col-span-2 text-xs font-mono text-cyan-400">
                  {formatNumber(trial.sharpe)}
                </div>
                <div className="col-span-2 text-xs font-mono text-fuchsia-400">
                  {formatNumber(trial.sortino)}
                </div>
                <div className="col-span-2 text-xs font-mono text-red-400">
                  {formatNumber(trial.maxDrawdown * 100, 1)}%
                </div>
                <div className="col-span-2 text-xs font-mono text-green-400">
                  {formatNumber(trial.totalReturn * 100, 1)}%
                </div>
                <div className="col-span-2">
                  <span className={`text-xs px-2 py-0.5 rounded font-mono ${getStatusColor(trial.status)}`}>
                    {trial.status}
                  </span>
                </div>
                <div className="col-span-1 flex items-center gap-1">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      // View details action
                    }}
                    className="p-1 hover:bg-cyan-500/20 rounded transition-colors"
                    title="View Details"
                  >
                    <svg className="w-3 h-3 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer Stats */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-900/80 border-t border-gray-800 text-xs font-mono text-gray-500">
        <span>Total Trials: {trials.length}</span>
        <span>Best Sharpe: {formatNumber(Math.max(...trials.map(t => t.sharpe)))}</span>
        <span>Avg Sharpe: {formatNumber(trials.reduce((sum, t) => sum + t.sharpe, 0) / trials.length)}</span>
      </div>
    </div>
  );
};
