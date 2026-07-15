import React, { useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { usePositionStore, useExecutionStore } from '../../core/store';

interface PositionActionsProps {
  positionId: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
}

export const PositionActions: React.FC<PositionActionsProps> = ({
  positionId,
  symbol,
  side,
  quantity,
  entryPrice,
  currentPrice
}) => {
  const closePosition = usePositionStore(useShallow(state => state.closePosition));
  const adjustPosition = usePositionStore(useShallow(state => state.adjustPosition));
  const sendOrder = useExecutionStore(useShallow(state => state.sendOrder));

  // Calculate PnL
  const diff = currentPrice - entryPrice;
  const unrealizedPnl = side === 'LONG' ? diff * quantity : -diff * quantity;
  const pnlPercent = ((currentPrice - entryPrice) / entryPrice) * 100;

  // Scale out (close 50%)
  const handleScaleOut = useCallback(() => {
    const closeQty = quantity * 0.5;
    closePosition(positionId, closeQty);
  }, [positionId, quantity, closePosition]);

  // Set trailing stop
  const handleTrailingStop = useCallback(() => {
    const trailDistance = Math.abs(currentPrice - entryPrice) * 0.5; // 50% of profit
    if (trailDistance > 0) {
      adjustPosition(positionId, {
        trailingStop: true,
        trailDistance
      });
    }
  }, [positionId, currentPrice, entryPrice, adjustPosition]);

  // Hedge with delta-neutral offset
  const handleHedge = useCallback(() => {
    const hedgeSide = side === 'LONG' ? 'SELL' : 'BUY';
    const packet = {
      type: 'MARKET' as const,
      side: hedgeSide,
      quantity: quantity,
      timestamp: Date.now(),
      clientId: crypto.randomUUID(),
      hedgeOf: positionId
    };
    
    const encoder = new TextEncoder();
    const binaryData = encoder.encode(JSON.stringify(packet)).buffer;
    sendOrder(binaryData);
  }, [positionId, side, quantity, sendOrder]);

  // Close all
  const handleCloseAll = useCallback(() => {
    closePosition(positionId, quantity);
  }, [positionId, quantity, closePosition]);

  // Take profit (close 25%)
  const handleTakeProfit = useCallback(() => {
    const tpQty = quantity * 0.25;
    closePosition(positionId, tpQty);
  }, [positionId, quantity, closePosition]);

  return (
    <div className="bg-gray-900/90 backdrop-blur-md border border-gray-700 rounded-lg p-4 shadow-2xl">
      {/* Position Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-3">
          <span className={`w-2 h-2 rounded-full ${side === 'LONG' ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
          <span className="font-bold text-white text-lg">{symbol}</span>
          <span className={`text-xs px-2 py-1 rounded font-bold ${
            side === 'LONG' ? 'bg-emerald-900/50 text-emerald-400' : 'bg-rose-900/50 text-rose-400'
          }`}>{side}</span>
        </div>
        <div className={`text-right ${unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          <div className="text-sm font-mono font-bold">
            {unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnl.toFixed(2)} USD
          </div>
          <div className="text-xs font-mono">
            {pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Action Buttons Grid */}
      <div className="grid grid-cols-2 gap-2">
        {/* Scale Out */}
        <button
          onClick={handleScaleOut}
          className="px-3 py-2 bg-cyan-900/30 hover:bg-cyan-800/50 border border-cyan-700 text-cyan-400 rounded text-xs font-bold transition-all hover:shadow-[0_0_15px_rgba(6,182,212,0.4)]"
        >
          📊 Scale Out 50%
        </button>

        {/* Take Profit */}
        <button
          onClick={handleTakeProfit}
          className="px-3 py-2 bg-emerald-900/30 hover:bg-emerald-800/50 border border-emerald-700 text-emerald-400 rounded text-xs font-bold transition-all hover:shadow-[0_0_15px_rgba(16,185,129,0.4)]"
        >
          💰 Take Profit 25%
        </button>

        {/* Trailing Stop */}
        <button
          onClick={handleTrailingStop}
          disabled={unrealizedPnl <= 0}
          className="px-3 py-2 bg-yellow-900/30 hover:bg-yellow-800/50 border border-yellow-700 text-yellow-400 rounded text-xs font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-[0_0_15px_rgba(234,179,8,0.4)]"
        >
          🎯 Trailing Stop
        </button>

        {/* Hedge */}
        <button
          onClick={handleHedge}
          className="px-3 py-2 bg-purple-900/30 hover:bg-purple-800/50 border border-purple-700 text-purple-400 rounded text-xs font-bold transition-all hover:shadow-[0_0_15px_rgba(168,85,247,0.4)]"
        >
          🛡️ Delta Hedge
        </button>
      </div>

      {/* Danger Zone */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <button
          onClick={handleCloseAll}
          className="w-full px-3 py-2 bg-rose-900/50 hover:bg-rose-800 border border-rose-700 text-rose-400 hover:text-rose-300 rounded text-xs font-bold transition-all hover:shadow-[0_0_20px_rgba(244,63,94,0.5)]"
        >
          ⚠️ CLOSE ENTIRE POSITION
        </button>
      </div>

      {/* Quick Stats */}
      <div className="mt-3 flex justify-between text-xs text-gray-500">
        <span>Entry: {entryPrice.toFixed(2)}</span>
        <span>Mark: {currentPrice.toFixed(2)}</span>
        <span>Size: {quantity.toFixed(4)}</span>
      </div>
    </div>
  );
};

export default PositionActions;
