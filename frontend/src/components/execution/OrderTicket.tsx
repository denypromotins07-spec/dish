import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useExecutionStore, useOrderBookStore } from '../../core/store';

// Protobuf-like binary encoder for minimal payload
const encodeOrderPacket = (data: any): ArrayBuffer => {
  // Simplified binary encoding simulation for demonstration
  // In production, this would use actual protobuf.js or similar
  const json = JSON.stringify(data);
  const encoder = new TextEncoder();
  return encoder.encode(json).buffer;
};

type OrderType = 'MARKET' | 'LIMIT' | 'STOP';
type Side = 'BUY' | 'SELL';

export const OrderTicket: React.FC = () => {
  const [orderType, setOrderType] = useState<OrderType>('LIMIT');
  const [side, setSide] = useState<Side>('BUY');
  const [priceRef, setPriceRef] = useState<string>('0.00');
  const [qtyRef, setQtyRef] = useState<string>('0.01');
  const [leverage, setLeverage] = useState<number>(1);
  
  const sendOrder = useExecutionStore(useShallow(state => state.sendOrder));
  const bestBid = useOrderBookStore(state => state.bestBid);
  const bestAsk = useOrderBookStore(state => state.bestAsk);

  // Direct DOM refs for high-frequency updates without re-renders
  const priceInputRef = useRef<HTMLInputElement>(null);
  const qtyInputRef = useRef<HTMLInputElement>(null);
  const pnlDisplayRef = useRef<HTMLSpanElement>(null);

  // Update price input when order type changes to MARKET
  useEffect(() => {
    if (orderType === 'MARKET' && priceInputRef.current) {
      priceInputRef.current.value = 'MARKET';
      priceInputRef.current.disabled = true;
    } else if (priceInputRef.current) {
      priceInputRef.current.disabled = false;
      // Snap to mid-price on change
      const mid = (bestBid + bestAsk) / 2;
      priceInputRef.current.value = mid.toFixed(2);
    }
  }, [orderType, bestBid, bestAsk]);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    
    const price = orderType === 'MARKET' ? undefined : parseFloat(priceRef);
    const quantity = parseFloat(qtyRef);

    if (!quantity || quantity <= 0) return;

    const packet = {
      type: orderType,
      side,
      price,
      quantity,
      leverage,
      timestamp: Date.now(),
      clientId: crypto.randomUUID()
    };

    // Send binary packet directly
    const binaryData = encodeOrderPacket(packet);
    sendOrder(binaryData);

    // Reset visual feedback only
    if (pnlDisplayRef.current) {
      pnlDisplayRef.current.textContent = 'ORDER SENT';
      pnlDisplayRef.current.style.color = '#00ffcc';
      setTimeout(() => {
        if (pnlDisplayRef.current) pnlDisplayRef.current.textContent = '';
      }, 1000);
    }
  }, [orderType, side, priceRef, qtyRef, leverage, sendOrder]);

  // Optimized handler for quantity slider to avoid React state thrashing
  const handleQtyChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQtyRef(val);
    if (qtyInputRef.current) {
      qtyInputRef.current.value = val;
    }
  }, []);

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl w-full max-w-md mx-auto">
      <h3 className="text-cyan-400 font-bold text-lg mb-4 tracking-wider uppercase">Order Entry</h3>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Type Selector */}
        <div className="flex space-x-2">
          {(['MARKET', 'LIMIT', 'STOP'] as OrderType[]).map(type => (
            <button
              key={type}
              type="button"
              onClick={() => setOrderType(type)}
              className={`flex-1 py-2 text-xs font-bold rounded transition-all duration-200 ${
                orderType === type 
                  ? 'bg-cyan-500 text-black shadow-[0_0_15px_rgba(6,182,212,0.6)]' 
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {type}
            </button>
          ))}
        </div>

        {/* Side Selector */}
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setSide('BUY')}
            className={`flex-1 py-3 text-sm font-bold rounded transition-all ${
              side === 'BUY'
                ? 'bg-emerald-500 text-black shadow-[0_0_20px_rgba(16,185,129,0.5)]'
                : 'bg-gray-800 text-emerald-500/50 hover:bg-gray-700'
            }`}
          >
            BUY
          </button>
          <button
            type="button"
            onClick={() => setSide('SELL')}
            className={`flex-1 py-3 text-sm font-bold rounded transition-all ${
              side === 'SELL'
                ? 'bg-rose-500 text-black shadow-[0_0_20px_rgba(244,63,94,0.5)]'
                : 'bg-gray-800 text-rose-500/50 hover:bg-gray-700'
            }`}
          >
            SELL
          </button>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Price</label>
            <input
              ref={priceInputRef}
              type="number"
              step="0.01"
              defaultValue={priceRef}
              className="w-full bg-black/50 border border-gray-700 rounded px-3 py-2 text-white focus:outline-none focus:border-cyan-500 transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Quantity</label>
            <input
              ref={qtyInputRef}
              type="number"
              step="0.001"
              defaultValue={qtyRef}
              onChange={handleQtyChange}
              className="w-full bg-black/50 border border-gray-700 rounded px-3 py-2 text-white focus:outline-none focus:border-cyan-500 transition-colors"
            />
          </div>
        </div>

        {/* Leverage Slider */}
        <div>
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Leverage</span>
            <span className="text-cyan-400">{leverage}x</span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            value={leverage}
            onChange={(e) => setLeverage(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className={`w-full py-4 text-lg font-black uppercase tracking-widest rounded shadow-lg transition-all transform active:scale-95 ${
            side === 'BUY'
              ? 'bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-black'
              : 'bg-gradient-to-r from-rose-600 to-rose-500 hover:from-rose-500 hover:to-rose-400 text-black'
          }`}
        >
          {side} {orderType}
        </button>

        {/* Status Display */}
        <div className="text-center h-4">
          <span ref={pnlDisplayRef} className="text-xs font-mono text-gray-500"></span>
        </div>
      </form>
    </div>
  );
};

export default OrderTicket;
