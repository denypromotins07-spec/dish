import React, { useCallback, useMemo, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useOrderStore } from '../../core/store';

interface OrderRowProps {
  orderId: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'LIMIT' | 'STOP' | 'MARKET';
  price: number;
  quantity: number;
  filled: number;
  status: 'PENDING' | 'PARTIAL' | 'WORKING';
  timeInForce: string;
}

const OrderRow = React.memo(({ 
  orderId, 
  symbol, 
  side, 
  type, 
  price, 
  quantity, 
  filled,
  status,
  timeInForce 
}: OrderRowProps) => {
  const cancelOrder = useOrderStore(useShallow(state => state.cancelOrder));
  const modifyOrder = useOrderStore(useShallow(state => state.modifyOrder));
  
  const fillPercent = (filled / quantity) * 100;
  const isBuy = side === 'BUY';

  const handleCancel = useCallback(() => {
    cancelOrder(orderId);
  }, [orderId, cancelOrder]);

  return (
    <div className="flex items-center justify-between py-3 px-4 border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors group">
      {/* Left Section: Symbol & Side */}
      <div className="flex items-center space-x-3 flex-1">
        <span className={`w-1 h-6 rounded-sm ${isBuy ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
        <div>
          <div className="font-bold text-white text-sm">{symbol}</div>
          <div className={`text-xs px-2 py-0.5 rounded inline-block ${
            isBuy ? 'bg-emerald-900/30 text-emerald-400' : 'bg-rose-900/30 text-rose-400'
          }`}>{side}</div>
        </div>
      </div>

      {/* Middle Section: Order Details */}
      <div className="flex items-center space-x-6 flex-2 justify-center">
        <div className="text-right">
          <div className="text-xs text-gray-500">Type</div>
          <div className="text-sm font-mono text-gray-300">{type}</div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">Price</div>
          <div className="text-sm font-mono text-cyan-400">{price.toFixed(2)}</div>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">Qty</div>
          <div className="text-sm font-mono text-gray-300">{quantity.toFixed(4)}</div>
        </div>
        <div className="text-right w-24">
          <div className="text-xs text-gray-500">Filled</div>
          <div className="flex items-center space-x-2">
            <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className={`h-full ${fillPercent >= 100 ? 'bg-emerald-500' : 'bg-yellow-500'}`}
                style={{ width: `${fillPercent}%` }}
              />
            </div>
            <span className="text-xs font-mono text-gray-400 w-12">
              {filled.toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Right Section: Status & Actions */}
      <div className="flex items-center space-x-3 flex-1 justify-end">
        <div className={`text-xs px-2 py-1 rounded font-bold ${
          status === 'WORKING' ? 'bg-cyan-900/30 text-cyan-400 animate-pulse' :
          status === 'PARTIAL' ? 'bg-yellow-900/30 text-yellow-400' :
          'bg-gray-800 text-gray-400'
        }`}>
          {status}
        </div>
        <div className="text-xs text-gray-600 font-mono">{timeInForce}</div>
        
        {/* Action Buttons (visible on hover) */}
        <div className="opacity-0 group-hover:opacity-100 transition-opacity flex space-x-1">
          <button
            onClick={handleCancel}
            className="px-2 py-1 text-xs bg-rose-900/50 text-rose-400 border border-rose-700 rounded hover:bg-rose-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => modifyOrder(orderId)}
            className="px-2 py-1 text-xs bg-cyan-900/50 text-cyan-400 border border-cyan-700 rounded hover:bg-cyan-800 transition-colors"
          >
            Modify
          </button>
        </div>
      </div>
    </div>
  );
}, (prev, next) => {
  return prev.filled === next.filled && prev.status === next.status;
});

export const WorkingOrdersList: React.FC = () => {
  const orders = useOrderStore(useShallow(state => state.workingOrders));
  const cancelAllOrders = useOrderStore(useShallow(state => state.cancelAllOrders));
  
  const containerRef = useRef<HTMLDivElement>(null);
  const [isCanceling, setIsCanceling] = useState(false);

  // Virtualization
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 20 });
  const ROW_HEIGHT = 72;

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    
    const scrollTop = containerRef.current.scrollTop;
    const visibleHeight = containerRef.current.clientHeight;
    const totalRows = orders.length;
    
    const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 3);
    const end = Math.min(totalRows, Math.ceil((scrollTop + visibleHeight) / ROW_HEIGHT) + 3);
    
    setVisibleRange({ start, end });
  }, [orders.length]);

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      handleScroll();
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll, orders.length]);

  const visibleOrders = useMemo(() => {
    return orders.slice(visibleRange.start, visibleRange.end);
  }, [orders, visibleRange]);

  const handleCancelAll = useCallback(async () => {
    setIsCanceling(true);
    await cancelAllOrders();
    setTimeout(() => setIsCanceling(false), 1000);
  }, [cancelAllOrders]);

  const totalHeight = orders.length * ROW_HEIGHT;

  // Group by symbol for summary
  const ordersBySymbol = useMemo(() => {
    return orders.reduce((acc, order) => {
      acc[order.symbol] = (acc[order.symbol] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
  }, [orders]);

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg shadow-xl flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h3 className="text-yellow-400 font-bold text-lg tracking-wider uppercase">Working Orders</h3>
        <div className="flex items-center space-x-3">
          <span className="text-xs text-gray-500">
            {orders.length} active orders
            {Object.keys(ordersBySymbol).length > 0 && (
              <span className="ml-2 text-gray-600">
                across {Object.keys(ordersBySymbol).length} symbols
              </span>
            )}
          </span>
          <button
            onClick={handleCancelAll}
            disabled={orders.length === 0 || isCanceling}
            className="px-3 py-1 text-xs font-bold bg-rose-900/50 text-rose-400 border border-rose-700 rounded hover:bg-rose-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isCanceling ? 'CANCELING...' : 'CANCEL ALL'}
          </button>
        </div>
      </div>

      {/* Column Headers */}
      <div className="px-4 py-2 bg-gray-900/50 border-b border-gray-800 text-xs text-gray-500 uppercase tracking-wider">
        <div className="flex items-center justify-between">
          <div className="flex-1">Order</div>
          <div className="flex-2 flex justify-center">
            <div className="w-16 text-right">Type</div>
            <div className="w-20 text-right">Price</div>
            <div className="w-20 text-right">Quantity</div>
            <div className="w-24 text-right">Filled</div>
          </div>
          <div className="flex-1 flex justify-end">
            <div className="w-20 text-right">Status</div>
            <div className="w-16 text-right">TIF</div>
            <div className="w-24 text-right">Actions</div>
          </div>
        </div>
      </div>

      {/* Order List (Virtualized) */}
      <div ref={containerRef} className="flex-1 overflow-auto" style={{ maxHeight: '300px' }}>
        <div style={{ height: `${totalHeight}px`, position: 'relative' }}>
          {visibleOrders.map((order, index) => (
            <div
              key={order.orderId}
              style={{
                position: 'absolute',
                top: (visibleRange.start + index) * ROW_HEIGHT,
                left: 0,
                right: 0,
                height: ROW_HEIGHT
              }}
            >
              <OrderRow {...order} />
            </div>
          ))}
        </div>
      </div>

      {/* Empty State */}
      {orders.length === 0 && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">📋</div>
            <div>No working orders</div>
            <div className="text-xs mt-1">Place a limit or stop order to see it here</div>
          </div>
        </div>
      )}

      {/* Footer Summary */}
      {orders.length > 0 && (
        <div className="px-4 py-2 border-t border-gray-800 bg-gray-900/50">
          <div className="flex justify-between text-xs text-gray-500">
            <span>Total Value: ${orders.reduce((a, b) => a + (b.price * b.quantity), 0).toFixed(2)}</span>
            <span>Buys: {orders.filter(o => o.side === 'BUY').length} | Sells: {orders.filter(o => o.side === 'SELL').length}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default WorkingOrdersList;
