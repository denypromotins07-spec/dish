import React, { useEffect, useRef, useCallback, useState } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useOrderBookStore, useExecutionStore } from '../../core/store';

interface ManualOrder {
  id: string;
  price: number;
  quantity: number;
  side: 'BID' | 'ASK';
  x: number;
  y: number;
}

export const ManualOrderBook: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const [orders, setOrders] = useState<ManualOrder[]>([]);
  const [draggingOrderId, setDraggingOrderId] = useState<string | null>(null);
  const [selectedSide, setSelectedSide] = useState<'BID' | 'ASK'>('BID');
  const [orderSize, setOrderSize] = useState<number>(0.1);

  const bestBid = useOrderBookStore(useShallow(state => state.bestBid));
  const bestAsk = useOrderBookStore(useShallow(state => state.bestAsk));
  const cancelOrder = useExecutionStore(useShallow(state => state.cancelOrder));

  // Price range for visualization
  const priceRange = (bestAsk - bestBid) * 3;
  const midPrice = (bestBid + bestAsk) / 2;

  // Convert price to Y coordinate
  const priceToY = useCallback((price: number, height: number) => {
    const ratio = (price - (midPrice - priceRange / 2)) / priceRange;
    return height * (1 - ratio);
  }, [midPrice, priceRange]);

  // Convert Y coordinate to price
  const yToPrice = useCallback((y: number, height: number) => {
    const ratio = 1 - (y / height);
    return (midPrice - priceRange / 2) + (ratio * priceRange);
  }, [midPrice, priceRange]);

  // Handle canvas click to place order
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const price = yToPrice(y, canvas.height);
    
    const newOrder: ManualOrder = {
      id: crypto.randomUUID(),
      price,
      quantity: orderSize,
      side: selectedSide,
      x,
      y
    };

    setOrders(prev => [...prev, newOrder]);
    
    // Send to backend
    const packet = {
      type: 'LIMIT' as const,
      side: selectedSide === 'BID' ? 'BUY' : 'SELL' as const,
      price,
      quantity: orderSize,
      timestamp: Date.now(),
      clientId: newOrder.id,
      manual: true
    };

    const encoder = new TextEncoder();
    const binaryData = encoder.encode(JSON.stringify(packet)).buffer;
    // In production: sendOrder(binaryData);
  }, [selectedSide, orderSize, yToPrice]);

  // Handle drag start
  const handleMouseDown = useCallback((e: React.MouseEvent, orderId: string) => {
    e.stopPropagation();
    setDraggingOrderId(orderId);
  }, []);

  // Handle drag move
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingOrderId) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const newPrice = yToPrice(y, canvas.height);

    setOrders(prev => prev.map(order => 
      order.id === draggingOrderId 
        ? { ...order, price: newPrice, y }
        : order
    ));
  }, [draggingOrderId, yToPrice]);

  // Handle drag end
  const handleMouseUp = useCallback(() => {
    if (draggingOrderId) {
      // Update order on backend with new price
      const order = orders.find(o => o.id === draggingOrderId);
      if (order) {
        // Cancel old order and place new one
        cancelOrder(draggingOrderId);
        
        const packet = {
          type: 'LIMIT' as const,
          side: order.side === 'BID' ? 'BUY' : 'SELL' as const,
          price: order.price,
          quantity: order.quantity,
          timestamp: Date.now(),
          clientId: order.id,
          manual: true,
          modify: true
        };

        const encoder = new TextEncoder();
        const binaryData = encoder.encode(JSON.stringify(packet)).buffer;
        // In production: sendOrder(binaryData);
      }
    }
    setDraggingOrderId(null);
  }, [draggingOrderId, orders, cancelOrder]);

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Draw gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, rect.height);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.05)'); // Cyan at top (high prices)
    gradient.addColorStop(0.5, 'rgba(17, 24, 39, 0.8)'); // Dark in middle
    gradient.addColorStop(1, 'rgba(236, 72, 153, 0.05)'); // Magenta at bottom (low prices)
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw mid-price line
    const midY = priceToY(midPrice, rect.height);
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(rect.width, midY);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw bid/ask reference lines
    const bidY = priceToY(bestBid, rect.height);
    const askY = priceToY(bestAsk, rect.height);
    
    ctx.beginPath();
    ctx.moveTo(0, bidY);
    ctx.lineTo(rect.width, bidY);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, askY);
    ctx.lineTo(rect.width, askY);
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw manual orders
    orders.forEach(order => {
      const y = priceToY(order.price, rect.height);
      const isBid = order.side === 'BID';
      
      // Glow effect
      ctx.shadowColor = isBid ? '#10b981' : '#f43f5e';
      ctx.shadowBlur = 15;

      // Order line
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.strokeStyle = isBid ? 'rgba(16, 185, 129, 0.8)' : 'rgba(244, 63, 94, 0.8)';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Order handle (circle for dragging)
      ctx.beginPath();
      ctx.arc(30, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = isBid ? '#10b981' : '#f43f5e';
      ctx.fill();

      // Quantity label
      ctx.shadowBlur = 0;
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.fillText(`${order.quantity.toFixed(3)} @ ${order.price.toFixed(2)}`, 50, y + 4);

      // Drag hint
      if (draggingOrderId === order.id) {
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#fbbf24';
        ctx.fillText('DRAGGING...', 50, y - 10);
      }
    });

  }, [orders, bestBid, bestAsk, midPrice, priceToY, draggingOrderId]);

  return (
    <div ref={containerRef} className="relative w-full h-96 bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg overflow-hidden">
      {/* Control Panel */}
      <div className="absolute top-0 left-0 right-0 z-10 p-3 bg-gray-900/90 backdrop-blur-sm border-b border-gray-800">
        <div className="flex items-center justify-between">
          <h3 className="text-cyan-400 font-bold text-sm uppercase">Manual Order Placement</h3>
          
          <div className="flex items-center space-x-3">
            {/* Side Selector */}
            <div className="flex space-x-1">
              <button
                onClick={() => setSelectedSide('BID')}
                className={`px-3 py-1 text-xs font-bold rounded ${
                  selectedSide === 'BID'
                    ? 'bg-emerald-500 text-black'
                    : 'bg-gray-800 text-emerald-500/50'
                }`}
              >
                BID
              </button>
              <button
                onClick={() => setSelectedSide('ASK')}
                className={`px-3 py-1 text-xs font-bold rounded ${
                  selectedSide === 'ASK'
                    ? 'bg-rose-500 text-black'
                    : 'bg-gray-800 text-rose-500/50'
                }`}
              >
                ASK
              </button>
            </div>

            {/* Size Input */}
            <div className="flex items-center space-x-2">
              <label className="text-xs text-gray-500">Size:</label>
              <input
                type="number"
                value={orderSize}
                onChange={(e) => setOrderSize(parseFloat(e.target.value) || 0)}
                step="0.01"
                className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white font-mono"
              />
            </div>

            {/* Clear All */}
            <button
              onClick={() => setOrders([])}
              className="px-3 py-1 text-xs bg-red-900/50 text-red-400 border border-red-700 rounded hover:bg-red-800 transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-crosshair"
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Instructions Overlay */}
      {orders.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center text-gray-500 text-sm">
            <div className="text-4xl mb-2">📍</div>
            <div>Click anywhere to place a {selectedSide} order</div>
            <div className="text-xs mt-1">Drag orders to adjust price</div>
          </div>
        </div>
      )}

      {/* Price Scale */}
      <div className="absolute right-0 top-12 bottom-0 w-16 bg-gray-900/50 border-l border-gray-800 pointer-events-none">
        <div className="h-full flex flex-col justify-between py-2 text-xs text-gray-500 font-mono">
          <span>{(midPrice + priceRange / 2).toFixed(2)}</span>
          <span>{midPrice.toFixed(2)}</span>
          <span>{(midPrice - priceRange / 2).toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
};

export default ManualOrderBook;
