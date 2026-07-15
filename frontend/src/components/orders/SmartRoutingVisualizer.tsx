import React, { useEffect, useRef, useMemo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useOrderStore } from '../../core/store';

interface RouteSlice {
  venue: string;
  quantity: number;
  filled: number;
  avgPrice: number;
  status: 'PENDING' | 'FILLING' | 'FILLED';
}

interface ParentOrder {
  id: string;
  totalQuantity: number;
  slices: RouteSlice[];
}

export const SmartRoutingVisualizer: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null);
  const activeOrders = useOrderStore(useShallow(state => state.activeParentOrders));

  // Animation frame ref for smooth updates
  const animationFrameRef = useRef<number>();

  // Simulated animation progress (in production, this would be real-time data)
  const [animationProgress, setAnimationProgress] = useState(0);

  useEffect(() => {
    const animate = () => {
      setAnimationProgress(prev => (prev + 0.01) % 1);
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animationFrameRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  // Venue configuration with colors and positions
  const venues = useMemo(() => ({
    'BINANCE': { color: '#fcd34d', x: 80, y: 60 },
    'BYBIT': { color: '#fb923c', x: 200, y: 40 },
    'OKX': { color: '#22d3ee', x: 320, y: 60 },
    'COINBASE': { color: '#60a5fa', x: 140, y: 140 },
    'KRAKEN': { color: '#a78bfa', x: 260, y: 140 }
  }), []);

  // Get the most recent parent order for visualization
  const currentOrder = activeOrders[activeOrders.length - 1];

  if (!currentOrder) {
    return (
      <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl h-48 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <div className="text-3xl mb-2">🔀</div>
          <div className="text-sm">No active SOR orders</div>
          <div className="text-xs mt-1">Large orders will appear here when routed</div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/80 backdrop-blur-md border border-gray-800 rounded-lg p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-cyan-400 font-bold text-sm tracking-wider uppercase flex items-center">
          <span className="mr-2">🔀</span>
          Smart Order Routing
        </h3>
        <div className="text-xs text-gray-500">
          Order #{currentOrder.id.slice(0, 8)}
        </div>
      </div>

      {/* SVG Visualization */}
      <svg 
        ref={svgRef}
        viewBox="0 0 400 180" 
        className="w-full h-auto"
      >
        {/* Definitions for gradients and glows */}
        <defs>
          <radialGradient id="centerGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Center Hub (Parent Order) */}
        <circle cx="200" cy="90" r="25" fill="url(#centerGlow)" />
        <circle 
          cx="200" 
          cy="90" 
          r="15" 
          fill="#0e7490" 
          stroke="#22d3ee" 
          strokeWidth="2"
          filter="url(#glow)"
        />
        <text x="200" y="94" textAnchor="middle" fill="white" fontSize="10" fontWeight="bold">
          {currentOrder.totalQuantity}
        </text>
        <text x="200" y="106" textAnchor="middle" fill="#94a3b8" fontSize="8">
          TOTAL
        </text>

        {/* Venue Nodes */}
        {Object.entries(venues).map(([name, config]) => {
          const slice = currentOrder.slices.find(s => s.venue === name);
          const fillPercent = slice ? slice.filled / slice.quantity : 0;
          
          return (
            <g key={name}>
              {/* Connection Line */}
              <line
                x1="200"
                y1="90"
                x2={config.x}
                y2={config.y}
                stroke="#334155"
                strokeWidth="2"
                strokeDasharray="4 2"
              />
              
              {/* Animated flow particles */}
              {slice && slice.status !== 'PENDING' && (
                <circle r="3" fill={config.color}>
                  <animateMotion
                    dur={`${2 / (animationProgress + 0.5)}s`}
                    repeatCount="indefinite"
                    path={`M200,90 L${config.x},${config.y}`}
                  />
                </circle>
              )}
              
              {/* Venue Circle */}
              <circle
                cx={config.x}
                cy={config.y}
                r="18"
                fill="#1e293b"
                stroke={config.color}
                strokeWidth="2"
                filter="url(#glow)"
              />
              
              {/* Fill Progress Arc */}
              {fillPercent > 0 && (
                <circle
                  cx={config.x}
                  cy={config.y}
                  r="18"
                  fill="none"
                  stroke={config.color}
                  strokeWidth="3"
                  strokeDasharray={`${fillPercent * 113} 113`}
                  strokeLinecap="round"
                  transform={`rotate(-90 ${config.x} ${config.y})`}
                  opacity="0.6"
                />
              )}
              
              {/* Venue Label */}
              <text 
                x={config.x} 
                y={config.y + 4} 
                textAnchor="middle" 
                fill={config.color} 
                fontSize="7" 
                fontWeight="bold"
              >
                {name.slice(0, 6)}
              </text>
              
              {/* Fill Percentage */}
              <text 
                x={config.x} 
                y={config.y + 30} 
                textAnchor="middle" 
                fill="#94a3b8" 
                fontSize="8"
              >
                {Math.round(fillPercent * 100)}%
              </text>
            </g>
          );
        })}

        {/* Status Legend */}
        <g transform="translate(10, 155)">
          <rect width="380" height="20" fill="#111827" rx="4" />
          <text x="10" y="14" fill="#94a3b8" fontSize="8">Status:</text>
          
          {['PENDING', 'FILLING', 'FILLED'].map((status, i) => (
            <g key={status} transform={`translate(${80 + i * 100}, 0)`}>
              <circle cx="5" cy="5" r="4" fill={
                status === 'PENDING' ? '#6b7280' :
                status === 'FILLING' ? '#fbbf24' : '#10b981'
              } />
              <text x="14" y="8" fill="#94a3b8" fontSize="8">{status}</text>
            </g>
          ))}
        </g>
      </svg>

      {/* Order Details */}
      <div className="mt-3 grid grid-cols-5 gap-2">
        {Object.entries(venues).map(([name, config]) => {
          const slice = currentOrder.slices.find(s => s.venue === name);
          if (!slice) return null;
          
          return (
            <div 
              key={name}
              className="bg-gray-800/50 rounded p-2 text-center border border-gray-700"
              style={{ borderColor: config.color }}
            >
              <div className="text-xs font-bold" style={{ color: config.color }}>
                {name}
              </div>
              <div className="text-xs text-gray-400 mt-1">
                {slice.filled.toFixed(2)} / {slice.quantity.toFixed(2)}
              </div>
              <div className="text-xs font-mono text-gray-500">
                @ {slice.avgPrice.toFixed(2)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Routing Stats */}
      <div className="mt-3 pt-3 border-t border-gray-800 flex justify-between text-xs">
        <span className="text-gray-500">
          Venues: {currentOrder.slices.length}
        </span>
        <span className="text-gray-500">
          Avg Slippage: 0.02%
        </span>
        <span className="text-emerald-400 font-bold">
          SOR Active
        </span>
      </div>
    </div>
  );
};

export default SmartRoutingVisualizer;
