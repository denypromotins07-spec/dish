import React, { useEffect, useRef, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { usePositionStore } from '../../core/store';

interface MarginGaugeProps {
  positionId?: string;
}

export const MarginGauge: React.FC<MarginGaugeProps> = ({ positionId }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Get margin data from store
  const positions = usePositionStore(useShallow(state => state.positions));
  const accountEquity = usePositionStore(useShallow(state => state.accountEquity));
  const totalMarginUsed = usePositionStore(useShallow(state => state.totalMarginUsed));
  const maintenanceMargin = usePositionStore(useShallow(state => state.maintenanceMargin));

  // Calculate metrics
  const marginRatio = totalMarginUsed / accountEquity;
  const maintenanceRatio = maintenanceMargin / accountEquity;
  const freeMargin = accountEquity - totalMarginUsed;
  const liquidationRisk = marginRatio > 0.8 ? 'HIGH' : marginRatio > 0.5 ? 'MEDIUM' : 'LOW';

  // Color interpolation based on risk
  const getRiskColor = (ratio: number) => {
    if (ratio < 0.5) return { r: 6, g: 182, b: 212 }; // Cyan
    if (ratio < 0.7) return { r: 234, g: 179, b: 8 }; // Yellow
    if (ratio < 0.85) return { r: 249, g: 115, b: 22 }; // Orange
    return { r: 244, g: 63, b: 94 }; // Rose/Red
  };

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const radius = Math.min(centerX, centerY) - 10;
    const lineWidth = 12;

    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);

    // Background ring (gray)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = '#1f2937'; // gray-800
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Calculate angles (start from -90 degrees, go clockwise)
    const startAngle = -Math.PI / 2;
    const endAngle = startAngle + (marginRatio * Math.PI * 2);

    // Gradient for the active ring
    const riskColor = getRiskColor(marginRatio);
    const gradient = ctx.createLinearGradient(0, 0, rect.width, rect.height);
    gradient.addColorStop(0, `rgb(${riskColor.r}, ${riskColor.g}, ${riskColor.b})`);
    gradient.addColorStop(1, `rgb(${Math.min(255, riskColor.r + 50)}, ${Math.min(255, riskColor.g + 50)}, ${Math.min(255, riskColor.b + 50)})`);

    // Active margin ring
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    
    // Glow effect
    ctx.shadowColor = `rgb(${riskColor.r}, ${riskColor.g}, ${riskColor.b})`;
    ctx.shadowBlur = marginRatio > 0.8 ? 20 : 10;
    ctx.stroke();
    
    // Reset shadow
    ctx.shadowBlur = 0;

    // Maintenance margin warning ring (inner)
    if (maintenanceRatio > 0) {
      const maintEndAngle = startAngle + (maintenanceRatio * Math.PI * 2);
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius - lineWidth - 4, startAngle, maintEndAngle);
      ctx.strokeStyle = 'rgba(234, 179, 8, 0.6)'; // Yellow transparent
      ctx.lineWidth = 4;
      ctx.stroke();
    }

    // Center text
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    // Margin ratio percentage
    ctx.font = 'bold 24px monospace';
    ctx.fillStyle = `rgb(${riskColor.r}, ${riskColor.g}, ${riskColor.b})`;
    ctx.fillText(`${(marginRatio * 100).toFixed(1)}%`, centerX, centerY - 10);

    // Label
    ctx.font = '12px sans-serif';
    ctx.fillStyle = '#9ca3af'; // gray-400
    ctx.fillText('Margin Used', centerX, centerY + 15);

    // Liquidation risk indicator
    if (liquidationRisk === 'HIGH') {
      ctx.font = 'bold 10px sans-serif';
      ctx.fillStyle = '#f43f5e';
      ctx.fillText('⚠ LIQ RISK', centerX, centerY + 30);
    }

  }, [marginRatio, maintenanceRatio, liquidationRisk, accountEquity]);

  return (
    <div ref={containerRef} className="relative w-48 h-48 mx-auto">
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* Overlay stats */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs px-4 py-2 bg-gray-900/80 rounded-b-lg backdrop-blur-sm">
        <div>
          <div className="text-gray-500">Free Margin</div>
          <div className="text-emerald-400 font-mono font-bold">${freeMargin.toFixed(2)}</div>
        </div>
        <div className="text-right">
          <div className="text-gray-500">Risk Level</div>
          <div className={`font-bold ${
            liquidationRisk === 'HIGH' ? 'text-rose-400 animate-pulse' :
            liquidationRisk === 'MEDIUM' ? 'text-yellow-400' :
            'text-cyan-400'
          }`}>{liquidationRisk}</div>
        </div>
      </div>
    </div>
  );
};

export default MarginGauge;
