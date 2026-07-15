import { useEffect, useRef, useCallback } from 'react';

interface TradePoint {
  slippage: number; // bps
  marketImpact: number; // bps
  implementationShortfall: number; // bps
  toxicity: number; // 0-1
}

interface TCAScatterPlotProps {
  trades: TradePoint[];
}

export const TCAScatterPlot: React.FC<TCAScatterPlotProps> = ({ trades }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 60, bottom: 50, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear with fade effect
    ctx.fillStyle = 'rgba(2, 2, 5, 0.2)';
    ctx.fillRect(0, 0, width, height);

    if (trades.length === 0) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }

    // Find min/max
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    trades.forEach(t => {
      minX = Math.min(minX, t.slippage, t.marketImpact);
      maxX = Math.max(maxX, t.slippage, t.marketImpact);
      minY = Math.min(minY, t.implementationShortfall);
      maxY = Math.max(maxY, t.implementationShortfall);
    });

    const xRange = maxX - minX || 1;
    const yRange = maxY - minY || 1;

    const xToPixel = (val: number) => padding.left + ((val - minX) / xRange) * chartWidth;
    const yToPixel = (val: number) => padding.top + chartHeight - ((val - minY) / yRange) * chartHeight;

    // Draw grid
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      const x = padding.left + (chartWidth / 5) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw scatter points with color based on toxicity
    trades.forEach(trade => {
      const x = xToPixel(trade.slippage);
      const y = yToPixel(trade.implementationShortfall);
      
      // Size based on toxicity
      const size = 3 + trade.toxicity * 8;
      
      // Color gradient: green (low toxicity) -> yellow -> red (high toxicity)
      let r, g, b;
      if (trade.toxicity < 0.33) {
        r = Math.floor(255 * (trade.toxicity / 0.33));
        g = 200;
        b = 100;
      } else if (trade.toxicity < 0.66) {
        r = 255;
        g = Math.floor(200 * (1 - (trade.toxicity - 0.33) / 0.33));
        b = Math.floor(100 * ((trade.toxicity - 0.33) / 0.33));
      } else {
        r = 255;
        g = Math.floor(100 * (1 - (trade.toxicity - 0.66) / 0.34));
        b = 50;
      }

      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.6)`;
      ctx.fill();
    });

    // Draw axes
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // X-axis label
    ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Slippage (bps)', width / 2, height - 15);

    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Implementation Shortfall (bps)', 0, 0);
    ctx.restore();

    // Axis values
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 5; i++) {
      const val = minY + (yRange / 5) * (5 - i);
      const y = padding.top + (chartHeight / 5) * i;
      ctx.fillText(val.toFixed(1), padding.left - 8, y);
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let i = 0; i <= 5; i++) {
      const val = minX + (xRange / 5) * i;
      const x = padding.left + (chartWidth / 5) * i;
      ctx.fillText(val.toFixed(1), x, height - padding.bottom + 8);
    }

    animationFrameRef.current = requestAnimationFrame(render);
  }, [trades]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleResize = () => {
      canvas.width = canvas.clientWidth * window.devicePixelRatio;
      canvas.height = canvas.clientHeight * window.devicePixelRatio;
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-yellow-500/20">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute top-2 left-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-yellow-500/20">
        <h4 className="text-yellow-400 font-mono text-xs uppercase tracking-wider">
          TCA: Slippage vs Impact
        </h4>
        <div className="mt-1 flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full bg-green-500/60"></span>
          <span className="text-gray-400">Low Toxicity</span>
          <span className="w-2 h-2 rounded-full bg-red-500/60 ml-2"></span>
          <span className="text-gray-400">High Toxicity</span>
        </div>
      </div>
    </div>
  );
};
