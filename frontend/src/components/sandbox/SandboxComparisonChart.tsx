import { useEffect, useRef, useCallback } from 'react';

interface EquityPoint {
  time: number;
  baseline: number;
  shocked: number;
}

interface SandboxComparisonChartProps {
  data: EquityPoint[];
  shockStartTime?: number;
  shockEndTime?: number;
}

export const SandboxComparisonChart: React.FC<SandboxComparisonChartProps> = ({
  data,
  shockStartTime,
  shockEndTime,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 60, bottom: 40, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#020205';
    ctx.fillRect(0, 0, width, height);

    if (data.length === 0) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }

    // Find min/max values
    let minValue = Infinity;
    let maxValue = -Infinity;
    let maxTime = 0;

    data.forEach(point => {
      minValue = Math.min(minValue, point.baseline, point.shocked);
      maxValue = Math.max(maxValue, point.baseline, point.shocked);
      maxTime = Math.max(maxTime, point.time);
    });

    const valueRange = maxValue - minValue || 1;

    // Helper functions
    const xToPixel = (time: number) => 
      padding.left + (time / maxTime) * chartWidth;
    
    const yToPixel = (value: number) => 
      padding.top + chartHeight - ((value - minValue) / valueRange) * chartHeight;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw shock region background
    if (shockStartTime !== undefined && shockEndTime !== undefined) {
      const shockX1 = xToPixel(shockStartTime);
      const shockX2 = xToPixel(shockEndTime);
      
      const gradient = ctx.createLinearGradient(shockX1, 0, shockX2, 0);
      gradient.addColorStop(0, 'rgba(239, 68, 68, 0)');
      gradient.addColorStop(0.5, 'rgba(239, 68, 68, 0.15)');
      gradient.addColorStop(1, 'rgba(239, 68, 68, 0)');
      
      ctx.fillStyle = gradient;
      ctx.fillRect(shockX1, padding.top, shockX2 - shockX1, chartHeight);

      // Shock region labels
      ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('SHOCK INJECTED', (shockX1 + shockX2) / 2, padding.top - 8);
    }

    // Draw baseline equity curve
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
    ctx.lineWidth = 2;

    data.forEach((point, i) => {
      const x = xToPixel(point.time);
      const y = yToPixel(point.baseline);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw shocked equity curve
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.lineWidth = 2;

    data.forEach((point, i) => {
      const x = xToPixel(point.time);
      const y = yToPixel(point.shocked);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw divergence area between curves
    ctx.beginPath();
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(239, 68, 68, 0.2)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.1)');
    
    data.forEach((point, i) => {
      const x = xToPixel(point.time);
      const yBaseline = yToPixel(point.baseline);
      
      if (i === 0) {
        ctx.moveTo(x, yBaseline);
      } else {
        ctx.lineTo(x, yBaseline);
      }
    });

    for (let i = data.length - 1; i >= 0; i--) {
      const point = data[i];
      const x = xToPixel(point.time);
      const yShocked = yToPixel(point.shocked);
      ctx.lineTo(x, yShocked);
    }
    
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw axes
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    // Draw Y-axis labels
    ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= 5; i++) {
      const value = minValue + (valueRange / 5) * (5 - i);
      const y = padding.top + (chartHeight / 5) * i;
      ctx.fillText(value.toFixed(2), padding.left - 8, y);
    }

    // Draw X-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i <= 10; i++) {
      const time = (maxTime / 10) * i;
      const x = padding.left + (chartWidth / 10) * i;
      const label = time < 3600 
        ? `${Math.floor(time / 60)}m` 
        : `${Math.floor(time / 3600)}h`;
      ctx.fillText(label, x, height - padding.bottom + 8);
    }

    // Legend
    ctx.fillStyle = 'rgba(31, 41, 55, 0.9)';
    ctx.fillRect(width - padding.right + 10, padding.top, 50, 50);
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.5)';
    ctx.strokeRect(width - padding.right + 10, padding.top, 50, 50);

    // Baseline legend
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(width - padding.right + 15, padding.top + 15);
    ctx.lineTo(width - padding.right + 35, padding.top + 15);
    ctx.stroke();
    ctx.fillStyle = 'rgba(6, 182, 212, 0.8)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Baseline', width - padding.right + 40, padding.top + 18);

    // Shocked legend
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.beginPath();
    ctx.moveTo(width - padding.right + 15, padding.top + 35);
    ctx.lineTo(width - padding.right + 35, padding.top + 35);
    ctx.stroke();
    ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.fillText('Shocked', width - padding.right + 40, padding.top + 38);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [data, shockStartTime, shockEndTime]);

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
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-red-500/20">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: 'auto' }}
      />
      <div className="absolute top-2 left-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-cyan-500/20">
        <h4 className="text-cyan-400 font-mono text-xs uppercase tracking-wider">
          Sandbox Comparison
        </h4>
        <div className="flex items-center gap-4 mt-1 text-xs font-mono">
          <span className="text-cyan-400">Baseline</span>
          <span className="text-red-400">Shocked Scenario</span>
        </div>
      </div>
    </div>
  );
};
