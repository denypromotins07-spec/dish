import { useEffect, useRef, useCallback } from 'react';

interface StrategyMetrics {
  name: string;
  sharpe: number;
  sortino: number;
  calmar: number;
  winRate: number;
  profitFactor: number;
  maxDrawdown: number;
}

interface StrategyRadarChartProps {
  strategies: StrategyMetrics[];
}

export const StrategyRadarChart: React.FC<StrategyRadarChartProps> = ({ strategies }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const rotationRef = useRef<number>(0);

  const metrics = ['Sharpe', 'Sortino', 'Calmar', 'Win Rate', 'Profit Factor', 'Max DD'];

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 60;

    // Clear with fade
    ctx.fillStyle = 'rgba(2, 2, 5, 0.15)';
    ctx.fillRect(0, 0, width, height);

    if (strategies.length === 0) {
      animationFrameRef.current = requestAnimationFrame(render);
      return;
    }

    // Normalize values (0-1 scale)
    const normalize = (value: number, metric: string): number => {
      switch (metric) {
        case 'Sharpe': return Math.min(value / 3, 1);
        case 'Sortino': return Math.min(value / 4, 1);
        case 'Calmar': return Math.min(value / 2, 1);
        case 'Win Rate': return value;
        case 'Profit Factor': return Math.min(value / 2, 1);
        case 'Max DD': return 1 - Math.min(Math.abs(value), 1);
        default: return 0;
      }
    };

    // Draw concentric circles (grid)
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (radius / 4) * i, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw axis lines and labels
    const angleStep = (Math.PI * 2) / metrics.length;
    ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    metrics.forEach((metric, i) => {
      const angle = i * angleStep - Math.PI / 2 + rotationRef.current;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      // Axis line
      ctx.strokeStyle = 'rgba(75, 85, 99, 0.5)';
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();

      // Label
      ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
      const labelX = centerX + Math.cos(angle) * (radius + 20);
      const labelY = centerY + Math.sin(angle) * (radius + 20);
      ctx.fillText(metric, labelX, labelY);
    });

    // Color palette for strategies
    const colors = [
      { r: 6, g: 182, b: 212 },   // Cyan
      { r: 232, g: 121, b: 249 }, // Fuchsia
      { r: 34, g: 211, b: 238 },  // Light cyan
      { r: 168, g: 85, b: 247 },  // Purple
      { r: 52, g: 211, b: 153 },  // Green
    ];

    // Draw radar polygons for each strategy
    strategies.forEach((strategy, idx) => {
      const color = colors[idx % colors.length];
      
      ctx.beginPath();
      metrics.forEach((metric, i) => {
        const angle = i * angleStep - Math.PI / 2 + rotationRef.current;
        const normalizedValue = normalize(
          metric === 'Sharpe' ? strategy.sharpe :
          metric === 'Sortino' ? strategy.sortino :
          metric === 'Calmar' ? strategy.calmar :
          metric === 'Win Rate' ? strategy.winRate :
          metric === 'Profit Factor' ? strategy.profitFactor :
          strategy.maxDrawdown,
          metric
        );
        
        const x = centerX + Math.cos(angle) * radius * normalizedValue;
        const y = centerY + Math.sin(angle) * radius * normalizedValue;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();

      // Fill with gradient
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
      gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`);
      gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0.05)`);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Stroke
      ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw vertices
      metrics.forEach((metric, i) => {
        const angle = i * angleStep - Math.PI / 2 + rotationRef.current;
        const normalizedValue = normalize(
          metric === 'Sharpe' ? strategy.sharpe :
          metric === 'Sortino' ? strategy.sortino :
          metric === 'Calmar' ? strategy.calmar :
          metric === 'Win Rate' ? strategy.winRate :
          metric === 'Profit Factor' ? strategy.profitFactor :
          strategy.maxDrawdown,
          metric
        );
        
        const x = centerX + Math.cos(angle) * radius * normalizedValue;
        const y = centerY + Math.sin(angle) * radius * normalizedValue;

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 1)`;
        ctx.fill();
      });
    });

    // Legend
    const legendX = 10;
    let legendY = 10;
    
    strategies.forEach((strategy, idx) => {
      const color = colors[idx % colors.length];
      
      ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 1)`;
      ctx.fillRect(legendX, legendY, 12, 12);
      
      ctx.fillStyle = 'rgba(156, 163, 175, 0.8)';
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(strategy.name, legendX + 18, legendY + 10);
      
      legendY += 20;
    });

    // Slow rotation animation
    rotationRef.current += 0.002;

    animationFrameRef.current = requestAnimationFrame(render);
  }, [strategies]);

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
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-purple-500/20">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute top-2 right-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-purple-500/20">
        <h4 className="text-purple-400 font-mono text-xs uppercase tracking-wider">
          Strategy Comparison
        </h4>
      </div>
    </div>
  );
};
