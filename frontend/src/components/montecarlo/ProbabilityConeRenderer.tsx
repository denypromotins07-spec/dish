import { useEffect, useRef, useCallback } from 'react';

interface MonteCarloPath {
  id: string;
  points: Float32Array; // [time, value] pairs
}

interface ProbabilityConeRendererProps {
  paths: MonteCarloPath[];
  baselineEquity?: Float32Array;
  confidenceLevels?: number[];
}

export const ProbabilityConeRenderer: React.FC<ProbabilityConeRendererProps> = ({
  paths,
  baselineEquity,
  confidenceLevels = [0.1, 0.5, 0.9],
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const offscreenCanvasRef = useRef<OffscreenCanvas | null>(null);

  // Pre-allocated color palette for alpha blending
  const colorPalette = useRef([
    { r: 6, g: 182, b: 212, a: 0.02 },   // Cyan (low alpha for dense areas)
    { r: 232, g: 121, b: 249, a: 0.03 }, // Fuchsia
    { r: 34, g: 211, b: 238, a: 0.04 },  // Light cyan
    { r: 168, g: 85, b: 247, a: 0.02 },  // Purple
  ]);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Use offscreen canvas for better performance
    if (!offscreenCanvasRef.current || 
        offscreenCanvasRef.current.width !== canvas.width ||
        offscreenCanvasRef.current.height !== canvas.height) {
      offscreenCanvasRef.current = new OffscreenCanvas(canvas.width, canvas.height);
    }
    
    const offscreen = offscreenCanvasRef.current;
    const offCtx = offscreen.getContext('2d');
    if (!offCtx) return;

    // Clear with fade effect for trailing
    offCtx.fillStyle = 'rgba(2, 2, 5, 0.3)';
    offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

    const width = offscreen.width;
    const height = offscreen.height;
    const padding = 40;

    // Find min/max across all paths
    let minValue = Infinity;
    let maxValue = -Infinity;
    let maxTime = 0;

    paths.forEach(path => {
      for (let i = 1; i < path.points.length; i += 2) {
        const val = path.points[i];
        minValue = Math.min(minValue, val);
        maxValue = Math.max(maxValue, val);
      }
      maxTime = Math.max(maxTime, path.points[path.points.length - 2]);
    });

    if (baselineEquity) {
      for (let i = 1; i < baselineEquity.length; i += 2) {
        minValue = Math.min(minValue, baselineEquity[i]);
        maxValue = Math.max(maxValue, baselineEquity[i]);
      }
      maxTime = Math.max(maxTime, baselineEquity[baselineEquity.length - 2]);
    }

    const range = maxValue - minValue || 1;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Draw probability cone using line simplification
    const colors = colorPalette.current;
    
    paths.forEach((path, idx) => {
      const color = colors[idx % colors.length];
      
      offCtx.beginPath();
      offCtx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`;
      offCtx.lineWidth = 0.5;

      for (let i = 0; i < path.points.length; i += 2) {
        const time = path.points[i];
        const value = path.points[i + 1];
        
        const x = padding + (time / maxTime) * chartWidth;
        const y = padding + chartHeight - ((value - minValue) / range) * chartHeight;

        if (i === 0) {
          offCtx.moveTo(x, y);
        } else {
          offCtx.lineTo(x, y);
        }
      }
      
      offCtx.stroke();
    });

    // Draw baseline equity curve
    if (baselineEquity && baselineEquity.length > 0) {
      offCtx.beginPath();
      offCtx.strokeStyle = 'rgba(34, 211, 238, 0.8)';
      offCtx.lineWidth = 2;
      offCtx.setLineDash([5, 3]);

      for (let i = 0; i < baselineEquity.length; i += 2) {
        const time = baselineEquity[i];
        const value = baselineEquity[i + 1];
        
        const x = padding + (time / maxTime) * chartWidth;
        const y = padding + chartHeight - ((value - minValue) / range) * chartHeight;

        if (i === 0) {
          offCtx.moveTo(x, y);
        } else {
          offCtx.lineTo(x, y);
        }
      }
      
      offCtx.stroke();
      offCtx.setLineDash([]);
    }

    // Draw axes
    offCtx.strokeStyle = 'rgba(75, 85, 99, 0.5)';
    offCtx.lineWidth = 1;
    offCtx.beginPath();
    offCtx.moveTo(padding, padding);
    offCtx.lineTo(padding, height - padding);
    offCtx.lineTo(width - padding, height - padding);
    offCtx.stroke();

    // Copy to main canvas
    ctx.drawImage(offscreen, 0, 0);

    animationFrameRef.current = requestAnimationFrame(render);
  }, [paths, baselineEquity]);

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
    <div className="relative w-full h-full bg-gray-950 rounded-lg overflow-hidden border border-cyan-500/20">
      <canvas
        ref={canvasRef}
        className="w-full h-full"
        style={{ imageRendering: 'auto' }}
      />
      <div className="absolute top-2 left-2 px-3 py-2 bg-black/60 backdrop-blur-sm rounded-lg border border-cyan-500/20">
        <h4 className="text-cyan-400 font-mono text-xs uppercase tracking-wider mb-1">
          Monte Carlo Probability Cone
        </h4>
        <div className="flex items-center gap-2 text-xs text-gray-400 font-mono">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-cyan-500/30 rounded-full"></span>
            {paths.length.toLocaleString()} Paths
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
            Baseline
          </span>
        </div>
      </div>
    </div>
  );
};
