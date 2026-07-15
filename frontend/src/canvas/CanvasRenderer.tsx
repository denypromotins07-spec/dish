import React, { useRef, useEffect, useCallback, memo } from 'react';

interface CanvasRendererProps {
  width?: number;
  height?: number;
  className?: string;
  onRender?: (ctx: CanvasRenderingContext2D, width: number, height: number) => void;
  fps?: number;
}

/**
 * Base React component wrapping a raw HTML5 Canvas element.
 * Uses requestAnimationFrame and direct DOM refs to draw data,
 * completely bypassing React's Virtual DOM diffing for tick-level updates.
 */
export const CanvasRenderer = memo<CanvasRendererProps>(({
  width,
  height,
  className = '',
  onRender,
  fps = 60,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const animationFrameRef = useRef<number>(0);
  const lastRenderTimeRef = useRef<number>(0);
  const frameIntervalRef = useRef<number>(1000 / fps);

  // Initialize canvas context once
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true, // Low-latency rendering
      willReadFrequently: false,
    });

    if (!ctx) {
      console.error('[Canvas] Failed to get 2D context');
      return;
    }

    ctxRef.current = ctx;

    // Enable image smoothing for better visual quality
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Set canvas resolution for HiDPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = (width || rect.width) * dpr;
    canvas.height = (height || rect.height) * dpr;
    
    ctx.scale(dpr, dpr);
    canvas.style.width = `${width || rect.width}px`;
    canvas.style.height = `${height || rect.height}px`;

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height]);

  // Render loop using requestAnimationFrame
  const renderLoop = useCallback((timestamp: number) => {
    if (!ctxRef.current || !canvasRef.current) return;

    const elapsed = timestamp - lastRenderTimeRef.current;

    if (elapsed >= frameIntervalRef.current) {
      lastRenderTimeRef.current = timestamp - (elapsed % frameIntervalRef.current);

      const ctx = ctxRef.current;
      const canvas = canvasRef.current;

      // Clear canvas efficiently
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Call custom render function if provided
      if (onRender) {
        onRender(ctx, canvas.width, canvas.height);
      }
    }

    animationFrameRef.current = requestAnimationFrame(renderLoop);
  }, [onRender]);

  // Start/stop render loop based on fps changes
  useEffect(() => {
    frameIntervalRef.current = 1000 / fps;
    
    animationFrameRef.current = requestAnimationFrame(renderLoop);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [fps, renderLoop]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const ctx = ctxRef.current;
      
      if (!canvas || !ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      
      canvas.width = (width || rect.width) * dpr;
      canvas.height = (height || rect.height) * dpr;
      
      ctx.scale(dpr, dpr);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [width, height]);

  return (
    <canvas
      ref={canvasRef}
      className={`gpu-accelerated ${className}`}
      style={{ 
        width: width || '100%', 
        height: height || '100%',
        contain: 'strict',
      }}
    />
  );
});

CanvasRenderer.displayName = 'CanvasRenderer';

export default CanvasRenderer;
