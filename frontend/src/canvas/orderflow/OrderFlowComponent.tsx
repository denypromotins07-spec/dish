import React, { useRef, useEffect, useCallback } from 'react';
import { CVDLineRenderer } from './CVDLineRenderer';
import { DeltaBars } from './DeltaBars';
import { createWebGLContext } from '../WebGLContext';
import { deltaBarsVertexShader, deltaBarsFragmentShader } from './DeltaBars';
import { useMarketStore } from '../../core/store';

interface OrderFlowComponentProps {
  width: number;
  height: number;
  cvdHeightRatio?: number; // Ratio of height for CVD vs Delta bars
}

export const OrderFlowComponent: React.FC<OrderFlowComponentProps> = ({
  width,
  height,
  cvdHeightRatio = 0.6
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cvdCanvasRef = useRef<HTMLCanvasElement>(null);
  const deltaCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const cvdRendererRef = useRef<CVDLineRenderer | null>(null);
  const deltaBarsRef = useRef<DeltaBars | null>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Calculate heights
  const cvdHeight = Math.floor(height * cvdHeightRatio);
  const deltaHeight = Math.floor(height * (1 - cvdHeightRatio));
  
  // Access tick data (zero-copy)
  const getTickBuffer = useMarketStore((state) => state.getTickBuffer);
  const subscribeToTicks = useMarketStore((state) => state.subscribeToTicks);
  
  // Initialize renderers
  useEffect(() => {
    const cvdCanvas = cvdCanvasRef.current;
    const deltaCanvas = deltaCanvasRef.current;
    
    if (!cvdCanvas || !deltaCanvas) return;
    
    // Create CVD renderer
    cvdRendererRef.current = new CVDLineRenderer(cvdCanvas, 100000);
    
    // Create WebGL context for delta bars
    const gl = createWebGLContext(deltaCanvas);
    if (!gl) {
      console.error('Failed to create WebGL context for delta bars');
      return;
    }
    
    glRef.current = gl;
    
    // Create delta bars renderer
    deltaBarsRef.current = new DeltaBars(gl, 50000);
    
    // Compile shaders
    deltaBarsRef.current.compileProgram(deltaBarsVertexShader, deltaBarsFragmentShader);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      deltaBarsRef.current?.destroy();
    };
  }, []);

  // Handle resize
  useEffect(() => {
    cvdRendererRef.current?.resize(width, cvdHeight);
    
    const deltaCanvas = deltaCanvasRef.current;
    const gl = glRef.current;
    if (deltaCanvas && gl) {
      const dpr = window.devicePixelRatio || 1;
      deltaCanvas.width = Math.floor(width * dpr);
      deltaCanvas.height = Math.floor(deltaHeight * dpr);
      gl.viewport(0, 0, deltaCanvas.width, deltaCanvas.height);
    }
  }, [width, cvdHeight, deltaHeight]);

  // Subscribe to tick updates
  useEffect(() => {
    const unsubscribe = subscribeToTicks((ticks) => {
      const cvdRenderer = cvdRendererRef.current;
      const deltaBars = deltaBarsRef.current;
      
      if (!cvdRenderer || !deltaBars) return;
      
      // Process ticks in batch
      for (const tick of ticks) {
        const aggressorSide = tick.aggressorSide || 0; // 1 = buy, -1 = sell
        const volume = tick.volume || 1;
        
        // Update CVD
        cvdRenderer.addTick(tick.timestamp, aggressorSide);
        
        // Update delta bars (aggregate by time bucket)
        // In production, this would bucket by candle/timeframe
        deltaBars.addBar(tick.timestamp, 
          aggressorSide < 0 ? volume : 0,
          aggressorSide > 0 ? volume : 0
        );
      }
    });
    
    return unsubscribe;
  }, [subscribeToTicks]);

  // Render loop
  const render = useCallback(() => {
    const cvdRenderer = cvdRendererRef.current;
    const deltaBars = deltaBarsRef.current;
    const gl = glRef.current;
    
    // Render CVD line
    if (cvdRenderer) {
      cvdRenderer.render();
    }
    
    // Render delta bars
    if (deltaBars && gl) {
      const timeRange: [number, number] = [Date.now() - 60000, Date.now()];
      const valueRange = deltaBars.getValueRange();
      
      deltaBars.render(
        deltaCanvasRef.current?.width || 0,
        deltaCanvasRef.current?.height || 0,
        timeRange,
        valueRange
      );
    }
    
    animationFrameRef.current = requestAnimationFrame(render);
  }, []);

  // Start render loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex flex-col"
      style={{ willChange: 'auto' }}
    >
      {/* CVD Line Chart */}
      <div 
        className="relative w-full border-b border-gray-800"
        style={{ height: `${cvdHeight}px` }}
      >
        <canvas
          ref={cvdCanvasRef}
          className="absolute inset-0 w-full h-full"
          style={{
            imageRendering: 'pixelated',
            willChange: 'contents'
          }}
        />
        <div className="absolute top-1 left-2 text-xs font-mono text-green-400">
          CVD: {cvdRendererRef.current?.getCurrentCVD().toFixed(0) || '0'}
        </div>
      </div>
      
      {/* Delta Bars */}
      <div 
        className="relative w-full"
        style={{ height: `${deltaHeight}px` }}
      >
        <canvas
          ref={deltaCanvasRef}
          className="absolute inset-0 w-full h-full"
          style={{
            imageRendering: 'pixelated',
            willChange: 'contents'
          }}
        />
        <div className="absolute top-1 left-2 text-xs font-mono text-cyan-400">
          Delta Bars: {deltaBarsRef.current?.getNumBars() || 0}
        </div>
      </div>
    </div>
  );
};

export default OrderFlowComponent;
