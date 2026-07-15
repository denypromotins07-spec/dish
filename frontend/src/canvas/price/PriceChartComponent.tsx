import React, { useRef, useEffect, useCallback } from 'react';
import { createWebGLContext, compileShaderProgram } from '../WebGLContext';
import { candlestickVertexShader, candlestickFragmentShader } from './CandlestickShaders';
import { PriceScale } from './PriceScale';
import { useMarketStore } from '../../core/store';

interface CandleData {
  open: number;
  high: number;
  low: number;
  close: number;
  timestamp: number;
  volume: number;
}

interface PriceChartComponentProps {
  width: number;
  height: number;
  scaleWidth?: number;
}

export const PriceChartComponent: React.FC<PriceChartComponentProps> = ({
  width,
  height,
  scaleWidth = 60
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const scaleCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const priceScaleRef = useRef<PriceScale | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Pre-allocated candle buffer (zero GC)
  const candlesRef = useRef<CandleData[]>([]);
  const maxCandlesRef = useRef<number>(50000);
  
  // Viewport state (managed outside React)
  const viewportRef = useRef({
    minTime: 0,
    maxTime: 0,
    minPrice: 0,
    maxPrice: 0,
    visibleStart: 0,
    visibleEnd: 0
  });

  // Access historical data and stream
  const getHistoricalCandles = useMarketStore((state) => state.getHistoricalCandles);
  const subscribeToCandles = useMarketStore((state) => state.subscribeToCandles);

  // Initialize WebGL context and shaders
  useEffect(() => {
    const mainCanvas = mainCanvasRef.current;
    const scaleCanvas = scaleCanvasRef.current;
    
    if (!mainCanvas || !scaleCanvas) return;
    
    // Create WebGL context for main chart
    const gl = createWebGLContext(mainCanvas);
    if (!gl) {
      console.error('Failed to create WebGL2 context');
      return;
    }
    
    glRef.current = gl;
    
    // Compile candlestick shaders
    const program = compileShaderProgram(gl, candlestickVertexShader, candlestickFragmentShader);
    if (!program) {
      console.error('Failed to compile candlestick shaders');
      return;
    }
    
    programRef.current = program;
    
    // Initialize price scale
    priceScaleRef.current = new PriceScale(scaleCanvas, scaleWidth);
    
    // Set initial viewport
    gl.viewport(0, 0, mainCanvas.width, mainCanvas.height);
    
    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (programRef.current) {
        gl.deleteProgram(programRef.current);
      }
    };
  }, [scaleWidth]);

  // Handle resize
  useEffect(() => {
    const mainCanvas = mainCanvasRef.current;
    const gl = glRef.current;
    const priceScale = priceScaleRef.current;
    
    if (!mainCanvas || !gl) return;
    
    const dpr = window.devicePixelRatio || 1;
    const chartWidth = width - scaleWidth;
    
    mainCanvas.width = Math.floor(chartWidth * dpr);
    mainCanvas.height = Math.floor(height * dpr);
    mainCanvas.style.width = `${chartWidth}px`;
    mainCanvas.style.height = `${height}px`;
    
    gl.viewport(0, 0, mainCanvas.width, mainCanvas.height);
    
    // Resize price scale
    priceScale?.resize(scaleWidth, height);
  }, [width, height, scaleWidth]);

  // Load historical data
  useEffect(() => {
    const historicalCandles = getHistoricalCandles();
    
    if (historicalCandles.length > 0) {
      candlesRef.current = historicalCandles.slice(-maxCandlesRef.current);
      
      // Calculate initial viewport
      if (candlesRef.current.length > 0) {
        const first = candlesRef.current[0];
        const last = candlesRef.current[candlesRef.current.length - 1];
        
        viewportRef.current.minTime = first.timestamp;
        viewportRef.current.maxTime = last.timestamp;
        viewportRef.current.visibleStart = 0;
        viewportRef.current.visibleEnd = candlesRef.current.length;
        
        // Calculate price range
        let minP = Infinity;
        let maxP = -Infinity;
        for (const c of candlesRef.current) {
          if (c.low < minP) minP = c.low;
          if (c.high > maxP) maxP = c.high;
        }
        
        viewportRef.current.minPrice = minP;
        viewportRef.current.maxPrice = maxP;
        
        // Update price scale
        priceScaleRef.current?.setPriceRange(minP, maxP);
      }
    }
    
    // Subscribe to live candle updates
    const unsubscribe = subscribeToCandles((newCandles) => {
      for (const candle of newCandles) {
        const candles = candlesRef.current;
        
        // Check if we need to update the last candle or add a new one
        if (candles.length > 0) {
          const lastCandle = candles[candles.length - 1];
          const sameBucket = Math.floor(candle.timestamp / 60000) === 
                            Math.floor(lastCandle.timestamp / 60000);
          
          if (sameBucket) {
            // Update existing candle
            lastCandle.close = candle.close;
            lastCandle.high = Math.max(lastCandle.high, candle.high);
            lastCandle.low = Math.min(lastCandle.low, candle.low);
            lastCandle.volume += candle.volume;
          } else {
            // Add new candle
            if (candles.length >= maxCandlesRef.current) {
              candles.shift(); // Remove oldest
            }
            candles.push({ ...candle });
          }
        } else {
          candles.push({ ...candle });
        }
        
        // Update viewport price range
        const vp = viewportRef.current;
        if (candle.high > vp.maxPrice) vp.maxPrice = candle.high;
        if (candle.low < vp.minPrice) vp.minPrice = candle.low;
        
        // Mark price scale as dirty
        priceScaleRef.current?.setPriceRange(vp.minPrice, vp.maxPrice);
      }
    });
    
    return unsubscribe;
  }, [getHistoricalCandles, subscribeToCandles]);

  // Render loop
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const priceScale = priceScaleRef.current;
    const mainCanvas = mainCanvasRef.current;
    
    if (!gl || !program || !priceScale || !mainCanvas) return;
    
    // Clear canvas
    gl.clearColor(0.02, 0.02, 0.05, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    const candles = candlesRef.current;
    const vp = viewportRef.current;
    
    if (candles.length > 0 && program) {
      // Update uniforms
      const uMinPrice = gl.getUniformLocation(program, 'u_minPrice');
      const uMaxPrice = gl.getUniformLocation(program, 'u_maxPrice');
      const uMinTime = gl.getUniformLocation(program, 'u_minTime');
      const uMaxTime = gl.getUniformLocation(program, 'u_maxTime');
      const uResolution = gl.getUniformLocation(program, 'u_resolution');
      const uCandleWidth = gl.getUniformLocation(program, 'u_candleWidth');
      const uBullishColor = gl.getUniformLocation(program, 'u_bullishColor');
      const uBearishColor = gl.getUniformLocation(program, 'u_bearishColor');
      const uVolumeGlow = gl.getUniformLocation(program, 'u_volumeGlow');
      
      gl.uniform1f(uMinPrice, vp.minPrice);
      gl.uniform1f(uMaxPrice, vp.maxPrice);
      gl.uniform1f(uMinTime, vp.minTime);
      gl.uniform1f(uMaxTime, vp.maxTime);
      gl.uniform2f(uResolution, mainCanvas.width, mainCanvas.height);
      gl.uniform1f(uCandleWidth, (mainCanvas.width / (vp.visibleEnd - vp.visibleStart)) * 0.8);
      
      // Cyberpunk colors: green for bullish, red for bearish
      gl.uniform3f(uBullishColor, 0.0, 1.0, 0.5);
      gl.uniform3f(uBearishColor, 1.0, 0.0, 0.5);
      gl.uniform1f(uVolumeGlow, 0.3);
      
      // In a full implementation, we would bind instance buffers here
      // and draw using gl.drawArraysInstanced
      
      // For now, render price scale
      priceScale.render();
      
      // Draw current price
      if (candles.length > 0) {
        const currentPrice = candles[candles.length - 1].close;
        priceScale.drawCurrentPrice(currentPrice);
      }
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

  // Zoom handler
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const vp = viewportRef.current;
    
    if (e.ctrlKey || e.metaKey) {
      // Zoom price axis
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      const priceRange = vp.maxPrice - vp.minPrice;
      const centerPrice = (vp.maxPrice + vp.minPrice) / 2;
      
      vp.minPrice = centerPrice - (priceRange * zoomFactor) / 2;
      vp.maxPrice = centerPrice + (priceRange * zoomFactor) / 2;
      
      priceScaleRef.current?.setPriceRange(vp.minPrice, vp.maxPrice);
    } else {
      // Pan time axis
      const timeRange = vp.maxTime - vp.minTime;
      const panAmount = (e.deltaY / 1000) * timeRange;
      
      vp.minTime += panAmount;
      vp.maxTime += panAmount;
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full flex"
      style={{ willChange: 'auto' }}
      onWheel={handleWheel}
    >
      {/* Main candlestick chart */}
      <canvas
        ref={mainCanvasRef}
        className="flex-1 h-full"
        style={{
          imageRendering: 'pixelated',
          willChange: 'contents'
        }}
      />
      
      {/* Price scale */}
      <canvas
        ref={scaleCanvasRef}
        className="border-l border-gray-800"
        style={{
          width: `${scaleWidth}px`,
          height: '100%',
          imageRendering: 'pixelated',
          willChange: 'contents'
        }}
      />
    </div>
  );
};

export default PriceChartComponent;
