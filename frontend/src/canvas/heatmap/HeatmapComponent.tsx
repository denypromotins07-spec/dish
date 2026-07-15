import React, { useRef, useEffect, useCallback } from 'react';
import { createWebGLContext, compileShaderProgram } from '../WebGLContext';
import { heatmapVertexShader, heatmapFragmentShader } from './HeatmapShaders';
import { OrderBookMatrix } from './OrderBookMatrix';
import { useRenderLoop } from '../RenderLoop';
import { useMarketStore } from '../../core/store';

interface HeatmapComponentProps {
  width: number;
  height: number;
  spoofThreshold?: number;
}

export const HeatmapComponent: React.FC<HeatmapComponentProps> = ({
  width,
  height,
  spoofThreshold = 2.0
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const matrixRef = useRef<OrderBookMatrix | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Access high-frequency order book data (bypasses React re-renders)
  const getOrderBookSnapshot = useMarketStore((state) => state.getOrderBookSnapshot);
  
  // Initialize WebGL context and shaders
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const gl = createWebGLContext(canvas);
    if (!gl) {
      console.error('Failed to create WebGL2 context');
      return;
    }
    
    glRef.current = gl;
    
    // Compile shader program
    const program = compileShaderProgram(gl, heatmapVertexShader, heatmapFragmentShader);
    if (!program) {
      console.error('Failed to compile heatmap shaders');
      return;
    }
    
    programRef.current = program;
    
    // Initialize order book matrix
    matrixRef.current = new OrderBookMatrix(gl, 1000);
    
    // Set initial viewport
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    // Enable blending for glow effects
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      matrixRef.current?.destroy();
      if (programRef.current) {
        gl.deleteProgram(programRef.current);
      }
    };
  }, []);

  // Handle canvas resize with device pixel ratio
  useEffect(() => {
    const canvas = canvasRef.current;
    const gl = glRef.current;
    if (!canvas || !gl) return;
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    
    gl.viewport(0, 0, canvas.width, canvas.height);
  }, [width, height]);

  // Render loop using requestAnimationFrame (bypasses React Virtual DOM)
  const render = useCallback(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const matrix = matrixRef.current;
    
    if (!gl || !program || !matrix) return;
    
    // Clear canvas
    gl.clearColor(0.02, 0.02, 0.05, 1.0); // Deep OLED black
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Get latest order book snapshot (zero-copy from store)
    const snapshot = getOrderBookSnapshot();
    
    if (snapshot.levels.length > 0) {
      // Update GPU buffers
      matrix.updateOrderBook(snapshot.levels);
      
      // Bind attributes
      matrix.bindAttributes(program);
      
      // Update uniforms
      const bounds = matrix.getViewportBounds();
      const timeLoc = gl.getUniformLocation(program, 'u_time');
      const minPriceLoc = gl.getUniformLocation(program, 'u_minPrice');
      const maxPriceLoc = gl.getUniformLocation(program, 'u_maxPrice');
      const minTimeLoc = gl.getUniformLocation(program, 'u_minTime');
      const maxTimeLoc = gl.getUniformLocation(program, 'u_maxTime');
      const spoofThresholdLoc = gl.getUniformLocation(program, 'u_spoofThreshold');
      const bidColorLoc = gl.getUniformLocation(program, 'u_bidColor');
      const askColorLoc = gl.getUniformLocation(program, 'u_askColor');
      const alphaBaseLoc = gl.getUniformLocation(program, 'u_alphaBase');
      
      gl.uniform1f(timeLoc, performance.now() / 1000);
      gl.uniform1f(minPriceLoc, bounds.minPrice);
      gl.uniform1f(maxPriceLoc, bounds.maxPrice);
      gl.uniform1f(minTimeLoc, bounds.minTime);
      gl.uniform1f(maxTimeLoc, bounds.maxTime);
      gl.uniform1f(spoofThresholdLoc, spoofThreshold);
      
      // Cyberpunk colors: cyan for bids, magenta for asks
      gl.uniform3f(bidColorLoc, 0.0, 0.9, 0.9);   // Neon cyan
      gl.uniform3f(askColorLoc, 0.9, 0.0, 0.9);  // Neon magenta
      gl.uniform1f(alphaBaseLoc, 0.8);
      
      // Render
      matrix.render();
    }
    
    animationFrameRef.current = requestAnimationFrame(render);
  }, [getOrderBookSnapshot, spoofThreshold]);

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
    <canvas
      ref={canvasRef}
      className="w-full h-full"
      style={{ 
        imageRendering: 'pixelated',
        willChange: 'contents'
      }}
    />
  );
};

export default HeatmapComponent;
