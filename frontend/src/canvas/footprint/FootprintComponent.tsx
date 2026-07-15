import React, { useRef, useEffect, useCallback } from 'react';
import { FootprintRenderer } from './FootprintRenderer';
import { VolumeProfile } from './VolumeProfile';
import { useMarketStore } from '../../core/store';

interface FootprintComponentProps {
  width: number;
  height: number;
  showVolumeProfile?: boolean;
}

export const FootprintComponent: React.FC<FootprintComponentProps> = ({
  width,
  height,
  showVolumeProfile = true
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const profileCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const footprintRef = useRef<FootprintRenderer | null>(null);
  const volumeProfileRef = useRef<VolumeProfile | null>(null);
  const animationFrameRef = useRef<number>(0);
  
  // Viewport state (managed outside React for performance)
  const viewportRef = useRef({
    visibleStart: 0,
    visibleEnd: 100,
    priceMin: 0,
    priceMax: 100,
    isPanning: false,
    isZooming: false,
    lastMouseX: 0,
    lastMouseY: 0,
    scale: 1.0
  });
  
  // Access tick data (zero-copy reference)
  const getTickBuffer = useMarketStore((state) => state.getTickBuffer);
  const addTickListener = useMarketStore((state) => state.addTickListener);
  
  // Initialize renderers
  useEffect(() => {
    const mainCanvas = mainCanvasRef.current;
    const profileCanvas = profileCanvasRef.current;
    
    if (!mainCanvas) return;
    
    // Create footprint renderer
    footprintRef.current = new FootprintRenderer(mainCanvas, width, height);
    
    // Create volume profile renderer
    if (showVolumeProfile && profileCanvas) {
      volumeProfileRef.current = new VolumeProfile(profileCanvas, 150, height);
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, showVolumeProfile]);

  // Handle canvas resize
  useEffect(() => {
    footprintRef.current?.resize(width, height);
    volumeProfileRef.current?.resize(150, height);
  }, [width, height]);

  // Render loop
  const render = useCallback(() => {
    const footprint = footprintRef.current;
    const volumeProfile = volumeProfileRef.current;
    const viewport = viewportRef.current;
    
    if (!footprint) return;
    
    // Get latest tick data
    const tickBuffer = getTickBuffer();
    
    // Process ticks into footprint nodes (batched)
    // In production, this would be done incrementally in the store
    const priceMin = viewport.priceMin;
    const priceMax = viewport.priceMax;
    
    // Clear and render
    footprint.render(viewport.visibleStart, viewport.visibleEnd, priceMin, priceMax);
    
    // Render volume profile if enabled
    if (volumeProfile) {
      volumeProfile.calculateValueArea();
      volumeProfile.render(priceMin, priceMax, height);
      
      // Composite to main canvas
      const ctx = mainCanvasRef.current?.getContext('2d');
      if (ctx) {
        // Draw profile on right side
        volumeProfile.composite(width - 150);
      }
    }
    
    animationFrameRef.current = requestAnimationFrame(render);
  }, [getTickBuffer, width, height]);

  // Start render loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [render]);

  // Zoom handler (CSS matrix transforms for GPU acceleration)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const viewport = viewportRef.current;
    
    if (e.ctrlKey || e.metaKey) {
      // Vertical scroll = zoom price axis
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      const priceRange = viewport.priceMax - viewport.priceMin;
      const centerPrice = (viewport.priceMax + viewport.priceMin) / 2;
      
      viewport.priceMin = centerPrice - (priceRange * zoomFactor) / 2;
      viewport.priceMax = centerPrice + (priceRange * zoomFactor) / 2;
    } else {
      // Horizontal scroll = pan time axis
      viewport.visibleStart += e.deltaY > 0 ? 5 : -5;
      viewport.visibleEnd += e.deltaY > 0 ? 5 : -5;
    }
  }, []);

  // Pan handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const viewport = viewportRef.current;
    viewport.isPanning = true;
    viewport.lastMouseX = e.clientX;
    viewport.lastMouseY = e.clientY;
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const viewport = viewportRef.current;
    
    if (viewport.isPanning) {
      const deltaX = e.clientX - viewport.lastMouseX;
      const deltaY = e.clientY - viewport.lastMouseY;
      
      viewport.visibleStart -= deltaX * 0.1;
      viewport.visibleEnd -= deltaX * 0.1;
      
      viewport.lastMouseX = e.clientX;
      viewport.lastMouseY = e.clientY;
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    viewportRef.current.isPanning = false;
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden"
      style={{
        touchAction: 'none',
        willChange: 'transform'
      }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Main footprint chart */}
      <canvas
        ref={mainCanvasRef}
        className="absolute inset-0"
        style={{
          imageRendering: 'pixelated',
          willChange: 'contents'
        }}
      />
      
      {/* Volume profile overlay */}
      {showVolumeProfile && (
        <canvas
          ref={profileCanvasRef}
          className="absolute right-0 top-0"
          style={{
            width: '150px',
            height: '100%',
            imageRendering: 'pixelated',
            willChange: 'contents'
          }}
        />
      )}
      
      {/* Price info overlay (minimal DOM) */}
      <div 
        className="absolute top-2 left-2 px-2 py-1 bg-black/50 backdrop-blur-sm rounded text-xs font-mono text-cyan-400"
        style={{ willChange: 'auto' }}
      >
        Footprint Chart
      </div>
    </div>
  );
};

export default FootprintComponent;
