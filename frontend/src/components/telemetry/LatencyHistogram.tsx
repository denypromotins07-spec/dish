import React, { useRef, useEffect, useCallback, memo } from 'react';
import { useHFStore } from '@/core/store';

interface LatencyHistogramProps {
  width?: number;
  height?: number;
  maxLatencyMs?: number;
}

/**
 * Custom Canvas-based HDR histogram visualizer.
 * Draws microsecond execution latencies and network jitter directly to canvas,
 * replacing heavy SVG-based charting libraries that would crash the browser.
 */
export const LatencyHistogram = memo<LatencyHistogramProps>(({
  width = 400,
  height = 120,
  maxLatencyMs = 100,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const histogramDataRef = useRef<Uint32Array>(new Uint32Array(100));
  const latencySamplesRef = useHFStore((state) => state.latencySamples);

  // Draw histogram to canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', {
      desynchronized: true,
      willReadFrequently: false,
    });
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = width;
    const displayHeight = height;

    // Set canvas resolution for HiDPI
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    // Get histogram data
    const samples = latencySamplesRef || new Uint32Array(0);
    if (samples.length === 0) {
      // Draw empty state
      ctx.fillStyle = '#333';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('NO LATENCY DATA', displayWidth / 2, displayHeight / 2);
      return;
    }

    // Build histogram buckets (logarithmic scale for HDR)
    const bucketCount = 100;
    const buckets = new Float32Array(bucketCount);
    const maxLatencyUs = maxLatencyMs * 1000; // Convert to microseconds

    for (let i = 0; i < samples.length; i++) {
      const latency = samples[i];
      // Logarithmic bucket assignment
      const bucket = Math.min(
        bucketCount - 1,
        Math.floor(Math.log2(latency + 1) / Math.log2(maxLatencyUs + 1) * bucketCount)
      );
      buckets[bucket]++;
    }

    // Find max bucket value for normalization
    let maxCount = 1;
    for (let i = 0; i < bucketCount; i++) {
      if (buckets[i] > maxCount) maxCount = buckets[i];
    }

    // Draw histogram bars
    const barWidth = displayWidth / bucketCount;
    const padding = 20;
    const graphHeight = displayHeight - padding * 2;

    for (let i = 0; i < bucketCount; i++) {
      const barHeight = (buckets[i] / maxCount) * graphHeight;
      const x = i * barWidth;
      const y = displayHeight - padding - barHeight;

      // Color based on latency range (green -> yellow -> red)
      const latencyRange = (i / bucketCount) * maxLatencyMs;
      let r, g, b;

      if (latencyRange < 1) {
        // Green for sub-millisecond
        r = 0;
        g = 255;
        b = 136;
      } else if (latencyRange < 10) {
        // Yellow for 1-10ms
        r = 255;
        g = 204;
        b = 0;
      } else {
        // Red for >10ms
        r = 255;
        g = 51;
        b = 102;
      }

      // Gradient fill
      const gradient = ctx.createLinearGradient(x, y, x, displayHeight - padding);
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.8)`);
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0.2)`);

      ctx.fillStyle = gradient;
      ctx.fillRect(x + 1, y, barWidth - 2, barHeight);

      // Glow effect for high bars
      if (barHeight > graphHeight * 0.7) {
        ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.5)`;
        ctx.shadowBlur = 10;
        ctx.fillRect(x + 1, y, barWidth - 2, barHeight);
        ctx.shadowBlur = 0;
      }
    }

    // Draw axis labels
    ctx.fillStyle = '#666';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0μs', 0, displayHeight - 5);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxLatencyMs}ms`, displayWidth, displayHeight - 5);

    // Draw current latency markers
    if (samples.length > 0) {
      const minLatency = Math.min(...Array.from(samples));
      const maxLatency = Math.max(...Array.from(samples));
      const avgLatency = samples.reduce((a, b) => a + b, 0) / samples.length;

      // Draw average line
      const avgX = (Math.log2(avgLatency + 1) / Math.log2(maxLatencyUs + 1)) * displayWidth;
      ctx.strokeStyle = '#00ffff';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(avgX, padding);
      ctx.lineTo(avgX, displayHeight - padding);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = '#00ffff';
      ctx.font = '9px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`AVG: ${(avgLatency / 1000).toFixed(2)}ms`, avgX + 5, padding + 10);
    }

  }, [width, height, maxLatencyMs, latencySamplesRef]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      draw();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [draw]);

  return (
    <div className="relative layout-stable">
      <canvas
        ref={canvasRef}
        className="gpu-accelerated"
        style={{
          width: `${width}px`,
          height: `${height}px`,
        }}
      />
      {/* Overlay stats */}
      <div className="absolute top-2 right-2 text-xs text-mono-tight text-gray-400">
        HDR LATENCY HISTOGRAM
      </div>
    </div>
  );
});

LatencyHistogram.displayName = 'LatencyHistogram';

export default LatencyHistogram;
