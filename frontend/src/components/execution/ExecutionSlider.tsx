import React, { useRef, useCallback, useEffect, useState } from 'react';

interface ExecutionSliderProps {
  min: number;
  max: number;
  step: number;
  initialValue: number;
  label: string;
  unit?: string;
  onChange: (value: number) => void;
  color?: 'cyan' | 'magenta' | 'emerald' | 'rose';
}

export const ExecutionSlider: React.FC<ExecutionSliderProps> = ({
  min,
  max,
  step,
  initialValue,
  label,
  unit = '',
  onChange,
  color = 'cyan'
}) => {
  const [displayValue, setDisplayValue] = useState(initialValue);
  const sliderRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  // Color mapping for Tailwind classes
  const colorMap = {
    cyan: { bg: 'bg-cyan-500', shadow: 'shadow-[0_0_20px_rgba(6,182,212,0.6)]', text: 'text-cyan-400' },
    magenta: { bg: 'bg-magenta-500', shadow: 'shadow-[0_0_20px_rgba(236,72,153,0.6)]', text: 'text-magenta-400' },
    emerald: { bg: 'bg-emerald-500', shadow: 'shadow-[0_0_20px_rgba(16,185,129,0.6)]', text: 'text-emerald-400' },
    rose: { bg: 'bg-rose-500', shadow: 'shadow-[0_0_20px_rgba(244,63,94,0.6)]', text: 'text-rose-400' }
  };

  const currentColor = colorMap[color];

  // Calculate percentage position
  const getPercentage = useCallback((value: number) => {
    return ((value - min) / (max - min)) * 100;
  }, [min, max]);

  // Calculate value from client X position
  const getValueFromPosition = useCallback((clientX: number) => {
    if (!trackRef.current) return initialValue;
    
    const rect = trackRef.current.getBoundingClientRect();
    const percentage = (clientX - rect.left) / rect.width;
    const clampedPercentage = Math.max(0, Math.min(1, percentage));
    const rawValue = min + (clampedPercentage * (max - min));
    
    // Snap to step
    const steppedValue = Math.round(rawValue / step) * step;
    return Math.max(min, Math.min(max, steppedValue));
  }, [min, max, step, initialValue]);

  // Handle pointer events directly for 60fps performance
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    isDragging.current = true;
    sliderRef.current?.setPointerCapture(e.pointerId);
    
    const newValue = getValueFromPosition(e.clientX);
    setDisplayValue(newValue);
    onChange(newValue);
  }, [getValueFromPosition, onChange]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging.current) return;
    
    const newValue = getValueFromPosition(e.clientX);
    setDisplayValue(newValue);
    onChange(newValue);
  }, [getValueFromPosition, onChange]);

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    isDragging.current = false;
    sliderRef.current?.releasePointerCapture(e.pointerId);
  }, []);

  // Update thumb position using direct DOM manipulation for smooth animation
  useEffect(() => {
    if (thumbRef.current) {
      const percentage = getPercentage(displayValue);
      thumbRef.current.style.transform = `translateX(${percentage}%)`;
    }
  }, [displayValue, getPercentage]);

  return (
    <div className="w-full select-none">
      {/* Header */}
      <div className="flex justify-between items-center mb-2">
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</label>
        <span className={`text-sm font-mono font-bold ${currentColor.text}`}>
          {displayValue.toFixed(step < 1 ? 3 : 0)}{unit}
        </span>
      </div>

      {/* Slider Track Container */}
      <div
        ref={sliderRef}
        className="relative h-8 flex items-center cursor-pointer"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        {/* Background Track */}
        <div
          ref={trackRef}
          className="w-full h-2 bg-gray-800 rounded-full overflow-hidden"
        >
          {/* Active Fill */}
          <div
            className={`h-full ${currentColor.bg} transition-all duration-75 ease-out`}
            style={{ width: `${getPercentage(displayValue)}%` }}
          />
        </div>

        {/* Thumb (Knob) - Hardware Accelerated */}
        <div
          ref={thumbRef}
          className={`absolute top-1/2 -mt-3 w-6 h-6 rounded-full ${currentColor.bg} ${currentColor.shadow} border-2 border-white transform -translate-x-1/2 will-change-transform transition-shadow duration-200`}
          style={{ left: 0 }}
        >
          {/* Inner glow */}
          <div className="absolute inset-1 rounded-full bg-white/30"></div>
        </div>

        {/* Min/Max Labels */}
        <div className="absolute -bottom-5 left-0 text-xs text-gray-600 font-mono">{min}</div>
        <div className="absolute -bottom-5 right-0 text-xs text-gray-600 font-mono">{max}</div>
      </div>

      {/* Quick Presets */}
      <div className="flex gap-2 mt-6">
        {[0.25, 0.5, 0.75, 1].map((pct) => {
          const presetValue = min + (pct * (max - min));
          return (
            <button
              key={pct}
              onClick={() => {
                const snapped = Math.round(presetValue / step) * step;
                setDisplayValue(snapped);
                onChange(snapped);
              }}
              className="flex-1 py-1 text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-white rounded transition-colors border border-gray-700 hover:border-gray-600"
            >
              {Math.round(pct * 100)}%
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ExecutionSlider;
