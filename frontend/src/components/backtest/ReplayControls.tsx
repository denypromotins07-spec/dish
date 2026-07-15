import { useState, useEffect, useRef, useCallback } from 'react';
import { useReplayStore } from './ReplayStore';
import { wsClient } from '../../core/wsClient';

type PlaybackSpeed = 1 | 10 | 100 | 1000;

export const ReplayControls: React.FC = () => {
  const { isPlaying, currentTime, totalDuration, speed, setSpeed, setIsPlaying, seekTo } = useReplayStore();
  const [localSpeed, setLocalSpeed] = useState<PlaybackSpeed>(speed);
  const sliderRef = useRef<HTMLInputElement>(null);
  const animationFrameRef = useRef<number>();

  // Debounced speed update to avoid spamming backend
  const updateSpeed = useCallback((newSpeed: PlaybackSpeed) => {
    setLocalSpeed(newSpeed);
    setSpeed(newSpeed);
    
    // Send throttle command to Rust backend
    wsClient.sendBinary({
      type: 'REPLAY_SPEED_CHANGE',
      payload: { speed: newSpeed }
    });
  }, [setSpeed]);

  const togglePlayback = useCallback(() => {
    const newState = !isPlaying;
    setIsPlaying(newState);
    wsClient.sendBinary({
      type: isPlaying ? 'REPLAY_PAUSE' : 'REPLAY_RESUME',
      payload: {}
    });
  }, [isPlaying, setIsPlaying]);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const timestamp = parseInt(e.target.value, 10);
    seekTo(timestamp);
  }, [seekTo]);

  const handleSeekEnd = useCallback(() => {
    // Only send seek command when user releases slider
    wsClient.sendBinary({
      type: 'REPLAY_SEEK',
      payload: { timestamp: currentTime }
    });
  }, [currentTime]);

  // Sync with backend time updates
  useEffect(() => {
    const unsubscribe = useReplayStore.subscribe(
      (state) => state.currentTime,
      (time) => {
        if (sliderRef.current && document.activeElement !== sliderRef.current) {
          sliderRef.current.value = time.toString();
        }
      }
    );
    return unsubscribe;
  }, []);

  const formatTime = (ms: number) => {
    const date = new Date(ms);
    return date.toISOString().split('T')[1].replace('Z', '');
  };

  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-900/80 backdrop-blur-md rounded-lg border border-cyan-500/20 shadow-lg shadow-cyan-500/10">
      <div className="flex items-center justify-between">
        <h3 className="text-cyan-400 font-mono text-sm uppercase tracking-wider">Backtest Replay Engine</h3>
        <div className="flex gap-2">
          {[1, 10, 100, 1000].map((s) => (
            <button
              key={s}
              onClick={() => updateSpeed(s as PlaybackSpeed)}
              className={`px-3 py-1 text-xs font-mono rounded transition-all duration-200 ${
                localSpeed === s
                  ? 'bg-cyan-500 text-black shadow-[0_0_10px_rgba(6,182,212,0.5)]'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={togglePlayback}
          className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 ${
            isPlaying
              ? 'bg-red-500/20 text-red-400 border border-red-500/50 hover:bg-red-500/30'
              : 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/50 hover:bg-cyan-500/30'
          }`}
        >
          {isPlaying ? (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-5 h-5 ml-1" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>

        <div className="flex-1 relative">
          <input
            ref={sliderRef}
            type="range"
            min="0"
            max={totalDuration}
            value={currentTime}
            onChange={handleSeek}
            onMouseUp={handleSeekEnd}
            onTouchEnd={handleSeekEnd}
            className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-cyan-500 hover:accent-cyan-400 transition-all"
            style={{
              background: `linear-gradient(to right, #06b6d4 0%, #06b6d4 ${(currentTime / totalDuration) * 100}%, #1f2937 ${(currentTime / totalDuration) * 100}%, #1f2937 100%)`
            }}
          />
          <div className="absolute -bottom-5 left-0 text-xs text-gray-500 font-mono">
            {formatTime(currentTime)}
          </div>
          <div className="absolute -bottom-5 right-0 text-xs text-gray-500 font-mono">
            {formatTime(totalDuration)}
          </div>
        </div>
      </div>

      <div className="flex justify-between text-xs text-gray-400 font-mono">
        <span>Progress: {((currentTime / totalDuration) * 100).toFixed(2)}%</span>
        <span>Speed: {localSpeed}x</span>
      </div>
    </div>
  );
};
