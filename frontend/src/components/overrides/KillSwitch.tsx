import React, { useState, useCallback, useRef } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useSystemStore, useExecutionStore } from '../../core/store';

export const KillSwitch: React.FC = () => {
  const [armed, setArmed] = useState(false);
  const [confirmStage, setConfirmStage] = useState(0); // 0 = none, 1 = first click, 2 = activated
  const holdTimerRef = useRef<NodeJS.Timeout | null>(null);
  const holdProgressRef = useRef<HTMLDivElement>(null);

  const emergencyFlatten = useExecutionStore(useShallow(state => state.emergencyFlatten));
  const systemStatus = useSystemStore(useShallow(state => state.status));

  // Handle slide-to-confirm interaction
  const handleMouseDown = useCallback(() => {
    if (!armed) {
      setArmed(true);
      setConfirmStage(1);
      return;
    }

    // Start hold timer (1.5 seconds to confirm)
    holdTimerRef.current = setTimeout(() => {
      setConfirmStage(2);
      emergencyFlatten();
      
      // Reset after activation
      setTimeout(() => {
        setArmed(false);
        setConfirmStage(0);
      }, 3000);
    }, 1500);

    // Animate progress bar
    if (holdProgressRef.current) {
      holdProgressRef.current.style.transition = 'width 1.5s linear';
      holdProgressRef.current.style.width = '100%';
    }
  }, [armed, emergencyFlatten]);

  const handleMouseUp = useCallback(() => {
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    
    if (holdProgressRef.current && confirmStage < 2) {
      holdProgressRef.current.style.transition = 'width 0.2s ease-out';
      holdProgressRef.current.style.width = '0%';
    }
  }, [confirmStage]);

  const handleDisarm = useCallback(() => {
    setArmed(false);
    setConfirmStage(0);
    if (holdProgressRef.current) {
      holdProgressRef.current.style.width = '0%';
    }
  }, []);

  return (
    <div className="bg-gray-900/90 backdrop-blur-md border-2 border-red-900/50 rounded-xl p-6 shadow-2xl max-w-sm mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-red-500 font-black text-xl tracking-widest uppercase flex items-center">
          <span className="animate-pulse mr-2">🔴</span>
          Emergency Stop
        </h3>
        {systemStatus === 'EMERGENCY' && (
          <span className="px-2 py-1 bg-red-600 text-white text-xs font-bold rounded animate-pulse">
            ACTIVATED
          </span>
        )}
      </div>

      {/* Warning Text */}
      <p className="text-xs text-gray-400 mb-4">
        {confirmStage === 0 && !armed && "Press to arm emergency flatten. All positions will be closed immediately."}
        {confirmStage === 1 && armed && "HOLD FOR 1.5 SECONDS TO CONFIRM"}
        {confirmStage === 2 && "EMERGENCY FLATTEN EXECUTED"}
      </p>

      {/* Main Button */}
      <div
        className={`relative h-24 rounded-xl overflow-hidden cursor-pointer select-none transition-all duration-200 ${
          confirmStage === 2 
            ? 'bg-red-600 shadow-[0_0_40px_rgba(220,38,38,0.8)]' 
            : armed 
              ? 'bg-gradient-to-r from-orange-600 to-red-600 shadow-[0_0_30px_rgba(239,68,68,0.6)]'
              : 'bg-gradient-to-r from-red-900 to-red-800 hover:from-red-800 hover:to-red-700 shadow-lg'
        }`}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onTouchStart={handleMouseDown}
        onTouchEnd={handleMouseUp}
      >
        {/* Progress overlay for hold confirmation */}
        <div 
          ref={holdProgressRef}
          className="absolute inset-0 bg-white/20"
          style={{ width: '0%' }}
        />
        
        {/* Content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
          {confirmStage === 2 ? (
            <>
              <span className="text-4xl mb-1">✅</span>
              <span className="text-white font-black text-lg">FLATTENED</span>
            </>
          ) : armed ? (
            <>
              <span className="text-3xl mb-1 animate-bounce">⚠️</span>
              <span className="text-white font-black text-lg">HOLD TO CONFIRM</span>
            </>
          ) : (
            <>
              <span className="text-3xl mb-1">🛑</span>
              <span className="text-red-400 font-black text-lg">KILL SWITCH</span>
            </>
          )}
        </div>
      </div>

      {/* Disarm Button (when armed but not confirmed) */}
      {armed && confirmStage === 1 && (
        <button
          onClick={handleDisarm}
          className="mt-3 w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-400 text-xs font-bold rounded transition-colors border border-gray-700"
        >
          CANCEL
        </button>
      )}

      {/* Status Indicators */}
      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <div className={`text-xs py-2 rounded ${
          systemStatus === 'NORMAL' ? 'bg-emerald-900/30 text-emerald-400' : 'bg-gray-800 text-gray-500'
        }`}>
          NORMAL
        </div>
        <div className={`text-xs py-2 rounded ${
          systemStatus === 'PAUSED' ? 'bg-yellow-900/30 text-yellow-400' : 'bg-gray-800 text-gray-500'
        }`}>
          PAUSED
        </div>
        <div className={`text-xs py-2 rounded ${
          systemStatus === 'EMERGENCY' ? 'bg-red-900/50 text-red-400 animate-pulse' : 'bg-gray-800 text-gray-500'
        }`}>
          EMERGENCY
        </div>
      </div>
    </div>
  );
};

export default KillSwitch;
