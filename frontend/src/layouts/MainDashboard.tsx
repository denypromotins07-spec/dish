import React, { memo, useCallback } from 'react';
import { useUIStore } from '@/core/store';

interface MainDashboardProps {
  children: React.ReactNode;
}

/**
 * Master layout shell using CSS Grid.
 * Allocates strict, non-shifting zones for order book heatmap,
 * footprint charts, portfolio telemetry, and execution panels.
 * Zero layout shift (CLS) when data floods in.
 */
export const MainDashboard = memo<MainDashboardProps>(({ children }) => {
  const activeTab = useUIStore((state) => state.activeTab);
  const showTelemetry = useUIStore((state) => state.showTelemetry);

  // Prevent re-renders on layout changes
  const handleTabChange = useCallback(() => {}, []);

  return (
    <div className="w-full h-full grid gap-px bg-background-secondary/20"
         style={{
           gridTemplateColumns: '280px 1fr 320px',
           gridTemplateRows: '48px 1fr 200px',
           gridTemplateAreas: `
             "header header header"
             "sidebar main rightbar"
             "sidebar bottom bottom"
           `,
         }}>
      
      {/* Header - Fixed height, no shifting */}
      <header className="grid-area-[header] flex items-center justify-between px-4 py-2 
                         glass-strong border-b border-surface-active/50 z-50">
        <div className="flex items-center gap-4">
          <h1 className="text-accent-cyan font-bold text-lg tracking-wider neon-text-bid">
            QUANTUM TRADER
          </h1>
          <nav className="flex items-center gap-2">
            {['dashboard', 'orderbook', 'charts', 'settings'].map((tab) => (
              <button
                key={tab}
                onClick={() => useUIStore.getState().setActiveTab(tab)}
                className={`px-3 py-1.5 text-xs font-medium uppercase tracking-wide transition-all duration-150
                           ${activeTab === tab 
                             ? 'bg-surface-active text-accent-cyan border border-accent-cyan/50' 
                             : 'text-gray-400 hover:text-white hover:bg-surface-hover'}`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs text-mono-tight">
            <span className="text-gray-500">SESSION:</span>
            <span className="text-accent-green">ACTIVE</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-mono-tight">
            <span className="text-gray-500">LATENCY:</span>
            <span className="text-accent-cyan">{useUIStore.getState().wsPing}ms</span>
          </div>
        </div>
      </header>

      {/* Sidebar - Order Book / Navigation */}
      <aside className="grid-area-[sidebar] bg-background-secondary border-r border-surface-active/30 
                        overflow-hidden flex flex-col">
        <div className="p-3 border-b border-surface-active/30">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Order Book
          </h2>
        </div>
        <div className="flex-1 overflow-hidden relative">
          {/* Order book canvas container - zero layout shift */}
          <div className="absolute inset-0 layout-stable">
            {children}
          </div>
        </div>
      </aside>

      {/* Main Content - Charts / Heatmaps */}
      <main className="grid-area-[main] bg-background-primary relative overflow-hidden">
        {/* Main chart canvas area - GPU accelerated */}
        <div className="absolute inset-0 gpu-accelerated layout-stable">
          {children}
        </div>
        
        {/* Scanline overlay effect */}
        <div className="scanline-overlay pointer-events-none opacity-50" />
      </main>

      {/* Right Bar - Execution / Portfolio */}
      <aside className="grid-area-[rightbar] bg-background-secondary border-l border-surface-active/30 
                        overflow-hidden flex flex-col">
        <div className="p-3 border-b border-surface-active/30">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Execution Panel
          </h2>
        </div>
        <div className="flex-1 overflow-hidden p-4 space-y-4">
          {/* Execution buttons - fixed positions */}
          <div className="grid grid-cols-2 gap-3">
            <button className="py-3 px-4 bg-gradient-to-r from-emerald-900/50 to-emerald-700/30 
                              border border-emerald-500/30 rounded text-emerald-400 font-bold 
                              hover:border-emerald-400 hover:shadow-lg hover:shadow-emerald-500/20 
                              transition-all duration-100 gpu-accelerated">
              BUY
            </button>
            <button className="py-3 px-4 bg-gradient-to-r from-rose-900/50 to-rose-700/30 
                              border border-rose-500/30 rounded text-rose-400 font-bold 
                              hover:border-rose-400 hover:shadow-lg hover:shadow-rose-500/20 
                              transition-all duration-100 gpu-accelerated">
              SELL
            </button>
          </div>
          
          {/* Position info */}
          <div className="space-y-2 text-xs text-mono-tight">
            <div className="flex justify-between">
              <span className="text-gray-500">POSITION</span>
              <span className="text-white">0.00 BTC</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">PNL</span>
              <span className="text-accent-green">+$0.00</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">EXPOSURE</span>
              <span className="text-white">$0.00</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Bottom Panel - Telemetry / Logs */}
      <footer className="grid-area-[bottom] bg-background-tertiary border-t border-surface-active/30 
                         overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 border-b border-surface-active/30">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            System Telemetry
          </h2>
          <button 
            onClick={() => useUIStore.getState().toggleTelemetry()}
            className="text-xs text-gray-500 hover:text-white transition-colors"
          >
            {showTelemetry ? 'HIDE' : 'SHOW'}
          </button>
        </div>
        {showTelemetry && (
          <div className="flex-1 overflow-hidden p-4 gpu-accelerated">
            {children}
          </div>
        )}
      </footer>
    </div>
  );
});

MainDashboard.displayName = 'MainDashboard';

export default MainDashboard;
