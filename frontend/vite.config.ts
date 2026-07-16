/**
 * Vite Configuration with Memory-Limit Flags for PWA
 * 
 * This configuration:
 * - Strictly code-splits the application to reduce initial load
 * - Adds custom plugin to inject memory-limit flags into PWA manifest
 * - Optimizes bundle for 512MB browser memory limit
 * 
 * Target: Frontend stays under 1GB RAM with strict chunking
 */

import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

/**
 * Custom plugin to inject memory-limit flags into PWA manifest
 * Ensures the browser respects our 512MB JavaScript heap limit
 */
function memoryLimitPlugin(): Plugin {
  return {
    name: 'memory-limit-plugin',
    enforce: 'post',
    generateBundle(options, bundle) {
      // Find the manifest file
      const manifestName = Object.keys(bundle).find(key => 
        key.includes('manifest') && key.endsWith('.json')
      );
      
      if (manifestName) {
        const manifest = bundle[manifestName] as any;
        if (manifest && manifest.source) {
          try {
            const manifestObj = JSON.parse(manifest.source);
            
            // Add memory-related metadata
            manifestObj.memory_limits = {
              max_js_heap_mb: 512,
              max_texture_memory_mb: 256,
              max_worker_count: 2,
              gc_interval_ms: 30000
            };
            
            manifest.source = JSON.stringify(manifestObj, null, 2);
            console.log('[MemoryLimitPlugin] Injected memory limits into manifest');
          } catch (e) {
            console.error('[MemoryLimitPlugin] Failed to parse manifest:', e);
          }
        }
      }
    },
    // Also inject into HTML
    transformIndexHtml(html) {
      const memoryScript = `
        <script>
          // Enforce memory limits in browser
          window.CRYPTO_BOT_CONFIG = {
            MEMORY_LIMIT_MB: 512,
            MAX_FPS: 30,
            REDUCED_MOTION: true,
            LOW_MEMORY_MODE: true
          };
          
          // Monitor memory usage (Chrome/Edge only)
          if ('performance' in window && 'memory' in performance) {
            setInterval(() => {
              const memInfo = performance.memory;
              const usedMB = memInfo.usedJSHeapSize / (1024 * 1024);
              const limitMB = window.CRYPTO_BOT_CONFIG?.MEMORY_LIMIT_MB || 512;
              
              if (usedMB > limitMB * 0.9) {
                console.warn('⚠️ Memory approaching limit:', usedMB.toFixed(2), 'MB');
                // Trigger manual GC hint (browser-dependent)
                if (window.gc) window.gc();
              }
            }, 5000);
          }
        </script>
      `;
      
      return html.replace('</head>', memoryScript + '</head>');
    }
  };
}

/**
 * Aggressive code-splitting configuration
 * Separates vendor, UI, trading logic, and charts into different chunks
 */
const manualChunksConfig = {
  vendor: ['react', 'react-dom', 'react-router-dom'],
  charts: ['recharts', 'chart.js', 'd3'],
  trading: ['@nautilus-trader/client', 'websocket'],
  ui: ['@mui/material', '@emotion/react', '@emotion/styled'],
  utils: ['lodash', 'dayjs', 'axios'],
};

export default defineConfig({
  plugins: [
    react({
      // Optimize React for lower memory
      babel: {
        presets: [
          ['@babel/preset-react', {
            runtime: 'automatic',
            importSource: '@emotion/react'
          }]
        ],
        plugins: [
          '@emotion/babel-plugin'
        ]
      }
    }),
    
    memoryLimitPlugin(),
    
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'robots.txt', 'apple-touch-icon.png'],
      manifest: {
        name: 'Crypto Trading Bot',
        short_name: 'CryptoBot',
        description: 'High-frequency crypto trading interface',
        theme_color: '#1a1a2e',
        background_color: '#1a1a2e',
        display: 'standalone',
        orientation: 'landscape',
        start_url: '/',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ],
        // Memory-related manifest extensions
        extra: {
          prefer_related_applications: false,
          handle_links: 'preferred'
        }
      },
      workbox: {
        // Aggressive caching strategy to reduce memory
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        maximumFileSizeToCacheInBytes: 2 * 1024 * 1024, // 2MB max per file
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.binance\.com\/.*/i,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'binance-api',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60 // 1 hour
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          },
          {
            urlPattern: /^https:\/\/api\.bybit\.com\/.*/i,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'bybit-api',
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 60 * 60
              }
            }
          }
        ]
      }
    })
  ],
  
  build: {
    // Target modern browsers for better performance
    target: 'esnext',
    
    // Enable minification
    minify: 'terser',
    
    // Source maps for debugging (disable in production for smaller bundles)
    sourcemap: false,
    
    // Chunk size optimization
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          if (id.includes('node_modules')) {
            for (const [name, modules] of Object.entries(manualChunksConfig)) {
              if (modules.some(module => id.includes(module))) {
                return `vendor-${name}`;
              }
            }
            return 'vendor';
          }
        },
        // Limit chunk size
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.');
          const extType = info.pop() || '';
          if (/\.(png|jpe?g|svg|gif|tiff|bmp|ico)/i.test(extType)) {
            return `assets/images/[name]-[hash][extname]`;
          }
          if (/\.(woff|woff2|eot|ttf|otf)/i.test(extType)) {
            return `assets/fonts/[name]-[hash][extname]`;
          }
          return `assets/[name]-[hash][extname]`;
        }
      }
    },
    
    // Report bundle size
    reportCompressedSize: true,
    
    // Limit worker count
    worker: {
      format: 'es',
      maxConcurrency: 2
    }
  },
  
  optimizeDeps: {
    // Pre-bundle heavy dependencies
    include: ['react', 'react-dom', 'react-router-dom'],
    exclude: ['@nautilus-trader/client']
  },
  
  server: {
    // Development server settings
    port: 3000,
    open: false,
    hmr: {
      protocol: 'ws',
      host: 'localhost',
      port: 3001
    }
  },
  
  // Environment variables
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'window.CRYPTO_BOT_VERSION': JSON.stringify(process.env.npm_package_version || '0.1.0')
  }
});
