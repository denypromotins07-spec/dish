/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Cyberpunk/Institutional dark palette
        background: {
          DEFAULT: '#000000',
          secondary: '#0a0a0f',
          tertiary: '#12121a',
        },
        surface: {
          DEFAULT: '#1a1a2e',
          hover: '#252540',
          active: '#2d2d4a',
        },
        // Neon accent colors
        bid: {
          DEFAULT: '#00ffff',
          glow: '#00cccc',
          dim: '#008888',
        },
        ask: {
          DEFAULT: '#ff00ff',
          glow: '#cc00cc',
          dim: '#880088',
        },
        accent: {
          cyan: '#00f0ff',
          magenta: '#ff00aa',
          purple: '#7c3aed',
          green: '#00ff88',
          red: '#ff3366',
          yellow: '#ffcc00',
        },
        // Status colors
        success: '#00ff88',
        warning: '#ffcc00',
        error: '#ff3366',
        info: '#00f0ff',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        'xs': ['0.625rem', { lineHeight: '0.75rem' }],
        'xxs': ['0.5rem', { lineHeight: '0.6rem' }],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      animation: {
        'pulse-fast': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'scanline': 'scanline 8s linear infinite',
        'blink': 'blink 1s step-end infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px theme("colors.bid.DEFAULT"), 0 0 10px theme("colors.bid.DEFAULT")' },
          '100%': { boxShadow: '0 0 10px theme("colors.bid.DEFAULT"), 0 0 20px theme("colors.bid.DEFAULT"), 0 0 30px theme("colors.bid.DEFAULT")' },
        },
        scanline: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      backdropBlur: {
        'xs': '2px',
      },
      borderWidth: {
        '3': '3px',
      },
      opacity: {
        '1': '0.01',
        '2': '0.02',
        '5': '0.05',
      },
    },
  },
  plugins: [],
};
