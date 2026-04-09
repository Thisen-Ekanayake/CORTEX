/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          base: 'var(--bg-base)',
          elevated: 'var(--bg-elevated)',
          overlay: 'var(--bg-overlay)',
        },
        surface: {
          glass: 'var(--surface-glass)',
          highlight: 'var(--surface-glass-highlight)',
        },
        border: {
          subtle: 'var(--border-subtle)',
          highlight: 'var(--border-highlight)',
        },
        text: {
          primary: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          tertiary: 'var(--text-tertiary)',
        },
        accent: {
          primary: 'var(--accent-primary)',
          glow: 'var(--accent-glow)',
          success: 'var(--accent-success)',
          warning: 'var(--accent-warning)',
          error: 'var(--accent-error)',
        }
      },
      boxShadow: {
        'panel': '0 8px 32px 0 rgba(0, 0, 0, 0.36)',
        'glass': '0 4px 30px rgba(0, 0, 0, 0.1)',
        'glow': '0 0 20px -5px var(--accent-primary)',
        'premium-blue': '0 10px 25px -5px rgba(88, 166, 255, 0.3), 0 8px 10px -6px rgba(88, 166, 255, 0.3)',
        'premium-glass': '0 10px 30px -10px rgba(0, 0, 0, 0.5)',
      },
      backdropBlur: {
        'xs': '2px',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
