import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy AI API calls to the cortex_api service (run from the repo root
      // on port 8001). Override the target with VITE_API_TARGET if needed.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
