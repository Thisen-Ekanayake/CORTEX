import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // User-management backend (auth, projects, conversations) on port 8000.
      // It also mounts routes under /api, so we expose it on a distinct prefix
      // and rewrite /backend-api/* → /api/* to avoid colliding with cortex_api.
      '/backend-api': {
        target: process.env.VITE_BACKEND_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/backend-api/, '/api'),
      },
      // Proxy AI API calls to the cortex_api service (run from the repo root
      // on port 8001). Override the target with VITE_API_TARGET if needed.
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
