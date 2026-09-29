import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const proxyTarget =
  process.env.PROXY_ORCHESTRATOR_URL ||
  process.env.VITE_ORCHESTRATOR_URL ||
  'http://orchestrator:8000'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true,
        secure: false,
        // Map frontend calls like /api/agents -> orchestrator /v1/agents
        rewrite: (path) => path.replace(/^\/api/, '/v1'),
      },
    },
  },
})
