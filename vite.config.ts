import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.GEOSENTINEL_API_TARGET || 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/docs': {
        target: process.env.GEOSENTINEL_API_TARGET || 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
      '/openapi.json': {
        target: process.env.GEOSENTINEL_API_TARGET || 'http://127.0.0.1:8001',
        changeOrigin: true,
      },
    },
  },
});
