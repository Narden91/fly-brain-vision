import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-only: forward API calls to the FastAPI backend (`uvicorn backend.main:app`).
    // The production build is served BY that same backend, so this proxy is never
    // needed there — the frontend always just fetches relative "/api/...".
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
