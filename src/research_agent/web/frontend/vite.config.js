import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built into ./dist, which FastAPI serves. During `npm run dev`, /api and /auth
// are proxied to the FastAPI server on :8800.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8800",
      "/auth": "http://localhost:8800",
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
