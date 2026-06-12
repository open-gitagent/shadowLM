import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build lands inside the Python package: the wheel ships the compiled UI and
// `shadowlm serve` serves it — users never need node. `npm run dev` proxies
// API calls to a locally running `shadowlm serve` for live frontend work.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  build: {
    outDir: "../shadowlm/_static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/v1": "http://127.0.0.1:8329",
      "/logo.png": "http://127.0.0.1:8329",
    },
  },
});
