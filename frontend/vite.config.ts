import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Build lands inside the Python package: the wheel ships the compiled UI and
// `shadowlm serve` serves it — users never need node. `npm run dev` proxies
// API calls to a locally running `shadowlm serve` for live frontend work.
// In dev the page comes from Vite, not `shadowlm serve`, so name the origins
// that may frame it here (the same built-ins + SHADOWLM_FRAME_ANCESTORS as
// serve.py's frame_ancestors); src/lib/embed.ts reads the tag.
const embedParents = [
  "https://dev.opencontroller.sh", "http://localhost:*", "https://localhost:*",
  ...(process.env.SHADOWLM_FRAME_ANCESTORS ?? "").split(/[\s,]+/).filter((o) => /^https?:\/\/[^/\s]+$/.test(o)),
];
const embedMeta = {
  name: "embed-parents",
  apply: "serve" as const,
  transformIndexHtml: () => [{
    tag: "meta", injectTo: "head" as const,
    attrs: { name: "shadowlm-embed-parents", content: [...new Set(embedParents)].join(" ") },
  }],
};

export default defineConfig({
  plugins: [react(), tailwindcss(), embedMeta],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  base: "./",
  build: {
    outDir: "../shadowlm/_static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // `shadowlm serve --dev` sets SHADOWLM_DEV_API to its own port.
      "/v1": process.env.SHADOWLM_DEV_API || "http://127.0.0.1:8329",
      "/logo.png": process.env.SHADOWLM_DEV_API || "http://127.0.0.1:8329",
      "/logo1.png": process.env.SHADOWLM_DEV_API || "http://127.0.0.1:8329",
    },
  },
});
