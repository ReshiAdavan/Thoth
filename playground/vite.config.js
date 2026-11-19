import { defineConfig } from "vite";
const proxy = { "/api": { target: "http://127.0.0.1:8765" } };
export default defineConfig({
  worker: { format: "es" },
  server: { proxy },
  preview: { proxy },
});
