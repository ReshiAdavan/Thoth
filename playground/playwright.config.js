import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:5173" },
  webServer: {
    command: process.env.THOTH_PREVIEW
      ? "npm run build && npx vite preview --host 127.0.0.1 --port 5173"
      : "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI && !process.env.THOTH_PREVIEW,
  },
  projects: [
    { name: "desktop", use: { browserName: "chromium" } },
    {
      name: "mobile",
      use: { browserName: "chromium", viewport: { width: 390, height: 844 } },
    },
  ],
});
