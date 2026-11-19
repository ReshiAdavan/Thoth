import { defineConfig } from "@playwright/test";
const port = process.env.THOTH_TEST_PORT || "5173";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: `http://127.0.0.1:${port}` },
  webServer: {
    command: process.env.THOTH_PREVIEW
      ? `npm run build && npx vite preview --host 127.0.0.1 --port ${port}`
      : `npm run dev -- --port ${port}`,
    url: `http://127.0.0.1:${port}`,
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
