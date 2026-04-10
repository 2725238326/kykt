import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const runtimeEnv =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};
const host = runtimeEnv.TAURI_DEV_HOST;

export default defineConfig({
  clearScreen: false,
  plugins: [react()],
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  server: {
    port: 5173,
    strictPort: true,
    host: host || "127.0.0.1",
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421
        }
      : undefined,
    watch: {
      ignored: ["**/src-tauri/**"]
    },
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/local_jobs": "http://127.0.0.1:8765"
    }
  },
  build: {
    target: runtimeEnv.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    minify: !runtimeEnv.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!runtimeEnv.TAURI_ENV_DEBUG
  }
});
