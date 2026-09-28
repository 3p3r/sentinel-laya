import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tailwindcss()],
  resolve: {
    conditions: ["browser", "import", "module", "default"],
  },
  build: {
    outDir: "dist",
    target: "esnext",
  },
});
