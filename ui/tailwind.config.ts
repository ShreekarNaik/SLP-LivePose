import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Stealth dark + teal palette
        bg: {
          0: "#05080d",     // deepest
          1: "#0a0f1a",     // base
          2: "#0f1724",     // panel
          3: "#152234",     // raised
        },
        ink: {
          0: "#e6f1f5",     // primary text
          1: "#9bb0bd",     // secondary
          2: "#5e7383",     // muted
        },
        teal: {
          // bright cyan-teal accent
          400: "#22e6c8",
          500: "#00ffd5",
          600: "#06b8a5",
          700: "#0a7d72",
        },
        edge: "#1a3340",     // panel borders
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 12px rgba(0, 255, 213, 0.25)",
      },
    },
  },
  plugins: [],
} satisfies Config;
