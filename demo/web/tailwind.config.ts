import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(220 20% 6%)",
        surface: "hsl(220 20% 10%)",
        "surface-elevated": "hsl(220 18% 13%)",
        "surface-border": "hsl(220 15% 20%)",
        fg: "hsl(60 10% 93%)",
        "fg-muted": "hsl(220 10% 65%)",
        "fg-faint": "hsl(220 10% 45%)",
        accent: "hsl(160 84% 39%)",
        "accent-secondary": "hsl(200 80% 55%)",
        line: "hsl(220 15% 20%)",
      },
      borderRadius: {
        card: "0.75rem",
      },
      fontFamily: {
        display: [
          "Space Grotesk",
          "system-ui",
          "-apple-system",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
