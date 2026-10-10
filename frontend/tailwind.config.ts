import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0A0B0E",
        surface: "#0F1115",
        raised: "#15181E",
        line: { DEFAULT: "#1C1F26", strong: "#282C35" },
        ink: { DEFAULT: "#ECEDEF", 2: "#9BA0AB", 3: "#6B7080" },
        accent: { DEFAULT: "#3E7BE0", hover: "#4F8AEC", press: "#356BC6", ink: "#8FB1F2" },
        risk: { high: "#E5484D", medium: "#DE9B2F", review: "#8B909C", ok: "#3FA678" },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontSize: {
        display: ["1.75rem", { lineHeight: "2.125rem", letterSpacing: "-0.02em", fontWeight: "600" }],
        title: ["1.0625rem", { lineHeight: "1.5rem", letterSpacing: "-0.02em", fontWeight: "600" }],
        metric: ["1.5rem", { lineHeight: "2rem", letterSpacing: "-0.02em", fontWeight: "500" }],
        body: ["0.875rem", { lineHeight: "1.375rem" }],
        small: ["0.8125rem", { lineHeight: "1.25rem" }],
        caption: ["0.75rem", { lineHeight: "1rem" }],
      },
      keyframes: {
        "slide-in": { from: { transform: "translateX(24px)", opacity: "0" }, to: { transform: "translateX(0)", opacity: "1" } },
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
      },
      animation: {
        "slide-in": "slide-in 180ms cubic-bezier(0.2, 0.8, 0.2, 1)",
        "fade-in": "fade-in 150ms ease-out",
      },
    },
  },
  plugins: [],
} satisfies Config;
