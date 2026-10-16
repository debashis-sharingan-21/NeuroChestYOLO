import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#000000", // Pure Black
        accent: "#FFFFFF",  // White
        background: "#050505", // Deepest Black
        foreground: "#F5F5F5", // Off-white
        surface: "#111111", // Charcoal
        muted: "#888888", // Medium Gray
        border: "#222222", // Subtle Border
      },
      fontFamily: {
        sans: ["var(--font-sora)", "ui-sans-serif", "system-ui"],
        display: ["var(--font-sora)", "ui-sans-serif", "system-ui"],
        drama: ["var(--font-instrument-serif)", "serif"],
        mono: ["var(--font-fira-code)", "monospace"],
      },
      borderRadius: {
        "3xl": "2rem",
        "4xl": "3rem",
      },
      animation: {
        "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
};
export default config;
