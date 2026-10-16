"use client";
import { usePageTransition } from "./PageTransition";

export default function Footer() {
  return (
    <footer className="relative z-50 bg-primary">
      <div className="max-w-7xl mx-auto mt-20 pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4 text-[10px] font-mono text-white/50 uppercase tracking-[0.2em] pb-10 px-10">
        <p>© 2026 Neuro Syndication Group. All rights reserved.</p>
        <a
          href="https://www.linkedin.com/in/debashis-mohapatra-113523302/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-white/60 hover:text-accent transition-colors"
        >
          Connect on LinkedIn // Debashis Mohapatra
        </a>
      </div>
    </footer>
  );
}
