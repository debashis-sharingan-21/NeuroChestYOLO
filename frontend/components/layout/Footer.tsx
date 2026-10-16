"use client";
import { usePageTransition } from "./PageTransition";

export default function Footer() {
  return (
    <footer className="relative z-50 bg-primary">
      <div className="max-w-7xl mx-auto mt-20 pt-8 border-t border-white flex flex-col md:flex-row justify-center text-[10px] font-mono text-white uppercase tracking-[0.2em] pb-10 px-10">
        <p>© 2026 Neuro Syndication Group. All rights reserved.</p>
      </div>
    </footer>
  );
}
