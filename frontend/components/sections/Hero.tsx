"use client";

import { gsap } from "@/lib/gsap";
import { useGSAP } from "@gsap/react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

const IMAGES = [
  "https://images.unsplash.com/photo-1614850715649-1d0106293bd1?q=80&w=2070&auto=format&fit=crop", // Bioluminescence
  "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=2070&auto=format&fit=crop", // Neon
  "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=1974&auto=format&fit=crop", // Tech
  "https://images.unsplash.com/photo-1531297484001-80022131f5a1?q=80&w=2020&auto=format&fit=crop", // Circuit
  "/images/philo.avif", // Abstract
];

export const INTRO_END_DELAY_SEC = 0.35 + (IMAGES.length - 1) * 0.25 + 1 + 1;

import Image from "next/image";
import { useAuthStore } from "@/lib/store";
import { usePageTransition } from "@/components/layout/PageTransition";
import { createClient } from "@/lib/supabase/client";
import { ChevronDown } from "lucide-react";

export default function Hero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRefs = useRef<(HTMLImageElement | null)[]>([]);
  const radialRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLDivElement>(null);

  const user = useAuthStore((state) => state.user);
  const { navigateTo } = usePageTransition();
  const supabase = createClient();

  const handleCoreClick = async () => {
    if (user) {
      navigateTo("/dashboard");
    } else {
      navigateTo("/auth/signin");
    }
  };

  useEffect(() => {
    // Prevent browser from restoring scroll position on reload
    if ("scrollRestoration" in history) {
      history.scrollRestoration = "manual";
    }
    window.scrollTo(0, 0);

    // Lock lenis globally before it potentially mounts
    (window as any).lenisLocked = true;
    if ((window as any).lenis) (window as any).lenis.stop();

    // Pre-warm the backend (Hugging Face Space)
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    fetch(`${backendUrl}/health`, { method: "GET" }).catch(() => {});

    return () => {
      // Ensure scroll is unlocked when component unmounts
      document.body.style.overflow = "";
      (window as any).lenisLocked = false;
      if ((window as any).lenis) (window as any).lenis.start();
    };
  }, []);

  useGSAP(
    () => {
      const mm = gsap.matchMedia();

      mm.add("(prefers-reduced-motion: no-preference)", () => {
        const imgs = imgRefs.current.filter(Boolean);
        const tl = gsap.timeline({
          onStart: () => {
            document.body.style.overflow = "hidden";
            (window as any).lenisLocked = true;
            if ((window as any).lenis) (window as any).lenis.stop();
            window.scrollTo(0, 0);
          },
          onComplete: () => {
            document.body.style.overflow = "";
            (window as any).lenisLocked = false;
            if ((window as any).lenis) (window as any).lenis.start();
          },
        });

        // Entrance: staggered image reveal
        tl.to(imgs, {
          clipPath: "inset(0% 0% 0% 0%)",
          duration: 1,
          delay: 0.35,
          stagger: { each: 0.25, ease: "power1.out" },
        });

        // Expand container to full screen
        tl.to(containerRef.current, {
          width: "100%",
          height: "100dvh",
          maxWidth: "none",
          aspectRatio: "unset",
          margin: 0,
          duration: 1,
          ease: "power3.inOut",
        });

        // Reveal background radial and logo
        tl.to(
          [radialRef.current, logoRef.current],
          {
            opacity: 1,
            duration: 0.85,
            ease: "power2.out",
            stagger: 0.1,
          },
          ">",
        );

        // Headline and buttons reveal
        const revealedElements =
          contentRef.current?.querySelectorAll(".line-reveal");
        if (revealedElements) {
          tl.set(contentRef.current, { opacity: 1 }, "<");

          tl.to(
            revealedElements,
            {
              y: 0,
              opacity: 1,
              duration: 0.8,
              stagger: 0.1,
              ease: "power3.out",
            },
            "<0.2",
          );
        }

        // Scroll indicator reveal
        const scrollIndicator =
          contentRef.current?.querySelector(".scroll-indicator");
        if (scrollIndicator) {
          tl.to(
            scrollIndicator,
            {
              opacity: 1,
              duration: 0.5,
              ease: "power2.out",
            },
            ">",
          );
        }
      });

      // Fallback for reduced motion
      mm.add("(prefers-reduced-motion: reduce)", () => {
        gsap.set([radialRef.current, contentRef.current, logoRef.current], {
          opacity: 1,
        });
        gsap.set(containerRef.current, {
          width: "100%",
          height: "100dvh",
          maxWidth: "none",
          aspectRatio: "unset",
          margin: 0,
        });
        const imgs = imgRefs.current.filter(Boolean);
        gsap.set(imgs, { clipPath: "inset(0% 0% 0% 0%)" });
        const headlineLines =
          contentRef.current?.querySelectorAll(".line-reveal");
        if (headlineLines) gsap.set(headlineLines, { y: 0, opacity: 1 });
        const scrollIndicator =
          contentRef.current?.querySelector(".scroll-indicator");
        if (scrollIndicator) gsap.set(scrollIndicator, { opacity: 1 });
      });
    },
    { scope: containerRef },
  );

  return (
    <section className="relative h-[100dvh] w-full overflow-hidden bg-primary">
      {/* Logo Overlay */}
      {/* <div ref={logoRef} className="absolute top-8 left-8 z-50 opacity-0">
            <span className="font-mono text-xl tracking-tighter text-accent font-bold underline decoration-accent/30 underline-offset-4">NEURO</span>
        </div> */}

      <div className="absolute inset-0 flex items-center justify-center">
        <div
          ref={containerRef}
          className="relative aspect-video w-[min(88vw,28rem)] overflow-hidden md:w-[42vw] z-10"
        >
          {IMAGES.map((src, i) => (
            <div
              key={`${src}-${i}`}
              className="absolute inset-0 size-full"
              style={{ zIndex: i }}
            >
              <Image
                ref={(el) => {
                  // @ts-ignore - Image ref is tricky, but works for GSAP if accessed correctly
                  imgRefs.current[i] = el;
                }}
                src={src}
                alt=""
                fill
                sizes="(max-width: 768px) 88vw, 42vw"
                priority
                className="object-cover"
                style={{ clipPath: "inset(0% 0% 100% 0%)" }}
              />
            </div>
          ))}

          <div
            ref={radialRef}
            className="pointer-events-none absolute inset-0 z-10 opacity-0"
            style={{
              background:
                "radial-gradient(ellipse 100% 88% at 50% 42%, transparent 22%, rgba(0,0,0,0.6) 58%, rgba(0,0,0,0.95) 100%)",
            }}
            aria-hidden
          />

          {/* Hero Content */}
          <div
            ref={contentRef}
            className="absolute inset-0 z-20 flex flex-col justify-center p-8 md:p-20 opacity-0"
          >
            <div className="max-w-4xl space-y-4">
              <p className="font-mono text-white/70 text-sm tracking-[0.3em] uppercase line-reveal opacity-0 translate-y-8">
                // Advanced Neural Chest X-Ray Analysis
              </p>
              <h1 className="text-white">
                <span className="block text-4xl md:text-7xl font-bold tracking-tight line-reveal opacity-0 translate-y-8">
                  Clarity in Every
                </span>
                <span className="block text-6xl md:text-9xl font-drama italic text-white line-reveal mt-2 opacity-0 translate-y-8">
                  Insight.
                </span>
              </h1>
              <div className="mt-8 flex flex-wrap gap-4 pt-9 line-reveal opacity-0 translate-y-8">
                <button
                  onClick={handleCoreClick}
                  className="flex items-center justify-center whitespace-nowrap px-8 py-4 bg-accent text-black font-bold rounded-full text-sm tracking-widest uppercase hover:bg-accent/80 focus:ring-2 focus:ring-accent focus:outline-none pl-[2.1rem] min-w-[12rem]"
                >
                  {user ? "Dashboard" : "Begin Analysis"}
                </button>
                <button
                  onClick={() =>
                    window.open(
                      "https://www.linkedin.com/in/sagar-sarangi",
                      "_blank",
                    )
                  }
                  className="flex items-center justify-center whitespace-nowrap px-8 py-4 bg-white/10 border border-white/20 text-white font-bold rounded-full text-sm tracking-widest uppercase hover:bg-white/20 focus:ring-2 focus:ring-white focus:outline-none pl-[2.1rem]"
                >
                  Connect
                </button>
              </div>
            </div>

            {/* Scroll Indicator */}
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-0 scroll-indicator">
              <span className="text-[10px] font-mono tracking-[0.2em] text-white/50 uppercase">
                Scroll to Explore
              </span>
              <ChevronDown className="w-4 h-4 text-white/50" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
