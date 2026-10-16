"use client";

import { gsap, ScrollTrigger } from "@/lib/gsap";
import { useGSAP } from "@gsap/react";
import { useRef } from "react";

export default function Philosophy() {
  const containerRef = useRef<HTMLDivElement>(null);
  const text1Ref = useRef<HTMLParagraphElement>(null);
  const text2Ref = useRef<HTMLHeadingElement>(null);

  useGSAP(() => {
    const mm = gsap.matchMedia();

    mm.add("(prefers-reduced-motion: no-preference)", () => {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: containerRef.current,
          start: "top 60%",
          toggleActions: "play none none reverse",
        },
      });

      tl.fromTo(
        text1Ref.current,
        { y: 30, opacity: 0 },
        { y: 0, opacity: 0.5, duration: 1, ease: "power3.out" }
      );

      tl.fromTo(
        text2Ref.current,
        { y: 50, opacity: 0 },
        { y: 0, opacity: 1, duration: 1.2, ease: "power3.out" },
        "-=0.6"
      );
    });

    mm.add("(prefers-reduced-motion: reduce)", () => {
      gsap.set(text1Ref.current, { y: 0, opacity: 0.5 });
      gsap.set(text2Ref.current, { y: 0, opacity: 1 });
    });
  }, { scope: containerRef });

  return (
    <section id="about" ref={containerRef} className="relative min-h-screen w-full bg-primary flex flex-col justify-center items-center px-8 md:px-20 py-32 overflow-hidden">
        {/* Parallax Background Texture */}
        <div className="absolute inset-0 z-0 opacity-20 pointer-events-none">
            <img 
                src="/images/ppl.jpg" 
                alt="" 
                className="w-full h-full object-cover grayscale"
            />
        </div>

        <div className="relative z-10 max-w-6xl w-full space-y-12">
            <p ref={text1Ref} className="font-sans text-xl md:text-3xl text-white opacity-0 text-center md:text-left">
                Most diagnostic AI systems focus on: <span className="underline underline-offset-8 decoration-white/10">Black-box prediction.</span>
            </p>
            
            <h2 ref={text2Ref} className="font-sans text-4xl md:text-7xl lg:text-8xl font-bold text-white leading-tigh text-center md:text-left opacity-0">
                We focus on: <span className="font-drama italic text-white block md:inline mt-4 md:mt-0">Visual Evidence.</span>
            </h2>
        </div>

        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 flex flex-col items-center gap-4 opacity-30">
            <div className="h-20 w-px bg-white animate-pulse" />
            <span className="font-mono text-[10px] tracking-[0.5em] uppercase text-white">Vision // 01</span>
        </div>
    </section>
  );
}
