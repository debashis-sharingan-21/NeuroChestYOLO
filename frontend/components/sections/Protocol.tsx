"use client";

import { gsap, ScrollTrigger } from "@/lib/gsap";
import { useGSAP } from "@gsap/react";
import { useRef, useState, useEffect } from "react";
import { cn } from "@/lib/utils";

const PROTOCOLS = [
  {
    num: "01",
    title: "Neural Classification",
    description:
      "Deep ensemble analysis using custom DenseNet architecture. 15 simultaneous disease labels identified with high-precision confidence scoring.",
    animation: "motif",
  },
  {
    num: "02",
    title: "Explainable Evidence",
    description:
      "GradCAM localization maps provide visual accountability. We don't just predict; we show the precise anatomical features driving the AI's logic.",
    animation: "waveform",
  },
  {
    num: "03",
    title: "NLP Report Gen",
    description:
      "Coming Soon: Automated radiology report synthesis. Advanced language models translate visual findings into standardized medical narratives.",
    animation: "dataStream",
  },
  {
    num: "04",
    title: "Object Localization",
    description:
      "Multi-model verification with YOLOv8. Bounding box detections cross-reference visual patterns for verified clinical localization.",
    animation: "laser",
  },
];

export default function Protocol() {
  const containerRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    const mm = gsap.matchMedia();
    const cards = gsap.utils.toArray<HTMLElement>(".protocol-card");

    mm.add("(prefers-reduced-motion: no-preference)", () => {
        cards.forEach((card, i) => {
          ScrollTrigger.create({
            trigger: card,
            start: "top top",
            pin: true,
            pinSpacing: false,
            end: () => `+=${window.innerHeight * 0.5}`,
          });

          if (i < cards.length - 1) {
            gsap.to(card, {
              scale: 0.9,
              filter: "blur(20px)",
              opacity: 0.5,
              ease: "none",
              scrollTrigger: {
                trigger: cards[i + 1],
                start: "top bottom",
                end: "top top",
                scrub: true,
              },
            });
          }
        });
    });

    mm.add("(prefers-reduced-motion: reduce)", () => {
        // Just stack normally without scaling/blurring
    });
  }, { scope: containerRef });

  return (
    <div id="protocol" ref={containerRef} className="relative bg-black">
      {PROTOCOLS.map((protocol, i) => (
        <section
          key={i}
          className="protocol-card relative h-screen w-full flex items-center justify-center p-8 md:p-20 overflow-hidden bg-primary"
        >
          <div className="absolute inset-0 z-0 opacity-40 flex items-center justify-center">
            {protocol.animation === "motif" && <MotifAnimation />}
            {protocol.animation === "waveform" && <WaveformAnimation />}
            {protocol.animation === "laser" && <LaserAnimation />}
            {protocol.animation === "dataStream" && <DataStreamAnimation />}
          </div>

          <div className="relative z-10 max-w-6xl w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div
              className={cn(
                "space-y-6",
                i % 2 !== 0 && "md:col-start-2 md:pl-12",
              )}
            >
              <span className="font-mono text-accent text-lg font-bold tracking-widest">
                {protocol.num} //
              </span>
              <h3 className="font-sans text-4xl md:text-6xl font-bold text-white tracking-tight">
                {protocol.title}
              </h3>
              <p className="font-sans text-lg text-white/60 leading-relaxed max-w-md">
                {protocol.description}
              </p>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}

function MotifAnimation() {
  const ref = useRef<SVGSVGElement>(null);
  useGSAP(() => {
    gsap.to(ref.current, {
      rotation: 360,
      duration: 20,
      repeat: -1,
      ease: "none",
    });
  });

  return (
    <svg ref={ref} width="600" height="600" viewBox="0 0 100 100" className="text-accent/30">
      <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 2" />
      <circle cx="50" cy="50" r="30" fill="none" stroke="currentColor" strokeWidth="1" />
      <path d="M50 5 L50 95 M5 50 L95 50" stroke="currentColor" strokeWidth="0.2" />
      {[...Array(12)].map((_, i) => (
        <rect
          key={i}
          x="48"
          y="0"
          width="4"
          height="10"
          fill="currentColor"
          transform={`rotate(${i * 30} 50 50)`}
          className="opacity-50"
        />
      ))}
    </svg>
  );
}

function DataStreamAnimation() {
  const svgRef = useRef<SVGSVGElement>(null);
  const ringRef = useRef<SVGCircleElement>(null);
  const nodesRef = useRef<(SVGCircleElement | null)[]>([]);

  useGSAP(() => {
    const mm = gsap.matchMedia();

    mm.add("(prefers-reduced-motion: no-preference)", () => {
      // Rotate the entire data ring
      gsap.to(svgRef.current, {
        rotation: -300,
        duration: 70,
        repeat: -1,
        ease: "none",
      });

      // Pulse the connection nodes
      nodesRef.current.forEach((node, i) => {
        if (!node) return;
        gsap.to(node, {
          opacity: 0.6,
          r: 1.5,
          duration: 1 + (i % 3) * 0.5,
          repeat: -1,
          yoyo: true,
          ease: "sine.inOut",
          delay: i * 0.3,
        });
      });

      // Animate the dashed ring
      gsap.to(ringRef.current, {
        strokeDashoffset: 100,
        duration: 100,
        repeat: -1,
        ease: "none",
      });
    });

    mm.add("(prefers-reduced-motion: reduce)", () => {
      gsap.set(nodesRef.current, { opacity: 0.5, r: 1 });
    });
  }, { scope: svgRef });

  return (
    <svg ref={svgRef} width="600" height="600" viewBox="0 0 100 100" className="text-accent/20">
      <circle 
        ref={ringRef}
        cx="50" 
        cy="50" 
        r="50" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="0.5" 
        strokeDasharray="4 4" 
      />
      <circle cx="50" cy="50" r="25" fill="none" stroke="currentColor" strokeWidth="0.2" opacity="0.9" />
      
      {/* Symmetric connection nodes */}
      {[...Array(16)].map((_, i) => {
        const angle = (i * 360) / 16;
        const x = 50 + 40 * Math.cos((angle * Math.PI) / 180);
        const y = 50 + 40 * Math.sin((angle * Math.PI) / 180);
        return (
          <g key={i}>
            <line 
              x1="50" y1="50" x2={x} y2={y} 
              stroke="currentColor" strokeWidth="0.1" opacity="0.8" 
            />
            <circle
              ref={(el) => { nodesRef.current[i] = el; }}
              cx={x}
              cy={y}
              r="1"
              fill="currentColor"
              className="opacity-80"
            />
          </g>
        );
      })}
      <circle cx="50" cy="50" r="5" fill="currentColor" className="opacity-60" />
    </svg>
  );
}

function LaserAnimation() {
  const svgRef = useRef<SVGSVGElement>(null);
  const lineRef = useRef<SVGLineElement>(null);
  const rectRef = useRef<SVGRectElement>(null);
  const [cells, setCells] = useState<{ x: number; y: number }[]>([]);

  useEffect(() => {
    const newCells = [];
    for (let x = 0; x < 10; x++) {
      for (let y = 0; y < 10; y++) {
        newCells.push({ x: x * 10, y: y * 10 });
      }
    }
    setCells(newCells);
  }, []);

  useGSAP(() => {
    const mm = gsap.matchMedia();

    mm.add("(prefers-reduced-motion: no-preference)", () => {
      // Sync the rect and line perfectly
      // rect height is 4, so line should be at y + 2
      gsap.to(rectRef.current, {
        attr: { y: 96 },
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      });
      
      gsap.to(lineRef.current, {
        attr: { y1: 98, y2: 98 },
        duration: 3,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
      });
    });

    mm.add("(prefers-reduced-motion: reduce)", () => {
      gsap.set(rectRef.current, { attr: { y: 48 } });
      gsap.set(lineRef.current, { attr: { y1: 50, y2: 50 } });
    });
  }, { scope: svgRef });

  return (
    <svg ref={svgRef} width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" className="text-accent/10">
      <defs>
        <linearGradient id="scanGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0" />
          <stop offset="50%" stopColor="currentColor" stopOpacity="1" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      
      {/* Grid */}
      <g>
        {cells.map((cell, i) => (
          <rect
            key={i}
            x={cell.x}
            y={cell.y}
            width="0.3"
            height="0.3"
            fill="currentColor"
            opacity={0.15}
          />
        ))}
      </g>

      {/* Scanning Bar - Narrower and aligned */}
      <rect
        ref={rectRef}
        x="0"
        y="0"
        width="100"
        height="4"
        fill="url(#scanGradient)"
        className="text-accent/30"
      />
      
      {/* Laser Core - Perfectly centered in rect */}
      <line
        ref={lineRef}
        x1="0"
        y1="2"
        x2="100"
        y2="2"
        stroke="white"
        strokeWidth="0.2"
        className="opacity-60"
      />
    </svg>
  );
}

function WaveformAnimation() {
  const pathsRef = useRef<(SVGPathElement | null)[]>([]);

  useGSAP(() => {
    pathsRef.current.forEach((path, i) => {
      if (!path) return;
      gsap.to(path, {
        strokeDashoffset: 0,
        duration: 4 + i,
        repeat: -1,
        ease: "sine.inOut",
        yoyo: true,
      });
    });
  });

  const generatePath = (offset: number) => {
    const points = [];
    const width = 800;
    const height = 200;
    const centerY = height / 2;
    const amplitude = 50;
    const frequency = 0.015;
    
    // Symmetric wave generation with higher resolution
    for (let x = 0; x <= width; x += 5) {
      // Use a bell-curve envelope for symmetry and smoothness at edges
      const envelope = Math.sin((x / width) * Math.PI);
      const y = centerY + Math.sin((x + offset) * frequency) * amplitude * envelope;
      // Round to 3 decimal places to avoid hydration mismatch from float precision
      points.push(`${x},${y.toFixed(3)}`);
    }
    return `M${points.join(" L")}`;
  };

  return (
    <svg width="100%" height="100%" viewBox="0 0 800 200" preserveAspectRatio="none" className="text-accent/40 scale-110">
      {[...Array(3)].map((_, i) => (
        <path
          key={i}
          ref={(el) => { pathsRef.current[i] = el; }}
          d={generatePath(i * 150)}
          fill="none"
          stroke="currentColor"
          strokeWidth={0.5 + i * 0.5}
          strokeDasharray="3000"
          strokeDashoffset="3000"
          opacity={0.7 - i * 0.15}
        />
      ))}
    </svg>
  );
}
