"use client";

import Lenis from "lenis";
import { useEffect, useState } from "react";
import { gsap } from "./gsap";

export function useLenis() {
  const [lenis, setLenis] = useState<Lenis | null>(null);

  useEffect(() => {
    const instance = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });

    // Expose globally to allow other components to pause/resume scrolling
    (window as any).lenis = instance;

    if ((window as any).lenisLocked) {
      instance.stop();
    }

    setLenis(instance);

    const tick = (time: number) => {
      instance.raf(time * 1000);
    };

    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);

    return () => {
      instance.destroy();
      gsap.ticker.remove(tick);
    };
  }, []);

  return lenis;
}
