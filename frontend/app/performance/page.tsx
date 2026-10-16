"use client";

import { useRef, useState } from "react";
import { gsap } from "@/lib/gsap";
import { useGSAP } from "@gsap/react";
import Image from "next/image";
import { X } from "lucide-react";

const BENCHMARKS = [
  {
    title: "AUC-ROC Analysis",
    description:
      "System-wide multi-label classification performance across diagnostic categories. Demonstrating robust discriminatory power with a mean test AUC across conditions.",
    image: "/images/nih/AUC_ROC.png",
    tag: "Metrics // 01",
  },
  {
    title: "Validation Benchmarks",
    description:
      "Real-time comparison of model validation AUC against established NIH baseline benchmarks. Consistent performance across tiered condition complexity.",
    image: "/images/nih/Mean_val_auc_compare_benchmark.png",
    tag: "Metrics // 02",
  },
  {
    title: "Probability Overlap",
    description:
      "Statistical overlap analysis of positive and negative findings. Calibrated probability thresholds ensure clinical reliability and minimized uncertainty.",
    image: "/images/nih/Overlap.png",
    tag: "Metrics // 03",
  },
];

const CHESTXDET_BENCHMARKS = [
  {
    title: "mAP@50 Benchmark",
    description:
      "Comparison of Mean Average Precision at 50% IoU against the Faster R-CNN baseline. Our model demonstrates superior localization precision across anatomical findings.",
    image: "/images/chestxdet/benchmark_ap50_vs_yolo.png",
    tag: "Metrics // 01",
  },
  {
    title: "Recall Benchmark",
    description:
      "Analysis of recall performance at a fixed false positive rate (0.1) compared to Faster R-CNN benchmarks, ensuring high sensitivity for critical diagnostic detection.",
    image: "/images/chestxdet/benchmark_recall_fp01_vs_yolo.png",
    tag: "Metrics // 02",
  },
  {
    title: "PR Curve Analysis",
    description:
      "Precision-Recall trade-off visualization, highlighting the model's ability to maintain high precision while maximizing clinical finding coverage.",
    image: "/images/chestxdet/BoxPR_curve.png",
    tag: "Metrics // 03",
  },
  {
    title: "F1 Curve Evaluation",
    description:
      "F1-score evaluation across varied confidence thresholds, identifying the optimal operating point for balanced diagnostic accuracy.",
    image: "/images/chestxdet/BoxF1_curve.png",
    tag: "Metrics // 04",
  },
  {
    title: "Confusion Matrix",
    description:
      "Normalized classification performance for localized regions, validating the model's discriminatory power across specific abnormality classes.",
    image: "/images/chestxdet/confusion_matrix_normalized.png",
    tag: "Metrics // 05",
  },
  {
    title: "Training Dynamics",
    description:
      "Optimization dynamics showing the convergence of localization and classification losses, ensuring robust and stable model performance.",
    image: "/images/chestxdet/training_curves_v3.png",
    tag: "Metrics // 06",
  },
];

export default function Performance() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedImage, setSelectedImage] = useState<{
    src: string;
    title: string;
  } | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      gsap.fromTo(
        ".fade-up",
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 1,
          stagger: 0.15,
          ease: "power3.out",
          delay: 0.4,
        },
      );
    },
    { scope: containerRef },
  );

  useGSAP(() => {
    if (selectedImage) {
      gsap.fromTo(
        modalRef.current,
        { opacity: 0, scale: 0.95 },
        { opacity: 1, scale: 1, duration: 0.4, ease: "power3.out" },
      );
      // Prevent scroll when modal is open
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
  }, [selectedImage]);

  const handleClose = () => {
    gsap.to(modalRef.current, {
      opacity: 0,
      scale: 0.95,
      duration: 0.3,
      ease: "power3.in",
      onComplete: () => setSelectedImage(null),
    });
  };

  return (
    <div ref={containerRef} className="min-h-screen bg-primary">
      {/* Hero Section - Cinematic */}
      <section className="relative h-screen w-full px-8 md:px-20 overflow-hidden">
        <div className="absolute inset-0 z-0">
          <Image
            src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop"
            alt="Medical Benchmarks"
            fill
            sizes="100vw"
            priority
            className="object-cover opacity-30 grayscale"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-primary via-primary/80 to-transparent" />
        </div>

        <div className="relative z-10 h-[80vh] w-full flex items-end pb-20 max-w-6xl">
          <div className="space-y-4">
            <p className="font-mono text-accent text-sm tracking-[0.5em] uppercase opacity-60 fade-up">
              // Technical Validation
            </p>
            <h1 className="text-white text-7xl md:text-[10rem] font-bold tracking-tighter leading-[0.85] fade-up">
              Clinical <br />
              <span className="font-drama italic text-white underline decoration-white/20">
                Precision.
              </span>
            </h1>
          </div>
        </div>
      </section>

      {/* NIH Model Section */}
      <section className="py-32 px-8 md:px-20 max-w-[1600px] mx-auto">
        <div className="text-center mb-10 space-y-4 -mb-">
          <p className="font-mono text-accent text-[10px] tracking-[0.5em] uppercase opacity-40 fade-up">
            // Diagnostic Ensemble
          </p>
          <h2 className="text-white text-4xl md:text-6xl font-bold tracking-tight fade-up">
            NIH Model
          </h2>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {BENCHMARKS.map((item, i) => (
            <div
              key={i}
              className="metric-card fade-up group relative bg-[#121212] border border-white/10 rounded-[2.5rem] overflow-hidden hover:bg-[#1a1a1a] transition-colors duration-500 hover:border-accent/30 shadow-2xl"
            >
              {/* Image Container - CLICKABLE */}
              <button
                onClick={() =>
                  setSelectedImage({ src: item.image, title: item.title })
                }
                className="relative aspect-video w-full overflow-hidden bg-black/40 p-6 cursor-zoom-in block"
              >
                <div className="relative h-full w-full rounded-2xl overflow-hidden shadow-xl">
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                    className="object-contain transition-transform duration-700 group-hover:scale-[1.05]"
                  />
                </div>
                <div className="absolute inset-0 bg-accent/0 group-hover:bg-accent/5 transition-colors duration-500 flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <span className="bg-black/60 backdrop-blur-md px-4 py-2 rounded-full text-white text-[10px] font-mono tracking-widest border border-white/10 uppercase">
                    Click to Zoom
                  </span>
                </div>
              </button>

              {/* Content Container */}
              <div className="p-10 space-y-6 text-left">
                <div className="space-y-2">
                  <span className="font-mono text-accent text-[10px] tracking-[0.3em] uppercase block opacity-80">
                    {item.tag}
                  </span>
                  <h2 className="text-white text-3xl font-bold tracking-tight">
                    {item.title}
                  </h2>
                </div>

                <p className="text-white/70 leading-relaxed text-sm lg:text-base font-medium">
                  {item.description}
                </p>

                <div className="pt-4 flex items-center gap-4">
                  <div className="h-px flex-1 bg-white/10" />
                  <span className="font-mono text-[10px] text-accent/50 uppercase tracking-widest font-bold">
                    Verified
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ChestXDet Model Section */}
      <section className="py-32 px-8 md:px-20 max-w-[1600px] mx-auto border-t border-white/5">
        <div className="text-center mb-10 space-y-4">
          <p className="font-mono text-accent text-[10px] tracking-[0.5em] uppercase opacity-40 fade-up">
            // Object Localization
          </p>
          <h2 className="text-white text-4xl md:text-6xl font-bold tracking-tight fade-up">
            ChestXDet Model
          </h2>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {CHESTXDET_BENCHMARKS.map((item, i) => (
            <div
              key={i}
              className="metric-card fade-up group relative bg-[#121212] border border-white/10 rounded-[2.5rem] overflow-hidden hover:bg-[#1a1a1a] transition-colors duration-500 hover:border-accent/30 shadow-2xl"
            >
              {/* Image Container - CLICKABLE */}
              <button
                onClick={() =>
                  setSelectedImage({ src: item.image, title: item.title })
                }
                className="relative aspect-video w-full overflow-hidden bg-black/40 p-6 cursor-zoom-in block"
              >
                <div className="relative h-full w-full rounded-2xl overflow-hidden shadow-xl">
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
                    className="object-contain transition-transform duration-700 group-hover:scale-[1.05]"
                  />
                </div>
                <div className="absolute inset-0 bg-accent/0 group-hover:bg-accent/5 transition-colors duration-500 flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <span className="bg-black/60 backdrop-blur-md px-4 py-2 rounded-full text-white text-[10px] font-mono tracking-widest border border-white/10 uppercase">
                    Click to Zoom
                  </span>
                </div>
              </button>

              {/* Content Container */}
              <div className="p-10 space-y-6 text-left">
                <div className="space-y-2">
                  <span className="font-mono text-accent text-[10px] tracking-[0.3em] uppercase block opacity-80">
                    {item.tag}
                  </span>
                  <h2 className="text-white text-3xl font-bold tracking-tight">
                    {item.title}
                  </h2>
                </div>

                <p className="text-white/70 leading-relaxed text-sm lg:text-base font-medium">
                  {item.description}
                </p>

                <div className="pt-4 flex items-center gap-4">
                  <div className="h-px flex-1 bg-white/10" />
                  <span className="font-mono text-[10px] text-accent/50 uppercase tracking-widest font-bold">
                    Verified
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Lightbox Modal */}
      {selectedImage && (
        <div
          ref={modalRef}
          className="fixed inset-0 z-[200] flex items-center justify-center p-4 md:p-20 bg-black/95 backdrop-blur-xl"
          onClick={handleClose}
        >
          <button
            className="absolute top-8 right-8 text-white/40 hover:text-white transition-colors"
            onClick={handleClose}
          >
            <X size={32} />
          </button>

          <div className="relative w-full h-full max-w-7xl flex flex-col items-center justify-center gap-8">
            <div className="relative w-full flex-1 rounded-3xl overflow-hidden border border-white/10 bg-black/50 p-4">
              <Image
                src={selectedImage.src}
                alt={selectedImage.title}
                fill
                sizes="100vw"
                className="object-contain"
              />
            </div>
            <div className="text-center space-y-2">
              <h3 className="text-white text-2xl font-bold tracking-tight">
                {selectedImage.title}
              </h3>
              <p className="font-mono text-accent/60 text-xs tracking-widest uppercase">
                Clinical Performance Metric
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Footer Visual */}
      <section className="py-20 flex justify-center opacity-10">
        <div className="h-px w-64 bg-gradient-to-r from-transparent via-white to-transparent" />
      </section>
    </div>
  );
}
