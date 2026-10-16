"use client";

import { useState, useEffect } from "react";
import { 
  ChevronLeft, 
  ChevronRight, 
  Maximize2, 
  X, 
  Scan, 
  ImageIcon,
  Loader2
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SampleGalleryProps {
  onSelectImage: (file: File) => void;
}

export function SampleGallery({ onSelectImage }: SampleGalleryProps) {
  const [images, setImages] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Lock body scroll when modal is open
  useEffect(() => {
    if (isModalOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isModalOpen]);

  useEffect(() => {
    async function fetchImages() {
      try {
        const response = await fetch("/api/sample-images");
        const data = await response.json();
        if (data.images) {
          setImages(data.images);
        }
      } catch (err) {
        console.error("Failed to fetch sample images", err);
      } finally {
        setIsLoading(false);
      }
    }
    fetchImages();
  }, []);

  const handleNext = (e: React.MouseEvent) => {
    e.stopPropagation();
    setCurrentIndex((prev) => (prev + 1) % images.length);
  };

  const handlePrev = (e: React.MouseEvent) => {
    e.stopPropagation();
    setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const handleSelect = async (imageName: string) => {
    try {
      const response = await fetch(`/testing_images/${imageName}`);
      const blob = await response.blob();
      const file = new File([blob], imageName, { type: blob.type });
      onSelectImage(file);
      setIsModalOpen(false);
    } catch (err) {
      console.error("Error selecting image:", err);
    }
  };

  if (isLoading) {
    return (
      <div className="py-12 flex items-center justify-center">
        <p className="text-[10px] font-mono text-white/20 uppercase tracking-[0.2em]">Synchronizing Archive...</p>
      </div>
    );
  }

  if (images.length === 0) return null;

  return (
    <div className="space-y-6">
      {/* Preview Block - Styled as a Dropzone Mirror */}
      <div 
        onClick={() => setIsModalOpen(true)}
        className="group relative h-[500px] border-2 border-dashed border-white/10 hover:border-accent/40 rounded-container flex flex-col items-center justify-center transition-all cursor-pointer bg-surface/20 overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-b from-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
        <div className="relative flex flex-col items-center text-center px-8">
          <div className="w-16 h-16 rounded-full bg-accent/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
            <ImageIcon className="text-accent h-8 w-8" />
          </div>
          <h3 className="text-xl font-bold text-white mb-2 tracking-tight">Reference Library</h3>
          <p className="text-white/40 text-sm max-w-xs font-mono uppercase tracking-wider leading-relaxed">
            Access Clinical Archive to select from Sample Reference Units
          </p>
        </div>
        
        {/* Dynamic unit count hint */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2">
          <div className="h-1 w-1 rounded-full bg-accent" />
          <span className="text-[10px] font-mono text-white/50 uppercase tracking-widest">{images.length} Units Online</span>
        </div>
      </div>

      {/* Modal / Zoomed Gallery */}
      {isModalOpen && (
        <div 
          data-lenis-prevent
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 md:p-12"
        >
          <div 
            className="absolute inset-0 bg-black/90 backdrop-blur-xl animate-in fade-in duration-500" 
            onClick={() => setIsModalOpen(false)}
          />
          
          <div className="relative w-full max-w-6xl bg-surface/50 border border-white/10 rounded-[32px] overflow-hidden flex flex-col h-full max-h-[85vh] animate-in zoom-in-95 duration-500">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-black/20">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-accent/5 flex items-center justify-center">
                  <Scan className="text-accent h-5 w-5" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white tracking-tight">Clinical Reference Archive</h2>
                  <p className="text-[10px] font-mono text-white/40 uppercase tracking-widest">Select target image for inference</p>
                </div>
              </div>
              <button 
                onClick={() => setIsModalOpen(false)}
                className="p-3 bg-white/5 hover:bg-red-500/20 text-white/60 hover:text-red-400 rounded-full transition-all"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Modal Content - Grid */}
            <div 
              data-lenis-prevent
              className="flex-1 overflow-y-auto p-8 custom-scrollbar"
            >
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                {images.map((img, idx) => (
                  <div 
                    key={idx}
                    onClick={() => handleSelect(img)}
                    className="group relative aspect-square bg-black/40 rounded-2xl border border-white/5 overflow-hidden cursor-pointer hover:border-accent/40 transition-colors hover:scale-[1.02]"
                  >
                    <img 
                      src={`/testing_images/${img}`} 
                      alt={img}
                      className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity duration-500"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-4 flex flex-col justify-end">
                      <p className="text-[8px] font-mono text-accent uppercase tracking-widest mb-0.5">Reference ID</p>
                      <p className="text-white text-[10px] font-bold truncate">{img}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-8 py-4 border-t border-white/5 bg-black/40 flex items-center justify-between">
              <p className="text-[9px] font-mono text-white/40 uppercase tracking-[0.2em]">
                Authorized Diagnostic Assets // v3.1.0
              </p>
              <div className="flex items-center gap-2">
                <div className="h-1 w-1 rounded-full bg-accent animate-pulse" />
                <span className="text-[9px] font-mono text-accent uppercase tracking-widest">Ready for Injection</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
