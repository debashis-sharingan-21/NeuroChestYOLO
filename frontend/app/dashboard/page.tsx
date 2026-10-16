"use client";

import { useState, useRef, useEffect } from "react";
import { 
  Activity, Zap, 
  Upload, Scan, Brain, Crosshair, AlertCircle, 
  CheckCircle2, Info, Loader2, ArrowRight, Eye, Layers
} from "lucide-react";
import { cn } from "@/lib/utils";
import { SampleGallery } from "@/components/sections/SampleGallery";

// --- Types ---
interface Prediction {
  prob: number;
  positive: boolean;
}

interface Finding {
  condition: string;
  prob: number;
  threshold: number;
  normalized_conf: number;
  risk: string;
}

interface YoloBox {
  class: string;
  conf: number;
  xyxy: number[];
}

interface AnalysisResults {
  status: string;
  predictions: Record<string, Prediction>;
  positive_conditions: string[];
  confirmed_findings: Finding[];
  risk_level: string;
  localization_type: Record<string, string>;
  images: {
    original_b64: string;
    heatmap_b64: string;
    boxes_b64: string;
    combined_b64: string;
    individual_heatmaps: Record<string, string>;
  };
  yolo_boxes: YoloBox[];
  case_flags: string[];
}

export default function DashboardPage() {
  // --- UI State ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"original" | "heatmap" | "boxes" | "combined">("original");
  const [backendReady, setBackendReady] = useState<boolean | null>(null); // null=checking, true=ready, false=unavailable

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // ── Backend Health Check (Every 3s) ───────
  useEffect(() => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    
    const checkHealth = async () => {
      try {
        const res = await fetch(`${backendUrl}/health`, { method: "GET" });
        if (res.ok) {
          const data = await res.json();
          setBackendReady(data.models_loaded === true);
        } else {
          setBackendReady(false);
        }
      } catch (err) {
        setBackendReady(false);
      }
    };

    checkHealth(); // initial
    const intervalId = setInterval(checkHealth, 3000);

    return () => clearInterval(intervalId);
  }, []);

  // Cancel analysis if backend goes offline
  useEffect(() => {
    if (backendReady === false && isAnalyzing) {
      abortControllerRef.current?.abort();
      setIsAnalyzing(false);
      setError("Backend connection lost during analysis. Please try again.");
      setSelectedFile(null);
      setPreviewUrl(null);
      setResults(null);
    }
  }, [backendReady, isAnalyzing]);

  // Auto-run analysis when a file is selected
  useEffect(() => {
    if (selectedFile && !results && !isAnalyzing && backendReady && !error) {
      runAnalysis();
    }
  }, [selectedFile, results, isAnalyzing, backendReady, error]);



  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (isAnalyzing && abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      setResults(null);
      setError(null);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      if (isAnalyzing && abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      setResults(null);
      setError(null);
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setError(null);

    abortControllerRef.current = new AbortController();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
      const response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        body: formData,
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        throw new Error("Failed to process image. Ensure backend is running.");
      }

      const data = await response.json();
      setResults(data);
      if (data.yolo_boxes && data.yolo_boxes.length > 0 && data.images && Object.keys(data.images.individual_heatmaps || {}).length > 0) {
        setViewMode("combined");
      } else if (data.yolo_boxes && data.yolo_boxes.length > 0) {
        setViewMode("boxes");
      } else if (data.images && Object.keys(data.images.individual_heatmaps || {}).length > 0) {
        setViewMode("heatmap");
      } else {
        setViewMode("original");
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || "An unexpected error occurred during analysis.");
        setSelectedFile(null);
        setPreviewUrl(null);
        setResults(null);
      }
    } finally {
      setIsAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="space-y-10 relative">
      {/* Offline Modal */}
      {backendReady === false && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-[#0a0a0a] border border-white/10 rounded-2xl p-6 md:p-10 max-w-md w-full flex flex-col items-center text-center space-y-4 shadow-2xl">
            <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-2 animate-pulse">
              <AlertCircle className="text-red-500 h-8 w-8" />
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">System Offline</h2>
            <p className="text-white/60 text-sm leading-relaxed">
              The diagnostic backend is currently offline or loading models. Please wait for the system to come online to perform analysis.
            </p>
            <div className="mt-6 flex items-center gap-3 px-5 py-2.5 bg-white/5 border border-white/10 rounded-full">
              <Loader2 className="h-4 w-4 text-accent animate-spin" />
              <span className="text-[10px] font-mono uppercase tracking-widest text-white/60">
                Attempting Reconnection...
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 md:gap-6">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight flex items-center gap-2 md:gap-3">
            <Brain className="text-accent h-6 w-6 md:h-8 md:w-8 shrink-0" /> 
            Neural Diagnostic Terminal
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2 md:gap-3">
            {/* Backend status indicator */}
            <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 rounded-full">
              <span className={cn(
                "w-2 h-2 rounded-full",
                backendReady === null ? "bg-yellow-400 animate-pulse" :
                backendReady ? "bg-green-400" : "bg-red-400"
              )} />
              <span className="text-[10px] font-mono uppercase tracking-widest text-white/40">
                {backendReady === null ? "Connecting..." : backendReady ? "Ready" : "Offline"}
              </span>
            </div>
            <button
              onClick={() => { 
                if (isAnalyzing && abortControllerRef.current) {
                  abortControllerRef.current.abort();
                }
                setSelectedFile(null); 
                setPreviewUrl(null); 
                setResults(null); 
                setError(null);
              }}
              className="flex items-center justify-center whitespace-nowrap px-6 py-2 bg-white/5 border border-white/10 text-white/60 hover:text-white rounded-full text-xs font-bold tracking-widest uppercase transition-colors pl-[1.6rem]"
            >
              Reset
            </button>


        </div>
      </div>

      {/* Main Analysis Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left: Interaction & Preview */}
        <div className="lg:col-span-7 space-y-6">
          
          {!previewUrl ? (
            // Upload Dropzone
            <div 
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="group relative min-h-[400px] md:min-h-[500px] border-2 border-dashed border-white/10 hover:border-accent/40 rounded-container flex flex-col items-center justify-center transition-all cursor-pointer bg-surface/20 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-b from-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative flex flex-col items-center text-center px-4 md:px-8">
                <div className="w-16 h-16 rounded-full bg-accent/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <Upload className="text-accent h-8 w-8" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2 tracking-tight">Initialize Scan Data</h3>
                <p className="text-white/40 text-sm max-w-xs font-mono uppercase tracking-wider leading-relaxed">
                  Choose Chest X-Ray PNG/JPG Recommended : 1024 x 1024 dim 
                </p>
              </div>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileSelect} 
                accept="image/*" 
                className="hidden" 
              />
            </div>
          ) : (
            // Preview & Result Visualization
            <div className="bg-surface/50 backdrop-blur-xl border border-white/5 rounded-container overflow-hidden p-4 md:p-6">
              <div className="relative aspect-square md:aspect-auto md:min-h-[600px] bg-black/50 rounded-xl border border-white/5 flex flex-col items-center justify-center p-4 md:p-8 overflow-hidden">
                
                {/* Layered Display */}
                {results ? (
                  <img 
                    src={
                      viewMode === "original" ? (results.images.original_b64 ? `data:image/jpeg;base64,${results.images.original_b64}` : previewUrl || "") : 
                      viewMode === "heatmap" ? `data:image/jpeg;base64,${results.images.heatmap_b64}` :
                      viewMode === "boxes" ? `data:image/jpeg;base64,${results.images.boxes_b64}` :
                      `data:image/jpeg;base64,${results.images.combined_b64 || results.images.boxes_b64 || results.images.heatmap_b64}`
                    } 
                    className="w-full h-full object-contain shadow-2xl"
                    alt="Diagnostic view"
                  />
                ) : (
                  <img src={previewUrl} className="w-full h-full object-contain opacity-50" alt="Preview" />
                )}

                {/* Processing Overlay */}
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
                    <Loader2 className="h-10 w-10 text-accent animate-spin" />
                    <p className="font-mono text-xs text-accent tracking-[0.3em] uppercase animate-pulse">Running Neural Inference...</p>
                  </div>
                )}
              </div>

              {/* View Controls Moved Outside */}
              {results && (
                <div className="flex flex-wrap justify-center items-center gap-2 bg-black/40 backdrop-blur-md border border-white/10 rounded-full p-2 px-4 w-fit mx-auto mt-4">
                    <ViewBtn active={viewMode === "original"} onClick={() => setViewMode("original")} label="Raw" icon={<Eye className="h-3 w-3" />} />
                    {Object.keys(results.images.individual_heatmaps || {}).length > 0 && (
                      <ViewBtn active={viewMode === "heatmap"} onClick={() => setViewMode("heatmap")} label="Heatmap" icon={<Brain className="h-3 w-3" />} />
                    )}
                    {results.yolo_boxes.length > 0 && (
                      <ViewBtn active={viewMode === "boxes"} onClick={() => setViewMode("boxes")} label="Detection" icon={<Crosshair className="h-3 w-3" />} />
                    )}
                    {results.images.combined_b64 && Object.keys(results.images.individual_heatmaps || {}).length > 0 && results.yolo_boxes.length > 0 && (
                      <ViewBtn active={viewMode === "combined"} onClick={() => setViewMode("combined")} label="Combined" icon={<Layers className="h-3 w-3" />} />
                    )}
                </div>
              )}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-container flex items-start gap-3">
              <AlertCircle className="text-red-400 h-5 w-5 mt-0.5" />
              <div className="space-y-1">
                <p className="text-red-400 text-sm font-bold">Inference Error</p>
                <p className="text-red-400/60 text-xs font-mono">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right: Results & Metrics */}
        <div className="lg:col-span-5 space-y-6">
          
        <div className={!results && !isAnalyzing ? "block" : "hidden"}>
          <SampleGallery onSelectImage={(file) => {
            setResults(null);
            setError(null);
            setSelectedFile(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
          }} />
        </div>

        {(results || isAnalyzing) && (
          <>
            {/* Risk Level Badge */}
              {results && (
                <div className={cn(
                  "p-4 md:p-6 rounded-container border transition-all flex flex-col justify-center",
                  results.risk_level === "CRITICAL" ? "bg-red-500/10 border-red-500/20" :
                  results.risk_level === "HIGH" ? "bg-orange-500/10 border-orange-500/20" :
                  results.risk_level === "MEDIUM" ? "bg-yellow-500/10 border-yellow-500/20" :
                  "bg-green-500/10 border-green-500/20"
                )}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-[10px] font-mono text-white/40 uppercase tracking-[0.2em]">Clinical Risk Index</p>
                  </div>
                  <p className={cn(
                    "text-4xl font-bold tracking-tighter uppercase",
                    results.risk_level === "CRITICAL" ? "text-red-500" :
                    results.risk_level === "HIGH" ? "text-orange-500" :
                    results.risk_level === "MEDIUM" ? "text-yellow-500" :
                    "text-green-500"
                  )}>
                    {results.risk_level}
                  </p>
                  {results.case_flags.filter(f => !f.startsWith("heatmap_only:")).length > 0 && (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {results.case_flags
                        .filter(f => !f.startsWith("heatmap_only:"))
                        .map(flag => (
                          <span key={flag} className="px-2 py-1 bg-white/10 border border-white/10 rounded text-[8px] font-bold uppercase tracking-widest text-white/80">
                            {flag.replace(/_/g, ' ')}
                          </span>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {/* Confirmed Findings */}
              <div className="bg-surface/50 backdrop-blur-xl border border-white/5 rounded-container p-4 md:p-6 space-y-4 md:space-y-6">
                <h3 className="text-white font-bold flex items-center gap-2 text-sm uppercase tracking-widest">
                  <CheckCircle2 className="text-accent h-4 w-4" />
                  Confirmed Findings
                </h3>
                
                <div className="space-y-4">
                  {!results ? (
                    <p className="text-white/20 text-xs font-mono italic">Waiting for analysis sequence...</p>
                  ) : (results?.confirmed_findings?.length ?? 0) === 0 ? (
                    <div className="flex flex-col items-center justify-center py-4 text-center">
                      <Activity className="text-green-500/40 h-8 w-8 mb-2" />
                      <p className="text-green-500/60 text-[10px] font-mono uppercase tracking-widest">System Clear: No conditions detected</p>
                    </div>
                  ) : (
                    <>
                      {results?.confirmed_findings?.map((item, idx) => (
                        <div key={item.condition} className="space-y-2">
                          <div className="flex justify-between items-end">
                            <span className="text-white text-xs font-bold tracking-tight">
                              {idx + 1}. {item.condition.replace(/_/g, ' ')}
                            </span>
                            <span className="text-accent text-[10px] font-mono">{(item.prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="relative h-1 bg-white/5 rounded-full overflow-hidden">
                            <div 
                              className="absolute top-0 left-0 h-full bg-black/40 transition-all duration-1000 ease-out z-10"
                              style={{ width: `${item.threshold * 100}%` }}
                            />
                            <div 
                              className="absolute top-0 h-full w-[2px] bg-black/80 z-20"
                              style={{ left: `${item.threshold * 100}%` }}
                            />
                            <div 
                              className="absolute top-0 left-0 h-full bg-accent transition-all duration-1000 ease-out z-0"
                              style={{ width: `${item.prob * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                      
                      {/* Technical Note */}
                      <div className="pt-4 mt-4 border-t border-white/5 flex gap-2">
                        <Info className="h-3 w-3 text-white/20 shrink-0" />
                        <p className="text-[9px] leading-relaxed text-white/50 font-mono uppercase tracking-wider">
                          Note: Displaying original model confidence. The grey overlay and marker denote the clinical threshold for a positive finding.
                        </p>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Object Detection */}
              <div className="bg-surface/50 backdrop-blur-xl border border-white/5 rounded-container p-4 md:p-6 space-y-4 md:space-y-6">
                <h3 className="text-white font-bold flex items-center gap-2 text-sm uppercase tracking-widest">
                  <Crosshair className="text-accent h-4 w-4" />
                  Object Detection
                </h3>
                
                <div className="space-y-4">
                  {!results ? (
                    <p className="text-white/20 text-xs font-mono italic">Waiting for analysis sequence...</p>
                  ) : results.yolo_boxes.length === 0 ? (
                    <p className="text-white/20 text-[10px] font-mono uppercase tracking-widest py-2">No localized geometric findings.</p>
                    ) : (
                        <>
                    {results.yolo_boxes.map((box, idx) => (
                      <div key={idx} className="space-y-2">
                        <div className="flex justify-between items-end">
                          <span className="text-white text-xs font-bold tracking-tight">
                            {idx + 1}. {box.class}
                          </span>
                          <span className="text-accent text-[10px] font-mono">{(box.conf * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-accent transition-all duration-1000 ease-out"
                            style={{ width: `${box.conf * 100}%` }}
                          />
                        </div>
                      </div>
                      ))}

                      {/* Technical Note */}
                      <div className="pt-4 mt-4 border-t border-white/5 flex gap-2">
                      <Info className="h-3 w-3 text-white/20 shrink-0" />
                      <p className="text-[9px] leading-relaxed text-white/50 font-mono uppercase tracking-wider">
                        Note: Detection confidence represents raw geometric certainty of the bounding box, 
                        independent of the clinical thresholds used in the findings above.
                      </p>
                      </div>
                    </>
                    )}
                  </div>
                </div>
            </>
          )}
        </div>
      </div>

      {/* Session Info Footer */}
      <div className="mt-12 md:mt-20 pb-10">
        <div className="w-full flex items-center justify-center px-4 md:px-8 py-4 bg-white/[0.02] border border-white/5 rounded-container backdrop-blur-sm">
          <p className="text-white/30 font-mono text-[10px] uppercase tracking-[0.2em]">
            // System: <span className="text-white/60 ml-2">Autonomous Neural Diagnostic Terminal</span>
          </p>
        </div>
      </div>

    </div>
  );
}

function ViewBtn({ active, onClick, label, icon }: { active: boolean, onClick: () => void, label: string, icon: React.ReactNode }) {
    return (
        <button 
            onClick={onClick}
            className={cn(
                "flex items-center gap-1 md:gap-1.5 px-2 md:px-3 py-1 md:py-1.5 rounded-full text-[9px] md:text-[10px] font-bold uppercase tracking-widest transition-all whitespace-nowrap",
                active ? "bg-accent text-primary" : "text-white/40 hover:text-white/70"
            )}
        >
            {icon}
            {label}
        </button>
    )
}

function StatCard({ label, value, status }: { label: string, value: string, status: string }) {
    return (
        <div className="bg-surface/30 border border-white/5 p-4 rounded-container">
            <p className="text-white/40 text-[9px] font-mono uppercase tracking-[0.2em] mb-1">{label}</p>
            <div className="flex items-center justify-between">
                <p className="text-lg font-bold text-white tracking-tight">{value}</p>
                <span className="text-[8px] font-mono text-accent uppercase tracking-widest">{status}</span>
            </div>
        </div>
    )
}
