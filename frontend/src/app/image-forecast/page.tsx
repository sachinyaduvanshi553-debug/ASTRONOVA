"use client";

import { useState } from "react";
import Image from "next/image";

// Signal color scheme
const CLASS_COLORS: Record<string, { bar: string; glow: string; text: string }> = {
  "M+": { bar: "#facc15", glow: "shadow-yellow-400/60", text: "text-yellow-400" },
  "X+": { bar: "#ef4444", glow: "shadow-red-500/60", text: "text-red-400" },
  "C+": { bar: "#22c55e", glow: "shadow-green-500/60", text: "text-green-400" },
  "SAFE": { bar: "#3b82f6", glow: "shadow-blue-500/60", text: "text-blue-400" },
};

export default function MultimodalForecastPage() {
  const [simulationId, setSimulationId] = useState("AR12673_20170906");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      // The FastAPI gateway runs on 8000. 
      // The Phase 14 endpoint is at /api/v1/vision/forecast
      const payload = {
        image_paths: ["datasets/raw/sdo_images/synthetic/dummy.jpg"],
        telemetry: [[1e-5, 1e-4], [2e-5, 2e-4]],
        magnetic_features: [[0.5, 0.1, 0.2]],
        image_size: 256,
        use_uncertainty: true,
        mc_passes: 10
      };

      const response = await fetch("http://127.0.0.1:8000/api/v1/vision/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "Failed to fetch prediction.");
      } else {
        setResult(data);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className="min-h-screen text-white flex flex-col items-center justify-start py-12 px-4 font-sans"
      style={{ background: "radial-gradient(ellipse at top, #0f172a 0%, #020617 100%)" }}
    >
      {/* Header */}
      <div className="w-full max-w-5xl mb-12 text-center">
        <div className="inline-flex items-center gap-2 mb-4 px-4 py-1.5 rounded-full border border-cyan-800/60 bg-cyan-950/30 text-cyan-400 text-xs font-bold tracking-widest uppercase shadow-[0_0_15px_rgba(34,211,238,0.2)]">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse inline-block shadow-[0_0_8px_#22d3ee]" />
          AstroNova Multimodal Engine
        </div>
        <h1 className="text-5xl md:text-6xl font-black tracking-tighter mb-4 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600">
          Spatiotemporal Flare Forecast
        </h1>
        <p className="text-slate-400 text-sm md:text-base max-w-2xl mx-auto leading-relaxed">
          Jointly analyzing GOES X-ray telemetry, SDO/AIA images, and magnetic features through our Vision-Language-Temporal Transformer architecture to predict flare events up to 24 hours out.
        </p>
      </div>

      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input Panel */}
        <div className="lg:col-span-4 space-y-6">
          <div
            className="rounded-2xl p-6 shadow-2xl border border-slate-800/50 bg-slate-900/60 backdrop-blur-xl"
          >
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2 text-slate-200">
              <svg className="w-5 h-5 text-cyan-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Observation Config
            </h2>
            <form onSubmit={handlePredict} className="flex flex-col gap-4">
              <div>
                <label className="text-xs font-bold text-slate-400 uppercase tracking-widest ml-1 mb-1 block">
                  Active Region ID
                </label>
                <input
                  type="text"
                  value={simulationId}
                  onChange={(e) => setSimulationId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 text-slate-200 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 transition-all shadow-inner"
                  placeholder="e.g. AR12673"
                  required
                />
              </div>
              
              <button
                type="submit"
                disabled={loading}
                className={`w-full py-3.5 rounded-xl font-black text-sm uppercase tracking-wider transition-all transform active:scale-95 flex items-center justify-center gap-2 ${
                  loading
                    ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                    : "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(8,145,178,0.4)] border border-cyan-400/30"
                }`}
              >
                {loading ? (
                  <>
                    <span className="w-4 h-4 rounded-full border-2 border-white/20 border-t-white animate-spin" />
                    Running Inference...
                  </>
                ) : (
                  <>▶ Run Forecast</>
                )}
              </button>
            </form>

            {error && (
              <div className="mt-4 bg-red-950/40 border border-red-900/50 text-red-400 px-4 py-3 rounded-xl text-sm">
                <span className="font-bold">Error:</span> {error}
              </div>
            )}
          </div>
          
          <div className="rounded-2xl p-6 border border-slate-800/50 bg-slate-900/40 backdrop-blur-xl">
             <h3 className="text-sm font-bold text-slate-300 mb-3 uppercase tracking-wider">System Status</h3>
             <ul className="space-y-2 text-xs text-slate-500 font-mono">
               <li className="flex justify-between"><span>Vision Encoder</span><span className="text-cyan-500">ONLINE</span></li>
               <li className="flex justify-between"><span>Temporal Transformer</span><span className="text-cyan-500">ONLINE</span></li>
               <li className="flex justify-between"><span>GOES Telemetry Sync</span><span className="text-cyan-500">SYNCED</span></li>
               <li className="flex justify-between"><span>SDO/AIA Stream</span><span className="text-green-500">LIVE</span></li>
             </ul>
          </div>
        </div>

        {/* Right Column: Output Panel */}
        <div className="lg:col-span-8 space-y-6">
          {!result && !loading && (
             <div className="h-full min-h-[400px] flex items-center justify-center rounded-2xl border border-dashed border-slate-800 bg-slate-900/20">
               <p className="text-slate-600 font-medium">Initialize forecast to view multimodal analysis.</p>
             </div>
          )}
          
          {loading && (
             <div className="h-full min-h-[400px] flex flex-col items-center justify-center gap-4 rounded-2xl border border-slate-800 bg-slate-900/40">
               <div className="relative w-16 h-16">
                 <div className="absolute inset-0 rounded-full border-t-2 border-cyan-500 animate-spin"></div>
                 <div className="absolute inset-2 rounded-full border-l-2 border-blue-500 animate-spin animate-reverse"></div>
               </div>
               <p className="text-cyan-500 animate-pulse text-sm font-bold tracking-widest uppercase">Processing Multimodal Tensors</p>
             </div>
          )}

          {result && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-700">
              {/* Visualizations Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* Predicted Image Panel */}
                <div className="rounded-2xl overflow-hidden border border-slate-800/80 bg-slate-900/80 shadow-xl group">
                  <div className="bg-slate-950/80 px-4 py-2 border-b border-slate-800 flex justify-between items-center">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">AI Forecast (t+60m)</span>
                    <span className="text-[10px] font-mono bg-blue-900/50 text-blue-300 px-2 py-0.5 rounded border border-blue-700/50">Synthetic Vision</span>
                  </div>
                  <div className="aspect-square bg-black relative flex items-center justify-center overflow-hidden">
                    {result.predicted_image_base64 ? (
                      <img 
                        src={`data:image/jpeg;base64,${result.predicted_image_base64}`} 
                        alt="Predicted Solar Image" 
                        className="w-full h-full object-cover opacity-90 group-hover:scale-105 transition-transform duration-700 mix-blend-screen"
                      />
                    ) : (
                       <div className="text-slate-700 font-mono text-sm">Image Decoder Disabled</div>
                    )}
                    {/* Disclaimer Overlay */}
                    <div className="absolute bottom-2 left-2 right-2 bg-black/60 backdrop-blur px-2 py-1 rounded text-[9px] text-slate-400 text-center border border-slate-800">
                      {result.scientific_disclaimer.split(".")[0]}
                    </div>
                  </div>
                </div>

                {/* Heatmap Panel */}
                <div className="rounded-2xl overflow-hidden border border-slate-800/80 bg-slate-900/80 shadow-xl group">
                  <div className="bg-slate-950/80 px-4 py-2 border-b border-slate-800 flex justify-between items-center">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Spatial Attention</span>
                    <span className="text-[10px] font-mono bg-purple-900/50 text-purple-300 px-2 py-0.5 rounded border border-purple-700/50">Grad-CAM</span>
                  </div>
                  <div className="aspect-square bg-black relative flex items-center justify-center overflow-hidden">
                    {result.flare_heatmap_base64 ? (
                      <img 
                        src={`data:image/jpeg;base64,${result.flare_heatmap_base64}`} 
                        alt="Attention Heatmap" 
                        className="w-full h-full object-cover opacity-90 group-hover:scale-105 transition-transform duration-700 mix-blend-screen"
                      />
                    ) : (
                      <div className="text-slate-700 font-mono text-sm">Heatmap Unavailable</div>
                    )}
                  </div>
                </div>

              </div>

              {/* Horizons Panel */}
              <div className="rounded-2xl p-6 border border-slate-800/50 bg-slate-900/60 backdrop-blur-xl shadow-xl">
                <h3 className="text-sm font-bold text-slate-300 mb-4 uppercase tracking-wider border-b border-slate-800 pb-2">
                  Multi-Horizon Risk Assessment
                </h3>
                
                <div className="space-y-4">
                  {result.horizons.map((hz: any) => {
                    const prob = hz.probability_M_plus * 100;
                    const c = hz.predicted_class.includes("M") || hz.predicted_class.includes("X") 
                      ? CLASS_COLORS["M+"] 
                      : (hz.predicted_class.includes("C") ? CLASS_COLORS["C+"] : CLASS_COLORS["SAFE"]);
                      
                    return (
                      <div key={hz.horizon} className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50 flex flex-col md:flex-row md:items-center gap-4 hover:border-slate-700 transition-colors">
                        
                        <div className="w-24 shrink-0">
                          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest bg-slate-900 px-2 py-1 rounded border border-slate-800">
                            {hz.horizon_display}
                          </span>
                        </div>
                        
                        <div className="flex-1 space-y-1.5">
                          <div className="flex justify-between text-xs">
                            <span className="text-slate-500">M-Class+ Probability</span>
                            <span className={`font-mono font-bold ${c.text}`}>{prob.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800/80">
                            <div 
                              className="h-full rounded-full transition-all duration-1000 ease-out"
                              style={{ 
                                width: `${Math.max(prob, 1)}%`,
                                backgroundColor: c.bar,
                                boxShadow: `0 0 10px ${c.bar}80`
                              }}
                            />
                          </div>
                          {hz.uncertainty_M_plus !== null && (
                            <div className="text-[10px] text-slate-500 font-mono text-right">
                              ± {(hz.uncertainty_M_plus * 100).toFixed(1)}% epistemic
                            </div>
                          )}
                        </div>
                        
                        <div className="w-24 shrink-0 text-right">
                          <span 
                            className="text-[10px] font-black uppercase tracking-wider px-2 py-1 rounded border"
                            style={{ 
                              color: c.bar, 
                              borderColor: `${c.bar}40`,
                              backgroundColor: `${c.bar}10` 
                            }}
                          >
                            {hz.predicted_class}
                          </span>
                        </div>
                        
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
