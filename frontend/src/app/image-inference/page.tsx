"use client";

import { useState } from "react";

// Signal color scheme: Class 0 = Green (Safe), 1 = Green, 2 = Yellow (Caution), 3 = Red (Warning), 4 = Red (Critical)
const CLASS_SIGNAL = [
  { label: "Class 0 — Safe",     bar: "#22c55e", glow: "shadow-green-500/60",  badge: "bg-green-900/60 text-green-300 border-green-600",  dot: "bg-green-400",   ring: "ring-green-500"  },
  { label: "Class 1 — Normal",   bar: "#4ade80", glow: "shadow-green-400/50",  badge: "bg-green-800/60 text-green-200 border-green-500",  dot: "bg-green-300",   ring: "ring-green-400"  },
  { label: "Class 2 — Caution",  bar: "#facc15", glow: "shadow-yellow-400/60", badge: "bg-yellow-900/60 text-yellow-300 border-yellow-600", dot: "bg-yellow-400", ring: "ring-yellow-400" },
  { label: "Class 3 — Warning",  bar: "#f97316", glow: "shadow-orange-500/60", badge: "bg-orange-900/60 text-orange-300 border-orange-600", dot: "bg-orange-400", ring: "ring-orange-500" },
  { label: "Class 4 — Critical", bar: "#ef4444", glow: "shadow-red-500/60",    badge: "bg-red-900/60 text-red-300 border-red-600",         dot: "bg-red-400",    ring: "ring-red-500"   },
];

export default function ImageInferencePage() {
  const [flareId, setFlareId] = useState("flare_00001");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:8004/api/v1/forecast/image-inference?flare_id=${flareId}`
      );
      const data = await response.json();
      if (!response.ok || data.error) {
        setError(data.error || "Failed to fetch prediction.");
      } else {
        setResult(data);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const predictedSignal = result ? CLASS_SIGNAL[result.predicted_class] : null;

  return (
    <div
      className="min-h-screen text-white flex flex-col items-center justify-start py-12 px-4"
      style={{ background: "radial-gradient(ellipse at 50% 10%, #1a0000 0%, #000000 70%)" }}
    >
      {/* Header */}
      <div className="w-full max-w-2xl mb-8 text-center">
        <div className="inline-flex items-center gap-2 mb-3 px-4 py-1 rounded-full border border-red-800/60 bg-red-950/30 text-red-400 text-xs font-bold tracking-widest uppercase">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse inline-block" />
          AstroNova — Live Signal Monitor
        </div>
        <h1
          className="text-5xl font-black tracking-tighter mb-3"
          style={{
            background: "linear-gradient(90deg, #ef4444 0%, #facc15 50%, #22c55e 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Solar Flare Classifier
        </h1>
        <p className="text-gray-500 text-sm max-w-lg mx-auto leading-relaxed">
          Submit a Flare ID to run live inference through the trained MLP model.
          Results are color-coded by hazard signal level.
        </p>
      </div>

      {/* Signal Legend */}
      <div className="w-full max-w-2xl flex flex-wrap justify-center gap-2 mb-8">
        {CLASS_SIGNAL.map((sig, i) => (
          <span
            key={i}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full border text-xs font-semibold ${sig.badge}`}
          >
            <span className={`w-2 h-2 rounded-full ${sig.dot}`} />
            {sig.label}
          </span>
        ))}
      </div>

      {/* Main Card */}
      <div
        className="w-full max-w-2xl rounded-2xl p-8 shadow-2xl border border-red-900/30"
        style={{ background: "rgba(10,0,0,0.85)", backdropFilter: "blur(16px)" }}
      >
        {/* Input Form */}
        <form onSubmit={handlePredict} className="flex flex-col gap-3 mb-8">
          <label className="text-xs font-bold text-gray-400 uppercase tracking-widest ml-1">
            Flare ID Sequence
          </label>
          <div className="flex gap-3">
            <input
              type="text"
              value={flareId}
              onChange={(e) => setFlareId(e.target.value)}
              className="flex-1 bg-black border border-red-900/60 text-white rounded-lg px-4 py-3 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-yellow-500 placeholder-gray-700 transition-all"
              placeholder="e.g. flare_00001"
              required
            />
            <button
              id="predict-btn"
              type="submit"
              disabled={loading}
              className={`px-8 py-3 rounded-lg font-black text-sm uppercase tracking-wider transition-all transform active:scale-95 ${
                loading
                  ? "bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700"
                  : "border border-red-600 text-black font-black shadow-lg shadow-red-900/40 hover:shadow-red-500/60"
              }`}
              style={!loading ? { background: "linear-gradient(90deg, #ef4444, #facc15)" } : {}}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full border-2 border-yellow-500 border-t-transparent animate-spin inline-block" />
                  Scanning...
                </span>
              ) : "▶ Predict"}
            </button>
          </div>
        </form>

        {/* Error Banner */}
        {error && (
          <div className="bg-red-950/60 border border-red-700 text-red-300 px-4 py-3 rounded-lg mb-6 text-sm flex items-start gap-2">
            <span className="text-red-500 font-black mt-0.5">✕</span>
            <div><span className="font-bold text-red-400">Error:</span> {error}</div>
          </div>
        )}

        {/* Results */}
        {result && predictedSignal && (
          <div className="rounded-xl overflow-hidden border border-gray-800">
            {/* Signal Header Bar */}
            <div
              className="flex items-center justify-between px-6 py-4"
              style={{
                background: `linear-gradient(135deg, rgba(0,0,0,0.9), rgba(0,0,0,0.7))`,
                borderBottom: `2px solid ${predictedSignal.bar}40`,
              }}
            >
              <div className="flex items-center gap-3">
                <span
                  className={`w-4 h-4 rounded-full animate-pulse ring-2 ring-offset-1 ring-offset-black ${predictedSignal.dot} ${predictedSignal.ring}`}
                />
                <h2 className="text-lg font-black text-white tracking-wide">Inference Results</h2>
              </div>
              <span
                className={`text-xs font-black px-3 py-1.5 rounded-full border tracking-wider uppercase ${predictedSignal.badge}`}
              >
                {predictedSignal.label}
              </span>
            </div>

            {/* Meta Info */}
            <div className="grid grid-cols-2 gap-px bg-gray-800">
              <div className="bg-black px-5 py-4">
                <p className="text-xs text-gray-600 uppercase tracking-widest mb-1">Flare ID</p>
                <p className="font-mono text-sm text-yellow-300">{result.flare_id}</p>
              </div>
              <div className="bg-black px-5 py-4">
                <p className="text-xs text-gray-600 uppercase tracking-widest mb-1">Timestamp</p>
                <p className="font-mono text-sm text-yellow-300">{result.timestamp}</p>
              </div>
            </div>

            {/* Probability Bars */}
            <div className="bg-black px-6 py-6 space-y-4">
              <p className="text-xs text-gray-600 uppercase tracking-widest mb-4 font-bold">
                Signal Probability Distribution
              </p>
              {result.probabilities.map((prob: number, index: number) => {
                const sig = CLASS_SIGNAL[index];
                const isActive = index === result.predicted_class;
                return (
                  <div key={index} className="flex items-center gap-4">
                    <div className={`w-3 h-3 rounded-full flex-shrink-0 ${sig.dot} ${isActive ? "ring-2 ring-offset-1 ring-offset-black " + sig.ring : "opacity-40"}`} />
                    <div className="w-28 text-xs font-mono text-gray-500 truncate">
                      {isActive ? (
                        <span className="text-white font-bold">{sig.label.split("—")[0].trim()}</span>
                      ) : sig.label.split("—")[0].trim()}
                    </div>
                    <div className="flex-1 h-2.5 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${Math.max(prob * 100, 0.5)}%`,
                          background: isActive ? sig.bar : "#374151",
                          boxShadow: isActive ? `0 0 8px ${sig.bar}90` : "none",
                          opacity: isActive ? 1 : 0.5,
                        }}
                      />
                    </div>
                    <div className={`w-12 text-right text-xs font-mono ${isActive ? "text-white font-bold" : "text-gray-600"}`}>
                      {(prob * 100).toFixed(1)}%
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Footer Status Line */}
            <div
              className="px-6 py-3 text-xs font-mono flex items-center justify-between"
              style={{ background: `${predictedSignal.bar}12`, borderTop: `1px solid ${predictedSignal.bar}30` }}
            >
              <span className="text-gray-600">AstroNova MLP v1.0 · 2560-dim spectral embedding</span>
              <span style={{ color: predictedSignal.bar }}>● SIGNAL CLASSIFIED</span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom hint */}
      <p className="mt-6 text-gray-700 text-xs text-center">
        Try: <span className="text-gray-500 font-mono">flare_00001</span> · <span className="text-gray-500 font-mono">flare_00002</span> · <span className="text-gray-500 font-mono">flare_00010</span>
      </p>
    </div>
  );
}
