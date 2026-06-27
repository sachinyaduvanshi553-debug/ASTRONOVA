import os


def create_file(path, content):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. globals.css ---
create_file("frontend/src/app/globals.css", """@import "tailwindcss";

@layer base {
  body {
    background-color: #0b0f19;
    color: #f3f4f6;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
}

/* Glassmorphism custom classes */
.glass-panel {
  background: rgba(17, 24, 39, 0.7);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.glass-card {
  background: rgba(31, 41, 55, 0.4);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.glow-glow {
  box-shadow: 0 0 15px rgba(59, 130, 246, 0.3);
}

.glow-orange {
  box-shadow: 0 0 15px rgba(249, 115, 22, 0.3);
}

.glow-red {
  box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
}
""")

# --- 2. layout.tsx ---
create_file("frontend/src/app/layout.tsx", """import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AstroNova - AI Solar Flare & Space Weather Platform',
  description: 'AI-Powered Solar Flare Forecasting & Space Weather Intelligence Platform for Space Operations.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
""")

# --- 3. page.tsx ---
create_file("frontend/src/app/page.tsx", """'use client';

import React, { useState, useEffect } from 'react';
import {
  Activity,
  AlertTriangle,
  Globe,
  Radio,
  Cpu,
  History,
  Shield,
  MessageSquare,
  Clock,
  Compass,
  ArrowUpRight,
  TrendingUp,
  Database,
  Volume2
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell
} from 'recharts';

// Preset local data to run out-of-the-box
const initialFluxData = Array.from({ length: 60 }, (_, i) => {
  const timeStr = `${10 + Math.floor(i / 60)}:${String(i % 60).padStart(2, '0')}`;
  // simulate quiescent background flux 1e-8 to 1e-7
  const base = 2e-8;
  const rand = Math.sin(i * 0.1) * 5e-9 + Math.random() * 2e-9;

  // Inject M-class solar flare at index 35-50
  let flare = 0;
  if (i >= 35 && i <= 50) {
    const progress = (i - 35) / 15;
    flare = 1.2e-5 * Math.sin(progress * Math.PI) * Math.exp(-progress * 2);
  }

  const soft = base + rand + flare;
  return {
    time: timeStr,
    softFlux: soft,
    hardFlux: soft * 0.15 + Math.random() * 1e-10
  };
});

const defaultXAIImportance = [
  { name: 'Soft Flux Rolling Mean (30m)', value: 42, color: '#3b82f6' },
  { name: 'Soft/Hard X-Ray Ratio', value: 28, color: '#10b981' },
  { name: 'Soft Flux Gradient (1st Deriv)', value: 18, color: '#f59e0b' },
  { name: 'Hard Flux Rolling Std (15m)', value: 12, color: '#ec4899' }
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('live');
  const [fluxData, setFluxData] = useState(initialFluxData);
  const [currentFlux, setCurrentFlux] = useState(1.45e-5); // current flare peak
  const [goesClass, setGoesClass] = useState('M1.4');
  const [shiScore, setShiScore] = useState(0.58);
  const [shiCategory, setShiCategory] = useState('High');

  // Copilot State
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    {
      sender: 'copilot',
      text: 'AstroNova space intelligence online. Loaded Aditya-L1 SoLEXS profiles. How can I assist you with space weather operations today?'
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  // Auto-update timer to simulate streaming data
  useEffect(() => {
    const interval = setInterval(() => {
      setFluxData((prev) => {
        const nextTime = new Date();
        const timeStr = `${String(nextTime.getHours()).padStart(2, '0')}:${String(nextTime.getMinutes()).padStart(2, '0')}:${String(nextTime.getSeconds()).padStart(2, '0')}`;

        // Random fluctuation
        const base = 2e-8;
        const rand = Math.random() * 3e-9;
        let flare = 0;

        // Occasional flare injection
        if (Math.random() > 0.85) {
          flare = (Math.random() * 5e-6) + 1e-6;
        }

        const newSoft = base + rand + flare;
        const newHard = newSoft * 0.12 + Math.random() * 1e-10;

        // Update current indicators
        setCurrentFlux(newSoft);
        const goesVal = newSoft < 1e-8 ? 'A0.0' :
                        newSoft < 1e-7 ? `B${(newSoft/1e-7*10).toFixed(1)}` :
                        newSoft < 1e-6 ? `C${(newSoft/1e-6*10).toFixed(1)}` :
                        newSoft < 1e-5 ? `M${(newSoft/1e-5*10).toFixed(1)}` :
                        `X${(newSoft/1e-4*10).toFixed(1)}`;
        setGoesClass(goesVal);

        const nextScore = Math.min(Math.max((newSoft * 2e4) + Math.random() * 0.1, 0.05), 0.98);
        setShiScore(nextScore);
        setShiCategory(nextScore < 0.2 ? 'Safe' : nextScore < 0.5 ? 'Moderate' : nextScore < 0.8 ? 'High' : 'Extreme');

        return [...prev.slice(1), { time: timeStr, softFlux: newSoft, hardFlux: newHard }];
      });
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { sender: 'user', text: chatInput };
    setChatHistory((prev) => [...prev, userMsg]);
    setChatInput('');
    setIsTyping(true);

    setTimeout(() => {
      let replyText = "Query analyzed. According to the vector database of space weather bulletins, this flare class matches historical NOAA Event 8472. Day-side ionospheric absorption is predicted to peak at 18.2 dB. Recommend standby for GSAT payload shielding verification.";
      if (chatInput.toLowerCase().includes('shielding') || chatInput.toLowerCase().includes('gsat')) {
        replyText = "GSAT satellites are currently in GEO orbits. M-class flares trigger alert level Amber. Ensure backup gyros are monitored and payload auto-recovery is enabled.";
      }
      setChatHistory((prev) => [...prev, { sender: 'copilot', text: replyText }]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-gray-100">
      {/* ────────────────────────────────────────────────────────────────────────────── */}
      {/* TOP HEADER */}
      {/* ────────────────────────────────────────────────────────────────────────────── */}
      <header className="glass-panel sticky top-0 z-50 flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <Compass className="w-8 h-8 text-blue-500 animate-spin-slow" />
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white">AstroNova</h1>
            <p className="text-xs text-gray-400">ISRO Solar Flare Forecasting & Space Weather Intelligence</p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2 px-3 py-1 bg-blue-950/40 border border-blue-500/30 rounded-full text-blue-400">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
            Telemetry: SoLEXS Connected
          </div>
          <div className="flex items-center gap-2 px-3 py-1 bg-orange-950/40 border border-orange-500/30 rounded-full text-orange-400">
            <span className="w-2 h-2 rounded-full bg-orange-400 animate-pulse"></span>
            SHI Category: {shiCategory}
          </div>
          <div className="flex items-center gap-2 text-gray-400 font-mono">
            <Clock className="w-4 h-4 text-gray-500" />
            UTC: {new Date().toUTCString().slice(17, 25)}
          </div>
        </div>
      </header>

      {/* ────────────────────────────────────────────────────────────────────────────── */}
      {/* MAIN CONTAINER */}
      {/* ────────────────────────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* SIDEBAR NAVIGATION */}
        <aside className="w-64 border-r border-gray-800 bg-[#0e1322]/80 flex flex-col p-4 gap-2">
          <button
            onClick={() => setActiveTab('live')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'live'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Activity className="w-5 h-5" />
            Live Solar Activity
          </button>

          <button
            onClick={() => setActiveTab('forecast')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'forecast'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <TrendingUp className="w-5 h-5" />
            Multi-Horizon Forecasting
          </button>

          <button
            onClick={() => setActiveTab('impact')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'impact'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Globe className="w-5 h-5" />
            Earth Impact Assessment
          </button>

          <button
            onClick={() => setActiveTab('satellites')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'satellites'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Shield className="w-5 h-5" />
            Satellite Risk Assessment
          </button>

          <button
            onClick={() => setActiveTab('history')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'history'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <History className="w-5 h-5" />
            Historical Similarity
          </button>

          <button
            onClick={() => setActiveTab('xai')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'xai'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Cpu className="w-5 h-5" />
            Explainable AI (XAI)
          </button>

          <button
            onClick={() => setActiveTab('copilot')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'copilot'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <MessageSquare className="w-5 h-5" />
            AI Operations Copilot
          </button>

          <div className="mt-auto border-t border-gray-800 pt-4 flex flex-col gap-2">
            <div className="p-3 bg-red-950/20 border border-red-500/20 rounded-lg flex items-start gap-2">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <div>
                <h4 className="text-xs font-semibold text-red-400">Ionospheric Anomaly</h4>
                <p className="text-[10px] text-gray-400">Scintillation warning over South-Asia quadrant active.</p>
              </div>
            </div>
          </div>
        </aside>

        {/* CONTENT CONTAINER */}
        <main className="flex-1 overflow-y-auto p-6 bg-[#0b0f19]">

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 1: LIVE SOLAR ACTIVITY */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'live' && (
            <div className="space-y-6">
              {/* TOP OVERVIEW CARD GRID */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Telemetry Flux (SoLEXS)</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-white font-mono">{currentFlux.toExponential(3)}</span>
                    <span className="text-[10px] text-gray-400">W/m²</span>
                  </div>
                  <span className="text-[10px] text-blue-400 font-mono">1.0 - 8.0 Å band</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Nowcast Classification</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold text-orange-500 font-mono">{goesClass}</span>
                  </div>
                  <span className="text-[10px] text-green-400">Confidence: 99.4%</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Solar Hazard Index (SHI)</span>
                  <div className="my-2 flex items-center justify-between">
                    <span className="text-3xl font-extrabold text-red-500 font-mono">{shiScore.toFixed(2)}</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                      shiCategory === 'Safe' ? 'bg-green-950 text-green-400 border border-green-500/20' :
                      shiCategory === 'Moderate' ? 'bg-orange-950 text-orange-400 border border-orange-500/20' :
                      'bg-red-950 text-red-400 border border-red-500/20'
                    }`}>
                      {shiCategory}
                    </span>
                  </div>
                  <span className="text-[10px] text-gray-400">Physics-informed composite index</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Alert Level</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-red-400">AMBER ALERT</span>
                  </div>
                  <span className="text-[10px] text-gray-400 font-mono">Mitigation protocols ready</span>
                </div>
              </div>

              {/* REAL-TIME CHART */}
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-base font-bold text-white">Live Solar Flux Telemetry</h3>
                    <p className="text-xs text-gray-400">Aditya-L1 SoLEXS real-time Soft & Hard X-ray fluxes</p>
                  </div>
                  <div className="flex gap-4 text-xs font-medium">
                    <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-blue-500 inline-block"></span>Soft X-Ray</span>
                    <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-pink-500 inline-block"></span>Hard X-Ray</span>
                  </div>
                </div>

                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={fluxData}>
                      <defs>
                        <linearGradient id="colorSoft" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorHard" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ec4899" stopOpacity={0.2}/>
                          <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} tickLine={false} />
                      <YAxis
                        scale="log"
                        domain={[1e-9, 1e-3]}
                        stroke="#9ca3af"
                        fontSize={11}
                        tickLine={false}
                        tickFormatter={(v) => v.toExponential(0)}
                      />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', color: '#fff' }} />
                      <Area type="monotone" dataKey="softFlux" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorSoft)" name="Soft X-Ray Flux" />
                      <Area type="monotone" dataKey="hardFlux" stroke="#ec4899" strokeWidth={1.5} fillOpacity={1} fill="url(#colorHard)" name="Hard X-Ray Flux" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 2: FORECASTING */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'forecast' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <h3 className="text-base font-bold text-white mb-2">Multi-Horizon Predictions</h3>
                <p className="text-xs text-gray-400 mb-6">Probability estimates for different solar flare classes</p>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {[5, 15, 30, 60].map((horizon, idx) => (
                    <div key={idx} className="glass-card p-4 rounded-lg flex flex-col justify-between">
                      <span className="text-xs font-semibold text-blue-400">{horizon} Min Horizon</span>
                      <div className="my-4">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-400">X-Class Flare Probability</span>
                          <span className="font-mono text-red-400">{(0.35 / (idx + 1) * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5">
                          <div className="bg-red-500 h-1.5 rounded-full" style={{ width: `${(0.35 / (idx + 1) * 100)}%` }}></div>
                        </div>
                      </div>
                      <div className="text-[10px] text-gray-400 flex justify-between">
                        <span>Expected Class: C{(8 - idx)}</span>
                        <span>Confidence: 89%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 3: EARTH IMPACT ASSESSMENT */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'impact' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                {/* SVG Continent Overlay representation */}
                <div className="glass-panel p-6 rounded-xl border border-gray-800 md:col-span-2">
                  <h3 className="text-base font-bold text-white mb-2">Regional Risk Map</h3>
                  <p className="text-xs text-gray-400 mb-6">Day-side exposure risk levels overlaid on major sectors</p>

                  <div className="relative bg-[#0d1220] rounded-xl border border-gray-800 p-6 flex flex-col justify-center items-center h-80 overflow-hidden">
                    {/* SVG map placeholder containing simple continent outlines and danger zones */}
                    <svg className="w-full h-60 text-gray-800 opacity-60" fill="currentColor" viewBox="0 0 800 400">
                      {/* Americas */}
                      <path d="M120 80h100v100H120zM140 180h80v150h-80z" />
                      {/* Europe / Africa */}
                      <path d="M380 60h100v120H380zM390 180h90v160h-90z" />
                      {/* Asia / India */}
                      <path d="M500 40h180v160H500z" />
                      <circle cx="560" cy="140" r="30" className="fill-red-500/20 stroke-red-500 stroke-2 animate-ping" />
                      <circle cx="560" cy="140" r="10" className="fill-red-600" />
                    </svg>

                    <div className="absolute top-4 left-4 bg-gray-900/90 border border-gray-800 p-3 rounded-lg text-xs">
                      <div className="font-semibold text-white mb-1">Impact Center: South-Asia</div>
                      <div className="text-red-400">Flux Absorption: 18.2 dB</div>
                      <div className="text-gray-400">Duration: 45 min</div>
                    </div>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
                  <div>
                    <h3 className="text-base font-bold text-white mb-2">System Disruption Indices</h3>
                    <p className="text-xs text-gray-400 mb-6">Estimated severity levels across telemetry and GPS receivers</p>

                    <div className="space-y-4">
                      {[
                        { name: 'GPS/GNSS Position Error', level: '+8.4m', color: 'text-orange-400' },
                        { name: 'HF Radio Transmissions', level: 'Severe Blackout', color: 'text-red-500' },
                        { name: 'Aviation VHF Radio', level: 'Moderate Fadeout', color: 'text-yellow-400' },
                        { name: 'Satellite Internet Downlink', level: 'High Jitter', color: 'text-orange-400' }
                      ].map((item, idx) => (
                        <div key={idx} className="flex justify-between items-center text-xs pb-2 border-b border-gray-800">
                          <span className="text-gray-400">{item.name}</span>
                          <span className={`font-semibold ${item.color}`}>{item.level}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <button className="w-full mt-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-semibold">
                    Export Flight Restriction Advisories
                  </button>
                </div>

              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 4: SATELLITE RISK ASSESSMENT */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'satellites' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-panel p-6 rounded-xl border border-gray-800">
                  <h3 className="text-base font-bold text-white mb-2">Satellite Catalog Risk Profiles</h3>
                  <p className="text-xs text-gray-400 mb-6">Real-time vulnerability mapping of ISRO satellites</p>

                  <div className="space-y-4">
                    {[
                      { name: 'INSAT-3D', type: 'Meteorological', orbit: 'GEO', risk: 0.38, status: 'Nominal' },
                      { name: 'GSAT-31', type: 'Communication', orbit: 'GEO', risk: 0.65, status: 'Disable Non-Essential' },
                      { name: 'Cartosat-3', type: 'Earth Obs.', orbit: 'LEO', risk: 0.82, status: 'Prepare Safe Mode' },
                      { name: 'EOS-04', type: 'Radar Imaging', orbit: 'LEO', risk: 0.21, status: 'Nominal' }
                    ].map((sat, idx) => (
                      <div key={idx} className="glass-card p-4 rounded-lg flex justify-between items-center text-xs">
                        <div>
                          <div className="font-semibold text-white">{sat.name} ({sat.type})</div>
                          <div className="text-[10px] text-gray-400">Orbit: {sat.orbit} | Risk Score: {sat.risk.toFixed(2)}</div>
                        </div>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                          sat.risk < 0.3 ? 'bg-green-950 text-green-400 border border-green-500/20' :
                          sat.risk < 0.7 ? 'bg-orange-950 text-orange-400 border border-orange-500/20' :
                          'bg-red-950 text-red-400 border border-red-500/20'
                        }`}>
                          {sat.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
                  <div>
                    <h3 className="text-base font-bold text-white mb-2">Orbit Vulnerability Profiles</h3>
                    <p className="text-xs text-gray-400 mb-6">Environmental radiation dose and drag factors</p>

                    <div className="space-y-6 my-4">
                      {[
                        { orbit: 'LEO (Low Earth Orbit)', score: 85, metric: 'Atmospheric drag index: Critical (+240%)' },
                        { orbit: 'MEO (Medium Earth Orbit)', score: 45, metric: 'Single event upset index: Elevated' },
                        { orbit: 'GEO (Geostationary)', score: 62, metric: 'Proton flux level: High (100 MeV)' }
                      ].map((item, idx) => (
                        <div key={idx} className="text-xs">
                          <div className="flex justify-between mb-1">
                            <span className="text-gray-300 font-semibold">{item.orbit}</span>
                            <span className="text-blue-400 font-bold">{item.score}%</span>
                          </div>
                          <div className="w-full bg-gray-800 rounded-full h-2 mb-1.5">
                            <div className="bg-blue-500 h-2 rounded-full" style={{ width: `${item.score}%` }}></div>
                          </div>
                          <div className="text-[10px] text-gray-400">{item.metric}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <button className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold">
                    Download Mitigation Operations PDF
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 5: HISTORICAL SIMILARITY */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'history' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <h3 className="text-base font-bold text-white mb-2">ChromaDB Vector Matching</h3>
                <p className="text-xs text-gray-400 mb-6">Historical solar weather profiles matching the current Soft X-ray gradient</p>

                <div className="space-y-4">
                  {[
                    { eventId: 'NOAA-8472', date: '2003-10-28', similarity: '94.2%', goesClass: 'X17.2', result: 'Extensive geomagnetic storm (Kp=9)' },
                    { eventId: 'NOAA-12673', date: '2017-09-06', similarity: '88.5%', goesClass: 'X9.3', result: 'Global HF radio blackouts' },
                    { eventId: 'NOAA-10486', date: '2003-11-04', similarity: '82.1%', goesClass: 'X28.0', result: 'Instrument saturation on multiple satellites' }
                  ].map((evt, idx) => (
                    <div key={idx} className="glass-card p-4 rounded-lg flex justify-between items-center text-xs">
                      <div>
                        <div className="font-semibold text-white">Event ID: {evt.eventId} (Class {evt.goesClass})</div>
                        <div className="text-[10px] text-gray-400">Date: {evt.date} | Description: {evt.result}</div>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-bold text-blue-400">{evt.similarity}</span>
                        <div className="text-[9px] text-gray-500">Vector Cosine match</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 6: EXPLAINABLE AI (XAI) */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'xai' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-panel p-6 rounded-xl border border-gray-800">
                  <h3 className="text-base font-bold text-white mb-2">SHAP Global Feature Importance</h3>
                  <p className="text-xs text-gray-400 mb-6">Model parameters ranked by impact on forecast confidence</p>

                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={defaultXAIImportance} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                        <XAxis type="number" stroke="#9ca3af" fontSize={11} />
                        <YAxis dataKey="name" type="category" stroke="#9ca3af" fontSize={10} width={130} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                        <Bar dataKey="value" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                          {defaultXAIImportance.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
                  <div>
                    <h3 className="text-base font-bold text-white mb-2">Local Attribution Report</h3>
                    <p className="text-xs text-gray-400 mb-6">Reasoning factors for the active M1.4 forecasting confidence</p>

                    <div className="space-y-4">
                      <div className="p-3 bg-emerald-950/20 border border-emerald-500/20 rounded-lg text-xs">
                        <div className="font-semibold text-emerald-400 mb-1">Attributing Positive Indicators</div>
                        <ul className="list-disc list-inside text-gray-300 space-y-1">
                          <li>Elevated Soft X-Ray Rolling Mean over last 30 minutes</li>
                          <li>Precursor Soft-to-Hard ratio exceeding 10.4 threshold</li>
                        </ul>
                      </div>

                      <div className="p-3 bg-rose-950/20 border border-rose-500/20 rounded-lg text-xs">
                        <div className="font-semibold text-rose-400 mb-1">Attributing Neutralizing Factors</div>
                        <ul className="list-disc list-inside text-gray-300 space-y-1">
                          <li>Lack of secondary hard flare growth in HEL1OS telemetry</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                  <span className="text-[10px] text-gray-400 text-center block mt-4">Captum Attributions generated via Integrated Gradients</span>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB 7: AI OPERATIONS COPILOT */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'copilot' && (
            <div className="glass-panel rounded-xl border border-gray-800 flex flex-col h-[500px]">
              <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-white">Space Weather Copilot</h3>
                  <p className="text-xs text-gray-400">Grounded to local space weather literature & ISRO manuals</p>
                </div>
                <div className="text-xs text-blue-400 flex items-center gap-1.5">
                  <Database className="w-4 h-4" />
                  RAG Database active
                </div>
              </div>

              {/* Chat screen */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[70%] p-3 rounded-lg text-xs ${
                      msg.sender === 'user'
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-[#181f30] text-gray-200 border border-gray-800 rounded-bl-none'
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-[#181f30] text-gray-400 border border-gray-800 p-3 rounded-lg text-xs rounded-bl-none animate-pulse">
                      Analyzing query & vector documents...
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <form onSubmit={handleSendMessage} className="p-4 border-t border-gray-800 flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask copilot about solar flares, shielding guidelines, or NOAA catalogs..."
                  className="flex-1 bg-[#090d16] border border-gray-800 rounded-lg px-4 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
                />
                <button
                  type="submit"
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold"
                >
                  Send
                </button>
              </form>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
""")

print("FRONTEND CODE FILES CREATED SUCCESSFULLY")
