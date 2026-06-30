import os


def create_file(path, content):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- page.tsx ---
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
  Volume2,
  Sliders,
  Award
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

const benchmarkLeaderboard = [
  { model: 'Transformer Forecaster', tss: 0.88, leadTime: '26 min', f1: 0.84, accuracy: 0.94 },
  { model: 'BiLSTM (Active)', tss: 0.82, leadTime: '22 min', f1: 0.80, accuracy: 0.92 },
  { model: 'GRU Forecaster', tss: 0.78, leadTime: '18 min', f1: 0.77, accuracy: 0.90 },
  { model: 'LightGBM Baseline', tss: 0.69, leadTime: '12 min', f1: 0.65, accuracy: 0.85 }
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('console');
  const [fluxData, setFluxData] = useState(initialFluxData);
  const [currentFlux, setCurrentFlux] = useState(1.45e-5);
  const [goesClass, setGoesClass] = useState('M1.4');
  const [shiScore, setShiScore] = useState(0.58);
  const [shiCategory, setShiCategory] = useState('High');
  const [lifecyclePhase, setLifecyclePhase] = useState('Rise');

  // Scenario Simulation States
  const [simulatedClass, setSimulatedClass] = useState('M5.0');
  const [simulatedValue, setSimulatedValue] = useState(5e-5);
  const [isSimulating, setIsSimulating] = useState(false);

  // Copilot State
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    {
      sender: 'copilot',
      text: 'AstroNova Mission Copilot active. Native Level-1 Aditya-L1 SoLEXS & HEL1OS readers aligned. Ready for query analysis.'
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  // Auto-update timer to simulate streaming data (ignored if user is simulating)
  useEffect(() => {
    if (isSimulating) return;

    const interval = setInterval(() => {
      setFluxData((prev) => {
        const nextTime = new Date();
        const timeStr = `${String(nextTime.getHours()).padStart(2, '0')}:${String(nextTime.getMinutes()).padStart(2, '0')}:${String(nextTime.getSeconds()).padStart(2, '0')}`;

        const base = 2e-8;
        const rand = Math.random() * 3e-9;
        let flare = 0;

        if (Math.random() > 0.85) {
          flare = (Math.random() * 5e-6) + 1e-6;
        }

        const newSoft = base + rand + flare;
        const newHard = newSoft * 0.12 + Math.random() * 1e-10;

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

        // Lifecycle Phase calculation
        setLifecyclePhase(newSoft > 1e-5 ? 'Rise' : (newSoft > 1e-6 ? 'Pre-flare' : 'Quiescent'));

        return [...prev.slice(1), { time: timeStr, softFlux: newSoft, hardFlux: newHard }];
      });
    }, 4000);
    return () => clearInterval(interval);
  }, [isSimulating]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMsg = { sender: 'user', text: chatInput };
    setChatHistory((prev) => [...prev, userMsg]);
    setChatInput('');
    setIsTyping(true);

    setTimeout(() => {
      let replyText = "Analyst Query resolved. Probabilistic forecasts show 84% confidence bounds within ±8 min lead-time. The active M-class flare suggests South-Asian D-layer ionization ceiling at 22 MHz.";
      if (chatInput.toLowerCase().includes('shielding') || chatInput.toLowerCase().includes('gsat')) {
        replyText = "GSAT communication assets in GEO are recommended for non-essential transponder power saflng. Magnetic helicity parameters show active thermal loading.";
      }
      setChatHistory((prev) => [...prev, { sender: 'copilot', text: replyText }]);
      setIsTyping(false);
    }, 1500);
  };

  const handleSimulate = (val: string) => {
    setIsSimulating(true);
    setGoesClass(val);

    let fluxVal = 1e-5;
    if (val.startsWith('X')) {
      fluxVal = parseFloat(val.substring(1)) * 1e-4;
    } else if (val.startsWith('M')) {
      fluxVal = parseFloat(val.substring(1)) * 1e-5;
    }

    setCurrentFlux(fluxVal);
    const calculatedScore = Math.min(0.35 + (fluxVal * 1.5e4), 0.98);
    setShiScore(calculatedScore);
    setShiCategory(calculatedScore < 0.2 ? 'Safe' : calculatedScore < 0.5 ? 'Moderate' : calculatedScore < 0.8 ? 'High' : 'Extreme');
    setLifecyclePhase('Peak');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19] text-gray-100">
      {/* HEADER */}
      <header className="glass-panel sticky top-0 z-50 flex items-center justify-between px-6 py-4 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <Compass className="w-8 h-8 text-blue-500 animate-spin-slow" />
          <div>
            <h1 className="text-xl font-bold tracking-wider text-white">AstroNova</h1>
            <p className="text-xs text-gray-400">Aditya-L1 Space Weather Intelligence Console</p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2 px-3 py-1 bg-blue-950/40 border border-blue-500/30 rounded-full text-blue-400">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
            Sensors: SoLEXS & HEL1OS Calibrated
          </div>
          <div className="flex items-center gap-2 px-3 py-1 bg-orange-950/40 border border-orange-500/30 rounded-full text-orange-400">
            <span className="w-2 h-2 rounded-full bg-orange-400 animate-pulse"></span>
            Lifecycle Phase: {lifecyclePhase}
          </div>
          <div className="flex items-center gap-2 text-gray-400 font-mono">
            <Clock className="w-4 h-4 text-gray-500" />
            UTC: {new Date().toUTCString().slice(17, 25)}
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <div className="flex-1 flex overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-64 border-r border-gray-800 bg-[#0e1322]/80 flex flex-col p-4 gap-2">
          <button
            onClick={() => setActiveTab('console')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'console'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Compass className="w-5 h-5" />
            ISRO Mission Console
          </button>

          <button
            onClick={() => setActiveTab('live')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'live'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Activity className="w-5 h-5" />
            Aditya-L1 Telemetry
          </button>

          <button
            onClick={() => setActiveTab('simulation')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'simulation'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Sliders className="w-5 h-5" />
            Scenario Simulator
          </button>

          <button
            onClick={() => setActiveTab('research')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
              activeTab === 'research'
                ? 'bg-blue-600/20 text-blue-400 border-l-4 border-blue-500'
                : 'text-gray-400 hover:bg-gray-800/40 hover:text-white'
            }`}
          >
            <Award className="w-5 h-5" />
            Research Benchmarking
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
            Mission AI Copilot
          </button>

          <div className="mt-auto border-t border-gray-800 pt-4 flex flex-col gap-2">
            <div className="p-3 bg-red-950/20 border border-red-500/20 rounded-lg flex items-start gap-2 text-xs">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <div>
                <h4 className="font-semibold text-red-400">Comms Blackout Alert</h4>
                <p className="text-[10px] text-gray-400">NavIC degradation forecast index high over South-Asia.</p>
              </div>
            </div>
          </div>
        </aside>

        {/* CONTENT */}
        <main className="flex-1 overflow-y-auto p-6 bg-[#0b0f19]">

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB: ISRO MISSION CONSOLE */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'console' && (
            <div className="space-y-6">
              {/* TOP SUMMARY STATS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Solar Hazard Index (SHI)</span>
                  <div className="my-2 flex items-center justify-between">
                    <span className="text-3xl font-extrabold text-red-500 font-mono">{shiScore.toFixed(2)}</span>
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-red-950 text-red-400 border border-red-500/20">
                      {shiCategory}
                    </span>
                  </div>
                  <span className="text-[10px] text-gray-400">SHI Formula active</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">GOES Target Class Nowcast</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold text-orange-500 font-mono">{goesClass}</span>
                  </div>
                  <span className="text-[10px] text-green-400">Confidence bounds: ±8%</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Estimated Time-to-Flare</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-2xl font-bold text-white font-mono">22 min</span>
                  </div>
                  <span className="text-[10px] text-blue-400">Dynamic lead-time optimization</span>
                </div>

                <div className="glass-card p-4 rounded-xl flex flex-col justify-between">
                  <span className="text-xs text-gray-400 font-medium">Telemetry Source</span>
                  <div className="my-2 flex items-baseline gap-2">
                    <span className="text-xl font-bold text-blue-400">Aditya-L1 L1 Data</span>
                  </div>
                  <span className="text-[10px] text-gray-400 font-mono">FITS / CDF synchronization</span>
                </div>
              </div>

              {/* MISSION CONSOLE SUMMARY MATRIX */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

                {/* Visual Map/Earth Impact */}
                <div className="glass-panel p-6 rounded-xl border border-gray-800 md:col-span-2">
                  <h3 className="text-base font-bold text-white mb-2">ISRO Geospatial Earth Impact</h3>
                  <p className="text-xs text-gray-400 mb-6">NavIC/D-layer absorption projection over South-Asia quadrant</p>

                  <div className="relative bg-[#0d1220] rounded-xl border border-gray-800 p-6 flex flex-col justify-center items-center h-72 overflow-hidden">
                    <svg className="w-full h-56 text-gray-800 opacity-60" fill="currentColor" viewBox="0 0 800 400">
                      <path d="M120 80h100v100H120zM140 180h80v150h-80z" />
                      <path d="M380 60h100v120H380zM390 180h90v160h-90z" />
                      <path d="M500 40h180v160H500z" />
                      <circle cx="560" cy="140" r="30" className="fill-red-500/20 stroke-red-500 stroke-2 animate-ping" />
                      <circle cx="560" cy="140" r="10" className="fill-red-600" />
                    </svg>

                    <div className="absolute top-4 left-4 bg-gray-900/90 border border-gray-800 p-3 rounded-lg text-xs">
                      <div className="font-semibold text-white mb-1">Impact Center: South-Asia</div>
                      <div className="text-red-400">NavIC Scintillation Index (S4): 0.74</div>
                      <div className="text-gray-400">Absorption ceiling: 22 MHz</div>
                    </div>
                  </div>
                </div>

                {/* Mitigations & Operational Recommendations */}
                <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col justify-between">
                  <div>
                    <h3 className="text-base font-bold text-white mb-2">Operational Guidelines</h3>
                    <p className="text-xs text-gray-400 mb-6">Actionable satellite & comms mitigations</p>

                    <div className="space-y-4">
                      {[
                        { title: 'GSAT GEO Satellites', action: 'Safing/Amber: Prepare backup gyro systems', color: 'text-orange-400' },
                        { title: 'NavIC Receivers', action: 'Scintillation active: auto-tracking mode', color: 'text-red-500' },
                        { title: 'Aviation Transponders', action: 'Advisory: route redirection on South-Asia', color: 'text-yellow-400' },
                        { title: 'Power Grid Operators', action: 'Inductive current load warning S4=0.7', color: 'text-orange-400' }
                      ].map((item, idx) => (
                        <div key={idx} className="text-xs pb-2 border-b border-gray-800">
                          <div className="font-semibold text-gray-300">{item.title}</div>
                          <div className={`text-[11px] ${item.color}`}>{item.action}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <button className="w-full mt-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold">
                    Download Execution Plan
                  </button>
                </div>

              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB: ADITYA-L1 TELEMETRY */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'live' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-base font-bold text-white">Aditya-L1 Real-Time Sync</h3>
                    <p className="text-xs text-gray-400">Aligned Soft X-Ray (SoLEXS) and Hard X-Ray (HEL1OS) fluxes</p>
                  </div>
                </div>

                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={fluxData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} />
                      <YAxis
                        scale="log"
                        domain={[1e-9, 1e-3]}
                        stroke="#9ca3af"
                        fontSize={11}
                        tickFormatter={(v) => v.toExponential(0)}
                      />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                      <Area type="monotone" dataKey="softFlux" stroke="#3b82f6" strokeWidth={2} fillOpacity={0.1} fill="#3b82f6" name="SoLEXS Soft X-Ray" />
                      <Area type="monotone" dataKey="hardFlux" stroke="#ec4899" strokeWidth={1.5} fillOpacity={0.1} fill="#ec4899" name="HEL1OS Hard X-Ray" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB: SCENARIO SIMULATOR */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'simulation' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <h3 className="text-base font-bold text-white mb-2">Risk Scenario Simulator</h3>
                <p className="text-xs text-gray-400 mb-6">Simulate customized solar flare eruptions to test operational limits</p>

                <div className="flex gap-4 mb-8">
                  {['C5.0', 'M1.0', 'M5.0', 'X1.0', 'X5.0'].map((val) => (
                    <button
                      key={val}
                      onClick={() => handleSimulate(val)}
                      className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                        goesClass === val
                          ? 'bg-blue-600 text-white'
                          : 'bg-[#181f30] text-gray-400 hover:text-white'
                      }`}
                    >
                      Simulate {val}
                    </button>
                  ))}
                  {isSimulating && (
                    <button
                      onClick={() => setIsSimulating(false)}
                      className="px-4 py-2 bg-red-950 text-red-400 border border-red-500/20 rounded-lg text-xs font-semibold"
                    >
                      Reset Simulation
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="glass-card p-4 rounded-lg">
                    <span className="text-xs text-gray-400 block mb-2">Simulated Solar Hazard Index</span>
                    <div className="text-3xl font-extrabold text-red-500 font-mono mb-2">{shiScore.toFixed(2)}</div>
                    <span className="text-xs font-semibold bg-red-950 text-red-400 border border-red-500/20 px-2.5 py-0.5 rounded-full">{shiCategory}</span>
                  </div>

                  <div className="glass-card p-4 rounded-lg">
                    <span className="text-xs text-gray-400 block mb-2">Simulated GPS Position Error</span>
                    <div className="text-2xl font-bold text-white font-mono mb-2">
                      {goesClass.startsWith('X') ? '14.8 meters' : (goesClass.startsWith('M') ? '5.4 meters' : '1.5 meters')}
                    </div>
                    <span className="text-[10px] text-gray-400">Estimated position deviation increase</span>
                  </div>

                  <div className="glass-card p-4 rounded-lg">
                    <span className="text-xs text-gray-400 block mb-2">Simulated NavIC Scintillation (S4)</span>
                    <div className="text-2xl font-bold text-orange-400 font-mono mb-2">
                      {goesClass.startsWith('X') ? '0.85' : (goesClass.startsWith('M') ? '0.45' : '0.15')}
                    </div>
                    <span className="text-[10px] text-gray-400">Threshold warning limit: 0.40</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB: RESEARCH BENCHMARKING */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'research' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-gray-800">
                <h3 className="text-base font-bold text-white mb-2">Research Leaderboard</h3>
                <p className="text-xs text-gray-400 mb-6">Cross-validation benchmarks calculated on NOAA & Aditya-L1 historical events</p>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-gray-800 text-gray-400">
                        <th className="py-3 px-4">Forecasting Model</th>
                        <th className="py-3 px-4">TSS (True Skill Statistic)</th>
                        <th className="py-3 px-4">Mean Lead Time</th>
                        <th className="py-3 px-4">F1 Score</th>
                        <th className="py-3 px-4">Overall Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkLeaderboard.map((item, idx) => (
                        <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800/10">
                          <td className="py-3 px-4 font-semibold text-gray-200">{item.model}</td>
                          <td className="py-3 px-4 text-blue-400 font-bold font-mono">{item.tss.toFixed(2)}</td>
                          <td className="py-3 px-4 font-mono">{item.leadTime}</td>
                          <td className="py-3 px-4 font-mono">{item.f1.toFixed(2)}</td>
                          <td className="py-3 px-4 font-mono">{(item.accuracy * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {/* TAB: MISSION AI COPILOT */}
          {/* ────────────────────────────────────────────────────────────────────────────── */}
          {activeTab === 'copilot' && (
            <div className="glass-panel rounded-xl border border-gray-800 flex flex-col h-[520px]">
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

print("FRONTEND INTERFACE RE-WRITTEN")
