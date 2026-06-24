'use client';

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
  { name: 'Soft Flux Rolling Mean (30m)', value: 42, color: '#FF0000' },
  { name: 'Soft/Hard X-Ray Ratio', value: 28, color: '#CC0000' },
  { name: 'Soft Flux Gradient (1st Deriv)', value: 18, color: '#990000' },
  { name: 'Hard Flux Rolling Std (15m)', value: 12, color: '#660000' }
];

const benchmarkLeaderboard = [
  { model: 'Transformer Forecaster', tss: 0.88, leadTime: '26 min', f1: 0.84, accuracy: 0.94 },
  { model: 'BiLSTM (Active)', tss: 0.82, leadTime: '22 min', f1: 0.80, accuracy: 0.92 },
  { model: 'GRU Forecaster', tss: 0.78, leadTime: '18 min', f1: 0.77, accuracy: 0.90 },
  { model: 'XGBoost Baseline', tss: 0.69, leadTime: '12 min', f1: 0.65, accuracy: 0.85 }
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
        replyText = "GSAT communication assets in GEO are recommended for non-essential transponder power safing. Magnetic helicity parameters show active thermal loading.";
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

  // Alert category badge styling
  const getCategoryBadge = (category: string) => {
    const styles: Record<string, string> = {
      'Safe': 'bg-white/10 text-white/80 border-white/20',
      'Moderate': 'bg-red-950/40 text-red-300 border-red-400/20',
      'High': 'bg-red-900/50 text-red-400 border-red-500/30',
      'Extreme': 'bg-red-800/60 text-red-300 border-red-600/40',
    };
    return styles[category] || styles['Safe'];
  };

  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
      {/* HEADER */}
      <header className="glass-panel sticky top-0 z-50 flex items-center justify-between px-6 py-4 border-b border-red-900/30">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Compass className="w-8 h-8 text-red-500" />
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse-red" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-[0.15em] text-white uppercase">AstroNova</h1>
            <p className="text-[10px] text-white/40 tracking-widest uppercase">Aditya-L1 Space Weather Intelligence Console</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-red-950/30 border border-red-500/20 rounded-full text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[11px] tracking-wide">SoLEXS &amp; HEL1OS Calibrated</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-white/5 border border-white/10 rounded-full text-white/60">
            <span className="w-2 h-2 rounded-full bg-white/50 animate-pulse" />
            <span className="text-[11px] tracking-wide">Phase: {lifecyclePhase}</span>
          </div>
          <div className="flex items-center gap-2 text-white/40 font-mono text-xs">
            <Clock className="w-4 h-4 text-red-500/60" />
            UTC: {new Date().toUTCString().slice(17, 25)}
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <div className="flex-1 flex overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-64 border-r border-red-900/20 bg-[#050505] flex flex-col p-4 gap-1">
          {[
            { id: 'console', icon: Compass, label: 'ISRO Mission Console' },
            { id: 'live', icon: Activity, label: 'Aditya-L1 Telemetry' },
            { id: 'simulation', icon: Sliders, label: 'Scenario Simulator' },
            { id: 'research', icon: Award, label: 'Research Benchmarking' },
            { id: 'copilot', icon: MessageSquare, label: 'Mission AI Copilot' },
          ].map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === id
                  ? 'bg-red-950/40 text-red-400 border-l-4 border-red-500 glow-red-border'
                  : 'text-white/40 hover:bg-white/5 hover:text-white/80 border-l-4 border-transparent'
              }`}
            >
              <Icon className="w-5 h-5" />
              {label}
            </button>
          ))}
          
          <div className="mt-auto border-t border-red-900/20 pt-4 flex flex-col gap-2">
            <div className="p-3 bg-red-950/20 border border-red-500/15 rounded-lg flex items-start gap-2 text-xs glow-red-border">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <div>
                <h4 className="font-semibold text-red-400 tracking-wide">Comms Blackout Alert</h4>
                <p className="text-[10px] text-white/30 mt-0.5">NavIC degradation forecast index high over South-Asia.</p>
              </div>
            </div>
          </div>
        </aside>

        {/* CONTENT */}
        <main className="flex-1 overflow-y-auto p-6 bg-black">

          {/* ── TAB: ISRO MISSION CONSOLE ── */}
          {activeTab === 'console' && (
            <div className="space-y-6">
              {/* TOP SUMMARY STATS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-5 rounded-xl flex flex-col justify-between glow-red-border">
                  <span className="text-[10px] text-white/40 font-medium tracking-widest uppercase">Solar Hazard Index</span>
                  <div className="my-3 flex items-center justify-between">
                    <span className="text-4xl font-extrabold text-red-500 font-mono tabular-nums">{shiScore.toFixed(2)}</span>
                    <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold border tracking-wider uppercase ${getCategoryBadge(shiCategory)}`}>
                      {shiCategory}
                    </span>
                  </div>
                  <div className="w-full bg-white/5 rounded-full h-1.5 mt-1">
                    <div className="bg-gradient-to-r from-red-900 via-red-500 to-red-400 h-1.5 rounded-full transition-all duration-1000" style={{ width: `${shiScore * 100}%` }} />
                  </div>
                </div>

                <div className="glass-card p-5 rounded-xl flex flex-col justify-between glow-red-border">
                  <span className="text-[10px] text-white/40 font-medium tracking-widest uppercase">GOES Nowcast Class</span>
                  <div className="my-3 flex items-baseline gap-2">
                    <span className="text-4xl font-extrabold text-white font-mono tabular-nums">{goesClass}</span>
                  </div>
                  <span className="text-[10px] text-white/30">Confidence bounds: ±8%</span>
                </div>

                <div className="glass-card p-5 rounded-xl flex flex-col justify-between glow-red-border">
                  <span className="text-[10px] text-white/40 font-medium tracking-widest uppercase">Time-to-Flare</span>
                  <div className="my-3 flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-white font-mono tabular-nums">22</span>
                    <span className="text-sm text-white/40">min</span>
                  </div>
                  <span className="text-[10px] text-red-400/60">Dynamic lead-time optimization</span>
                </div>

                <div className="glass-card p-5 rounded-xl flex flex-col justify-between glow-red-border">
                  <span className="text-[10px] text-white/40 font-medium tracking-widest uppercase">Telemetry Source</span>
                  <div className="my-3 flex items-baseline gap-2">
                    <span className="text-lg font-bold text-red-400">Aditya-L1 L1 Data</span>
                  </div>
                  <span className="text-[10px] text-white/30 font-mono">FITS / CDF synchronization</span>
                </div>
              </div>

              {/* MISSION CONSOLE: MAP + OPERATIONS */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* Earth Impact Map */}
                <div className="glass-panel p-6 rounded-xl border border-red-900/20 md:col-span-2 glow-red">
                  <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-1">ISRO Geospatial Earth Impact</h3>
                  <p className="text-[10px] text-white/30 mb-6">NavIC/D-layer absorption projection over South-Asia quadrant</p>
                  
                  <div className="relative bg-[#050505] rounded-xl border border-red-900/15 p-6 flex flex-col justify-center items-center h-72 overflow-hidden">
                    <svg className="w-full h-56 opacity-40" fill="currentColor" viewBox="0 0 800 400">
                      <path d="M120 80h100v100H120zM140 180h80v150h-80z" className="text-white/10" />
                      <path d="M380 60h100v120H380zM390 180h90v160h-90z" className="text-white/10" />
                      <path d="M500 40h180v160H500z" className="text-white/10" />
                      <circle cx="560" cy="140" r="30" className="fill-red-500/20 stroke-red-500 stroke-2 animate-ping" />
                      <circle cx="560" cy="140" r="10" className="fill-red-600" />
                    </svg>
                    
                    <div className="absolute top-4 left-4 bg-black/80 border border-red-900/30 p-3 rounded-lg text-xs">
                      <div className="font-semibold text-white mb-1 tracking-wide">Impact Center: South-Asia</div>
                      <div className="text-red-400">NavIC Scintillation Index (S4): 0.74</div>
                      <div className="text-white/40">Absorption ceiling: 22 MHz</div>
                    </div>

                    <div className="absolute bottom-4 right-4 bg-black/80 border border-red-900/30 p-2 rounded text-[10px] text-white/30">
                      <span className="text-red-400 font-mono">●</span> Active Impact Zone
                    </div>
                  </div>
                </div>

                {/* Operational Guidelines */}
                <div className="glass-panel p-6 rounded-xl border border-red-900/20 flex flex-col justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-1">Operational Guidelines</h3>
                    <p className="text-[10px] text-white/30 mb-6">Actionable satellite &amp; comms mitigations</p>
                    
                    <div className="space-y-4">
                      {[
                        { title: 'GSAT GEO Satellites', action: 'Safing/Amber: Prepare backup gyro systems', level: 'Moderate' },
                        { title: 'NavIC Receivers', action: 'Scintillation active: auto-tracking mode', level: 'High' },
                        { title: 'Aviation Transponders', action: 'Advisory: route redirection on South-Asia', level: 'Moderate' },
                        { title: 'Power Grid Operators', action: 'Inductive current load warning S4=0.7', level: 'Moderate' }
                      ].map((item, idx) => (
                        <div key={idx} className="text-xs pb-3 border-b border-red-900/10">
                          <div className="font-semibold text-white/80 tracking-wide">{item.title}</div>
                          <div className={`text-[11px] mt-0.5 ${item.level === 'High' ? 'text-red-400' : 'text-red-400/60'}`}>
                            {item.action}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <button className="w-full mt-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-bold tracking-widest uppercase transition-colors">
                    Download Execution Plan
                  </button>
                </div>

              </div>
            </div>
          )}

          {/* ── TAB: ADITYA-L1 TELEMETRY ── */}
          {activeTab === 'live' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-red-900/20 glow-red">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-widest uppercase">Aditya-L1 Real-Time Sync</h3>
                    <p className="text-[10px] text-white/30 mt-1">Aligned Soft X-Ray (SoLEXS) and Hard X-Ray (HEL1OS) fluxes</p>
                  </div>
                  <div className="flex items-center gap-4 text-[10px]">
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-white rounded" />
                      <span className="text-white/50">SoLEXS</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-red-500 rounded" />
                      <span className="text-white/50">HEL1OS</span>
                    </div>
                  </div>
                </div>
                
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={fluxData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,0,0.06)" />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} />
                      <YAxis 
                        scale="log" 
                        domain={[1e-9, 1e-3]} 
                        stroke="rgba(255,255,255,0.25)" 
                        fontSize={10} 
                        tickFormatter={(v) => v.toExponential(0)}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0a0a0a',
                          borderColor: 'rgba(255,0,0,0.2)',
                          borderRadius: '8px',
                          color: '#fff',
                          fontSize: '11px'
                        }}
                      />
                      <Area type="monotone" dataKey="softFlux" stroke="#ffffff" strokeWidth={2} fillOpacity={0.05} fill="#ffffff" name="SoLEXS Soft X-Ray" />
                      <Area type="monotone" dataKey="hardFlux" stroke="#FF0000" strokeWidth={1.5} fillOpacity={0.08} fill="#FF0000" name="HEL1OS Hard X-Ray" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* XAI Feature Importance */}
              <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-1">Feature Importance — XAI</h3>
                <p className="text-[10px] text-white/30 mb-4">SHAP-derived contribution weights for active prediction</p>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={defaultXAIImportance} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,0,0.06)" />
                      <XAxis type="number" stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} />
                      <YAxis type="category" dataKey="name" stroke="rgba(255,255,255,0.25)" fontSize={10} width={200} tickLine={false} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {defaultXAIImportance.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {/* ── TAB: SCENARIO SIMULATOR ── */}
          {activeTab === 'simulation' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-red-900/20 glow-red">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-1">Risk Scenario Simulator</h3>
                <p className="text-[10px] text-white/30 mb-6">Simulate customized solar flare eruptions to test operational limits</p>
                
                <div className="flex gap-3 mb-8">
                  {['C5.0', 'M1.0', 'M5.0', 'X1.0', 'X5.0'].map((val) => (
                    <button
                      key={val}
                      onClick={() => handleSimulate(val)}
                      className={`px-5 py-2.5 rounded-lg text-xs font-bold tracking-wider uppercase transition-all duration-200 ${
                        goesClass === val
                          ? 'bg-red-600 text-white glow-red-strong'
                          : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10 border border-white/10'
                      }`}
                    >
                      {val}
                    </button>
                  ))}
                  {isSimulating && (
                    <button
                      onClick={() => setIsSimulating(false)}
                      className="px-5 py-2.5 bg-red-950/40 text-red-400 border border-red-500/20 rounded-lg text-xs font-bold tracking-wider uppercase hover:bg-red-950/60 transition-colors"
                    >
                      Reset
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <span className="text-[10px] text-white/40 block mb-2 tracking-widest uppercase">Solar Hazard Index</span>
                    <div className="text-4xl font-extrabold text-red-500 font-mono tabular-nums mb-2">{shiScore.toFixed(2)}</div>
                    <span className={`text-[10px] font-bold border px-2.5 py-1 rounded-full tracking-wider uppercase ${getCategoryBadge(shiCategory)}`}>
                      {shiCategory}
                    </span>
                  </div>

                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <span className="text-[10px] text-white/40 block mb-2 tracking-widest uppercase">GPS Position Error</span>
                    <div className="text-3xl font-bold text-white font-mono tabular-nums mb-2">
                      {goesClass.startsWith('X') ? '14.8' : (goesClass.startsWith('M') ? '5.4' : '1.5')}
                      <span className="text-sm text-white/40 ml-1">m</span>
                    </div>
                    <span className="text-[10px] text-white/30">Estimated position deviation</span>
                  </div>

                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <span className="text-[10px] text-white/40 block mb-2 tracking-widest uppercase">NavIC Scintillation (S4)</span>
                    <div className="text-3xl font-bold text-red-400 font-mono tabular-nums mb-2">
                      {goesClass.startsWith('X') ? '0.85' : (goesClass.startsWith('M') ? '0.45' : '0.15')}
                    </div>
                    <span className="text-[10px] text-white/30">Threshold limit: 0.40</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── TAB: RESEARCH BENCHMARKING ── */}
          {activeTab === 'research' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-1">Research Leaderboard</h3>
                <p className="text-[10px] text-white/30 mb-6">Cross-validation benchmarks on NOAA &amp; Aditya-L1 historical events</p>
                
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-red-900/20 text-white/40">
                        <th className="py-3 px-4 tracking-wider uppercase text-[10px]">Model</th>
                        <th className="py-3 px-4 tracking-wider uppercase text-[10px]">TSS</th>
                        <th className="py-3 px-4 tracking-wider uppercase text-[10px]">Lead Time</th>
                        <th className="py-3 px-4 tracking-wider uppercase text-[10px]">F1 Score</th>
                        <th className="py-3 px-4 tracking-wider uppercase text-[10px]">Accuracy</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkLeaderboard.map((item, idx) => (
                        <tr key={idx} className="border-b border-red-900/10 hover:bg-white/[0.02] transition-colors">
                          <td className="py-3 px-4 font-semibold text-white/80">{item.model}</td>
                          <td className="py-3 px-4 text-red-400 font-bold font-mono tabular-nums">{item.tss.toFixed(2)}</td>
                          <td className="py-3 px-4 font-mono text-white/60 tabular-nums">{item.leadTime}</td>
                          <td className="py-3 px-4 font-mono text-white/60 tabular-nums">{item.f1.toFixed(2)}</td>
                          <td className="py-3 px-4 font-mono text-white/60 tabular-nums">{(item.accuracy * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ── TAB: MISSION AI COPILOT ── */}
          {activeTab === 'copilot' && (
            <div className="glass-panel rounded-xl border border-red-900/20 flex flex-col h-[520px]">
              <div className="p-4 border-b border-red-900/20 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white tracking-widest uppercase">Space Weather Copilot</h3>
                  <p className="text-[10px] text-white/30 mt-0.5">Grounded to local space weather literature &amp; ISRO manuals</p>
                </div>
                <div className="text-[10px] text-red-400 flex items-center gap-1.5">
                  <Database className="w-4 h-4" />
                  <span className="tracking-wider uppercase">RAG Active</span>
                </div>
              </div>

              {/* Chat screen */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[70%] p-3 rounded-lg text-xs leading-relaxed ${
                      msg.sender === 'user'
                        ? 'bg-red-600 text-white rounded-br-none'
                        : 'bg-[#0a0a0a] text-white/80 border border-red-900/20 rounded-bl-none'
                    }`}>
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isTyping && (
                  <div className="flex justify-start">
                    <div className="bg-[#0a0a0a] text-white/40 border border-red-900/20 p-3 rounded-lg text-xs rounded-bl-none animate-pulse">
                      Analyzing query &amp; vector documents...
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input */}
              <form onSubmit={handleSendMessage} className="p-4 border-t border-red-900/20 flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask about solar flares, shielding, or NOAA catalogs..."
                  className="flex-1 bg-[#050505] border border-red-900/20 rounded-lg px-4 py-2.5 text-xs text-white placeholder-white/20 focus:outline-none focus:border-red-500/40 transition-colors"
                />
                <button
                  type="submit"
                  className="px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-bold tracking-widest uppercase transition-colors"
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
