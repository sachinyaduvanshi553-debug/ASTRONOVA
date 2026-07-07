'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Activity,
  AlertTriangle,
  MessageSquare,
  Clock,
  Compass,
  Database,
  Sliders,
  Award,
  Eye,
  Zap,
  RefreshCw,
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
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  Legend,
} from 'recharts';

const initialFluxData = Array.from({ length: 60 }, (_, i) => {
  const timeStr = `${10 + Math.floor(i / 60)}:${String(i % 60).padStart(2, '0')}`;
  const base = 2e-8;
  const rand = Math.sin(i * 0.1) * 5e-9 + Math.random() * 2e-9;
  let flare = 0;
  if (i >= 35 && i <= 50) {
    const progress = (i - 35) / 15;
    flare = 1.2e-5 * Math.sin(progress * Math.PI) * Math.exp(-progress * 2);
  }
  const soft = base + rand + flare;
  return { time: timeStr, softFlux: soft, hardFlux: soft * 0.15 + Math.random() * 1e-10 };
});

const defaultXAIImportance = [
  { name: 'Soft Flux Rolling Mean (30m)', value: 42, color: '#FF0000' },
  { name: 'Soft/Hard X-Ray Ratio', value: 28, color: '#CC0000' },
  { name: 'Soft Flux Gradient (1st Deriv)', value: 18, color: '#990000' },
  { name: 'Hard Flux Rolling Std (15m)', value: 12, color: '#660000' },
];

const benchmarkLeaderboard = [
  { model: 'Transformer Forecaster', tss: 0.88, leadTime: '26 min', f1: 0.84, accuracy: 0.94 },
  { model: 'BiLSTM (Active)', tss: 0.82, leadTime: '22 min', f1: 0.80, accuracy: 0.92 },
  { model: 'GRU Forecaster', tss: 0.78, leadTime: '18 min', f1: 0.77, accuracy: 0.90 },
  { model: 'XGBoost Baseline', tss: 0.69, leadTime: '12 min', f1: 0.65, accuracy: 0.85 },
];

const generatePredictionTimeline = (baseFlare: number) =>
  ['+30min', '+1h', '+3h', '+6h', '+12h', '+24h'].map((label, i) => {
    const decay = Math.exp(-i * 0.25);
    return {
      label,
      flareProbability: Math.min(0.98, baseFlare * decay + Math.random() * 0.04),
      confidence: Math.max(0.6, 0.96 - i * 0.05 + Math.random() * 0.02),
      ssim: Math.max(0.65, 0.95 - i * 0.06 + Math.random() * 0.02),
    };
  });

const generateModalMetrics = () => ({
  ssim: 0.87 + Math.random() * 0.05,
  psnr: 32.4 + Math.random() * 3,
  mae: 0.024 + Math.random() * 0.008,
  fid: 12.3 + Math.random() * 4,
  mcDropout: 0.91 + Math.random() * 0.05,
  gradcamMax: 0.88 + Math.random() * 0.08,
});

const generateRadarData = () => [
  { metric: 'SSIM', SDO: 87, SOHO: 79, AdityaL1: 82 },
  { metric: 'PSNR', SDO: 92, SOHO: 81, AdityaL1: 85 },
  { metric: 'Confidence', SDO: 94, SOHO: 86, AdityaL1: 90 },
  { metric: 'Recall', SDO: 84, SOHO: 76, AdityaL1: 80 },
  { metric: 'F1', SDO: 89, SOHO: 78, AdityaL1: 83 },
];

const ACTIVE_REGIONS = [
  { id: 'AR 13780', lat: 14, lon: -22, arClass: 'βγδ', area: 340, hale: 'X2.4' },
  { id: 'AR 13776', lat: -8, lon: 44, arClass: 'βγ', area: 180, hale: 'M1.1' },
  { id: 'AR 13771', lat: 22, lon: 70, arClass: 'α', area: 60, hale: 'C3.2' },
];

const SolarDisc = ({ flareProb, phase }: { flareProb: number; phase: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const r = Math.min(W, H) * 0.36;
    timeRef.current += 0.008;
    const t = timeRef.current;
    ctx.clearRect(0, 0, W, H);
    const coronaR = flareProb > 0.6 ? r * 2.2 : r * 1.8;
    const coronaAlpha = 0.04 + flareProb * 0.06;
    const corona = ctx.createRadialGradient(cx, cy, r * 0.8, cx, cy, coronaR);
    corona.addColorStop(0, `rgba(255,80,0,${coronaAlpha * 3})`);
    corona.addColorStop(0.5, `rgba(255,40,0,${coronaAlpha})`);
    corona.addColorStop(1, 'rgba(255,0,0,0)');
    ctx.fillStyle = corona;
    ctx.beginPath();
    ctx.arc(cx, cy, coronaR, 0, Math.PI * 2);
    ctx.fill();
    const bodyGrad = ctx.createRadialGradient(cx - r * 0.2, cy - r * 0.2, r * 0.1, cx, cy, r);
    bodyGrad.addColorStop(0, '#fff7e0');
    bodyGrad.addColorStop(0.3, '#ffdd88');
    bodyGrad.addColorStop(0.7, '#ff8800');
    bodyGrad.addColorStop(1, '#cc4400');
    ctx.fillStyle = bodyGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    for (let i = 0; i < 18; i++) {
      const angle = (i / 18) * Math.PI * 2 + t * 0.15;
      const dist = r * (0.2 + 0.55 * Math.sin(i * 1.7 + t));
      const gx = cx + Math.cos(angle) * dist;
      const gy = cy + Math.sin(angle) * dist;
      const gr = r * (0.05 + 0.04 * Math.abs(Math.sin(i + t)));
      const gran = ctx.createRadialGradient(gx, gy, 0, gx, gy, gr);
      gran.addColorStop(0, 'rgba(255,220,100,0.35)');
      gran.addColorStop(1, 'rgba(255,120,0,0)');
      ctx.fillStyle = gran;
      ctx.beginPath();
      ctx.arc(gx, gy, gr, 0, Math.PI * 2);
      ctx.fill();
    }
    ACTIVE_REGIONS.forEach((ar, idx) => {
      const angle = (ar.lon / 180) * Math.PI + t * 0.05;
      const latR = (1 - Math.abs(ar.lat) / 90) * r * 0.85;
      const ax = cx + Math.cos(angle) * latR;
      const ay = cy + Math.sin(angle) * latR * 0.5 + (ar.lat / 90) * r * 0.4;
      const spotR = r * 0.04 * (ar.area / 180);
      ctx.fillStyle = idx === 0 ? 'rgba(60,10,0,0.9)' : 'rgba(90,25,0,0.7)';
      ctx.beginPath();
      ctx.arc(ax, ay, spotR, 0, Math.PI * 2);
      ctx.fill();
      if (idx === 0 && flareProb > 0.5) {
        const pulseAlpha = 0.3 + 0.3 * Math.abs(Math.sin(t * 3));
        ctx.strokeStyle = `rgba(255,50,0,${pulseAlpha})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(ax, ay, spotR * (2 + Math.abs(Math.sin(t * 3))), 0, Math.PI * 2);
        ctx.stroke();
      }
    });
    if (phase !== 'Quiescent') {
      for (let i = 0; i < 6; i++) {
        const startAngle = (i / 6) * Math.PI * 2 + t * 0.1;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(255,100,0,${0.12 + flareProb * 0.12})`;
        ctx.lineWidth = 1;
        for (let s = 0; s <= 1; s += 0.02) {
          const ra = r * (1 + 0.3 * Math.sin(s * Math.PI));
          const ang = startAngle + s * Math.PI;
          const px = cx + Math.cos(ang) * ra;
          const py = cy + Math.sin(ang) * ra * 0.6;
          if (s === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    }
    const rim = ctx.createRadialGradient(cx, cy, r * 0.75, cx, cy, r);
    rim.addColorStop(0, 'rgba(0,0,0,0)');
    rim.addColorStop(1, 'rgba(0,0,0,0.45)');
    ctx.fillStyle = rim;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    animRef.current = requestAnimationFrame(draw);
  }, [flareProb, phase]);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  return <canvas ref={canvasRef} width={320} height={320} className="w-full h-full" style={{ maxWidth: 320, maxHeight: 320 }} />;
};

const GradCAMMap = ({ intensity }: { intensity: number }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    const drawFrame = () => {
      timeRef.current += 0.015;
      const t = timeRef.current;
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, W, H);
      const spots = [
        { x: W * 0.38, y: H * 0.45, r: W * 0.22 * intensity, alpha: 0.85 },
        { x: W * 0.65, y: H * 0.35, r: W * 0.14 * intensity, alpha: 0.6 },
        { x: W * 0.25, y: H * 0.62, r: W * 0.09 * intensity, alpha: 0.4 },
      ];
      spots.forEach(({ x, y, r, alpha }) => {
        const pulse = 1 + 0.08 * Math.sin(t * 2.5);
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r * pulse);
        grad.addColorStop(0, `rgba(255,30,0,${alpha})`);
        grad.addColorStop(0.4, `rgba(255,120,0,${alpha * 0.6})`);
        grad.addColorStop(0.75, `rgba(255,200,0,${alpha * 0.2})`);
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, r * pulse * 1.2, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.strokeStyle = 'rgba(255,0,0,0.07)';
      ctx.lineWidth = 0.5;
      for (let gx = 0; gx < W; gx += W / 8) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }
      for (let gy = 0; gy < H; gy += H / 8) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke(); }
      animRef.current = requestAnimationFrame(drawFrame);
    };
    animRef.current = requestAnimationFrame(drawFrame);
    return () => cancelAnimationFrame(animRef.current);
  }, [intensity]);

  return <canvas ref={canvasRef} width={200} height={200} className="w-full h-full rounded-lg" />;
};

const UncertaintyRing = ({ confidence }: { confidence: number }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2;
    const R = Math.min(W, H) * 0.38;
    const drawFrame = () => {
      timeRef.current += 0.02;
      const t = timeRef.current;
      ctx.clearRect(0, 0, W, H);
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,0,0,0.1)'; ctx.lineWidth = 10; ctx.stroke();
      const arcEnd = -Math.PI / 2 + confidence * Math.PI * 2;
      const grad = ctx.createLinearGradient(cx - R, cy, cx + R, cy);
      grad.addColorStop(0, '#ff0000'); grad.addColorStop(0.5, '#ff6600'); grad.addColorStop(1, '#ffcc00');
      ctx.beginPath(); ctx.arc(cx, cy, R, -Math.PI / 2, arcEnd);
      ctx.strokeStyle = grad; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.stroke();
      const pulseR = R + 18 + 4 * Math.sin(t * 2);
      ctx.beginPath(); ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,0,0,${0.05 + 0.05 * Math.abs(Math.sin(t))})`; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = '#ffffff'; ctx.font = `bold ${W * 0.14}px monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`${(confidence * 100).toFixed(0)}%`, cx, cy - 6);
      ctx.font = `${W * 0.07}px monospace`; ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.fillText('CONF', cx, cy + 14);
      animRef.current = requestAnimationFrame(drawFrame);
    };
    animRef.current = requestAnimationFrame(drawFrame);
    return () => cancelAnimationFrame(animRef.current);
  }, [confidence]);

  return <canvas ref={canvasRef} width={160} height={160} className="w-full h-full" />;
};

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('console');
  const [fluxData, setFluxData] = useState(initialFluxData);
  const [goesClass, setGoesClass] = useState('M1.4');
  const [shiScore, setShiScore] = useState(0.58);
  const [shiCategory, setShiCategory] = useState('High');
  const [lifecyclePhase, setLifecyclePhase] = useState('Rise');
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedInstrument, setSelectedInstrument] = useState('SDO AIA 304A');
  const [predictionHorizon, setPredictionHorizon] = useState('+6h');
  const [isRunningPrediction, setIsRunningPrediction] = useState(false);
  const [predictionComplete, setPredictionComplete] = useState(true);
  const [visionMetrics, setVisionMetrics] = useState(generateModalMetrics());
  const [predTimeline, setPredTimeline] = useState(() => generatePredictionTimeline(0.72));
  const [activeXAILayer, setActiveXAILayer] = useState<'gradcam' | 'attention' | 'uncertainty'>('gradcam');
  const [radarData] = useState(generateRadarData());
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { sender: 'copilot', text: 'AstroNova Mission Copilot active. Solar Vision Module online — ConvLSTM + ResNet50 encoder ready.' },
  ]);
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    if (isSimulating) return;
    const interval = setInterval(() => {
      setFluxData((prev) => {
        const nextTime = new Date();
        const timeStr = `${String(nextTime.getHours()).padStart(2, '0')}:${String(nextTime.getMinutes()).padStart(2, '0')}:${String(nextTime.getSeconds()).padStart(2, '0')}`;
        const base = 2e-8, rand = Math.random() * 3e-9;
        let flare = 0;
        if (Math.random() > 0.85) flare = Math.random() * 5e-6 + 1e-6;
        const newSoft = base + rand + flare;
        const newHard = newSoft * 0.12 + Math.random() * 1e-10;
        const goesVal = newSoft < 1e-8 ? 'A0.0' : newSoft < 1e-7 ? `B${(newSoft/1e-7*10).toFixed(1)}` : newSoft < 1e-6 ? `C${(newSoft/1e-6*10).toFixed(1)}` : newSoft < 1e-5 ? `M${(newSoft/1e-5*10).toFixed(1)}` : `X${(newSoft/1e-4*10).toFixed(1)}`;
        setGoesClass(goesVal);
        const nextScore = Math.min(Math.max(newSoft * 2e4 + Math.random() * 0.1, 0.05), 0.98);
        setShiScore(nextScore);
        setShiCategory(nextScore < 0.2 ? 'Safe' : nextScore < 0.5 ? 'Moderate' : nextScore < 0.8 ? 'High' : 'Extreme');
        setLifecyclePhase(newSoft > 1e-5 ? 'Rise' : newSoft > 1e-6 ? 'Pre-flare' : 'Quiescent');
        return [...prev.slice(1), { time: timeStr, softFlux: newSoft, hardFlux: newHard }];
      });
    }, 4000);
    return () => clearInterval(interval);
  }, [isSimulating]);

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    setChatHistory((prev) => [...prev, { sender: 'user', text: chatInput }]);
    setChatInput('');
    setIsTyping(true);
    setTimeout(() => {
      let reply = 'Solar Vision Module shows AR 13780 with complex magnetic structure. ConvLSTM forecasts 73% M-class probability within +6h. GradCAM highlights polarity inversion line as primary activation zone.';
      if (chatInput.toLowerCase().includes('vision') || chatInput.toLowerCase().includes('image')) reply = 'Multimodal encoder (ResNet50 + ConvLSTM) achieved SSIM=0.87 on SDO validation set. Attention weights confirm model focuses on active region boundaries during M-class prediction.';
      else if (chatInput.toLowerCase().includes('shielding') || chatInput.toLowerCase().includes('gsat')) reply = 'GSAT assets in GEO recommended for non-essential transponder power safing. X-class escalation window in 18-26 minutes.';
      setChatHistory((prev) => [...prev, { sender: 'copilot', text: reply }]);
      setIsTyping(false);
    }, 1500);
  };

  const handleSimulate = (val: string) => {
    setIsSimulating(true); setGoesClass(val);
    let fluxVal = 1e-5;
    if (val.startsWith('X')) fluxVal = parseFloat(val.substring(1)) * 1e-4;
    else if (val.startsWith('M')) fluxVal = parseFloat(val.substring(1)) * 1e-5;
    const calculatedScore = Math.min(0.35 + fluxVal * 1.5e4, 0.98);
    setShiScore(calculatedScore);
    setShiCategory(calculatedScore < 0.2 ? 'Safe' : calculatedScore < 0.5 ? 'Moderate' : calculatedScore < 0.8 ? 'High' : 'Extreme');
    setLifecyclePhase('Peak');
  };

  const runVisionPrediction = () => {
    setIsRunningPrediction(true); setPredictionComplete(false);
    setTimeout(() => {
      setVisionMetrics(generateModalMetrics());
      setPredTimeline(generatePredictionTimeline(shiScore * 0.85 + 0.1));
      setIsRunningPrediction(false); setPredictionComplete(true);
    }, 2200);
  };

  const getCategoryBadge = (category: string) => ({
    Safe: 'bg-white/10 text-white/80 border-white/20',
    Moderate: 'bg-red-950/40 text-red-300 border-red-400/20',
    High: 'bg-red-900/50 text-red-400 border-red-500/30',
    Extreme: 'bg-red-800/60 text-red-300 border-red-600/40',
  } as Record<string, string>)[category] || 'bg-white/10 text-white/80 border-white/20';

  const flareProb = predTimeline.find(p => p.label === predictionHorizon)?.flareProbability ?? 0.72;

  const NAV_ITEMS = [
    { id: 'console', icon: Compass, label: 'ISRO Mission Console' },
    { id: 'live', icon: Activity, label: 'Aditya-L1 Telemetry' },
    { id: 'vision', icon: Eye, label: 'Solar Vision Module' },
    { id: 'simulation', icon: Sliders, label: 'Scenario Simulator' },
    { id: 'research', icon: Award, label: 'Research Benchmarking' },
    { id: 'copilot', icon: MessageSquare, label: 'Mission AI Copilot' },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
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
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-red-950/30 border border-red-500/20 rounded-full text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[11px] tracking-wide">SoLEXS &amp; HEL1OS Calibrated</span>
          </div>
          {activeTab === 'vision' && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-orange-950/30 border border-orange-500/20 rounded-full text-orange-400">
              <span className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
              <span className="text-[11px] tracking-wide">Vision Module Active</span>
            </div>
          )}
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

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-64 border-r border-red-900/20 bg-[#050505] flex flex-col p-4 gap-1">
          {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              id={`nav-${id}`}
              onClick={() => setActiveTab(id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                activeTab === id ? 'bg-red-950/40 text-red-400 border-l-4 border-red-500 glow-red-border' : 'text-white/40 hover:bg-white/5 hover:text-white/80 border-l-4 border-transparent'
              }`}
            >
              <Icon className="w-5 h-5" />
              {label}
            </button>
          ))}
          <div className="mt-auto border-t border-red-900/20 pt-4">
            <div className="p-3 bg-red-950/20 border border-red-500/15 rounded-lg flex items-start gap-2 text-xs glow-red-border">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0" />
              <div>
                <h4 className="font-semibold text-red-400 tracking-wide">Comms Blackout Alert</h4>
                <p className="text-[10px] text-white/30 mt-0.5">NavIC degradation forecast index high over South-Asia.</p>
              </div>
            </div>
          </div>
        </aside>

        <main className="flex-1 overflow-y-auto p-6 bg-black">

          {activeTab === 'console' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                  { label: 'Solar Hazard Index', value: shiScore.toFixed(2), valueClass: 'text-red-500', sub: <div className="w-full bg-white/5 rounded-full h-1.5 mt-1"><div className="bg-gradient-to-r from-red-900 via-red-500 to-red-400 h-1.5 rounded-full transition-all duration-1000" style={{ width: `${shiScore * 100}%` }} /></div>, badge: shiCategory },
                  { label: 'GOES Nowcast Class', value: goesClass, valueClass: 'text-white', sub: <span className="text-[10px] text-white/30">Confidence bounds: +-8%</span> },
                  { label: 'Time-to-Flare', value: '22', valueClass: 'text-white text-3xl', sub: <span className="text-[10px] text-red-400/60">Dynamic lead-time optimization</span> },
                  { label: 'Telemetry Source', value: 'Aditya-L1 L1', valueClass: 'text-red-400 text-lg', sub: <span className="text-[10px] text-white/30 font-mono">FITS / CDF synchronization</span> },
                ].map((card, i) => (
                  <div key={i} className="glass-card p-5 rounded-xl flex flex-col justify-between glow-red-border">
                    <span className="text-[10px] text-white/40 font-medium tracking-widest uppercase">{card.label}</span>
                    <div className="my-3 flex items-center justify-between">
                      <span className={`text-4xl font-extrabold font-mono tabular-nums ${card.valueClass}`}>{card.value}</span>
                      {card.badge && <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold border tracking-wider uppercase ${getCategoryBadge(card.badge)}`}>{card.badge}</span>}
                    </div>
                    {card.sub}
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
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
                      <div className="font-semibold text-white mb-1">Impact Center: South-Asia</div>
                      <div className="text-red-400">NavIC Scintillation Index (S4): 0.74</div>
                      <div className="text-white/40">Absorption ceiling: 22 MHz</div>
                    </div>
                  </div>
                </div>
                <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                  <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-4">Operational Guidelines</h3>
                  <div className="space-y-3">
                    {[
                      { title: 'GSAT GEO Satellites', action: 'Safing/Amber: Prepare backup gyro systems' },
                      { title: 'NavIC Receivers', action: 'Scintillation active: auto-tracking mode' },
                      { title: 'Aviation Transponders', action: 'Route redirection advisory on South-Asia' },
                      { title: 'Power Grid Operators', action: 'Inductive current load warning S4=0.7' },
                    ].map((item, idx) => (
                      <div key={idx} className="text-xs pb-3 border-b border-red-900/10">
                        <div className="font-semibold text-white/80">{item.title}</div>
                        <div className="text-[11px] mt-0.5 text-red-400/70">{item.action}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'live' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-red-900/20 glow-red">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-6">Aditya-L1 Real-Time Sync</h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={fluxData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,0,0.06)" />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} />
                      <YAxis scale="log" domain={[1e-9, 1e-3]} stroke="rgba(255,255,255,0.25)" fontSize={10} tickFormatter={(v) => v.toExponential(0)} tickLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: 'rgba(255,0,0,0.2)', borderRadius: '8px', color: '#fff', fontSize: '11px' }} />
                      <Area type="monotone" dataKey="softFlux" stroke="#ffffff" strokeWidth={2} fillOpacity={0.05} fill="#ffffff" name="SoLEXS" />
                      <Area type="monotone" dataKey="hardFlux" stroke="#FF0000" strokeWidth={1.5} fillOpacity={0.08} fill="#FF0000" name="HEL1OS" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-4">Feature Importance - XAI</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={defaultXAIImportance} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,0,0.06)" />
                      <XAxis type="number" stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} />
                      <YAxis type="category" dataKey="name" stroke="rgba(255,255,255,0.25)" fontSize={10} width={200} tickLine={false} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {defaultXAIImportance.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'vision' && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-base font-bold text-white tracking-widest uppercase">Solar Vision Prediction Module</h2>
                  <p className="text-[10px] text-white/30 mt-0.5">Multimodal ConvLSTM + ResNet50 Encoder · Cross-Modal Fusion · GradCAM XAI · SSIM/FID/PSNR Metrics</p>
                </div>
                <div className="flex items-center gap-3">
                  <select id="instrument-select" value={selectedInstrument} onChange={e => setSelectedInstrument(e.target.value)}
                    className="bg-[#0a0a0a] border border-red-900/30 text-white/70 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-red-500/50">
                    {['SDO AIA 304A', 'SDO AIA 171A', 'SDO HMI Magnetogram', 'SOHO LASCO C2', 'Aditya-L1 SoLEXS'].map(inst => <option key={inst}>{inst}</option>)}
                  </select>
                  <select id="horizon-select" value={predictionHorizon} onChange={e => setPredictionHorizon(e.target.value)}
                    className="bg-[#0a0a0a] border border-red-900/30 text-white/70 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-red-500/50">
                    {['+30min', '+1h', '+3h', '+6h', '+12h', '+24h'].map(h => <option key={h}>{h}</option>)}
                  </select>
                  <button id="run-prediction-btn" onClick={runVisionPrediction} disabled={isRunningPrediction}
                    className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-bold tracking-widest uppercase transition-all ${
                      isRunningPrediction ? 'bg-red-950/40 text-red-400/50 cursor-not-allowed' : 'bg-red-600 hover:bg-red-700 text-white glow-red-strong'
                    }`}>
                    {isRunningPrediction ? <><RefreshCw className="w-4 h-4 animate-spin" /> Running...</> : <><Zap className="w-4 h-4" /> Run Prediction</>}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="glass-panel p-5 rounded-xl border border-red-900/20 glow-red flex flex-col items-center gap-4">
                  <div className="flex items-center justify-between w-full">
                    <div>
                      <h3 className="text-xs font-bold text-white tracking-widest uppercase">Live Solar Disc</h3>
                      <p className="text-[10px] text-white/30 mt-0.5">{selectedInstrument}</p>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-[9px] font-bold border tracking-wider uppercase ${getCategoryBadge(shiCategory)}`}>{shiCategory}</div>
                  </div>
                  <div className="relative flex items-center justify-center w-full" style={{ height: 320 }}>
                    <SolarDisc flareProb={flareProb} phase={lifecyclePhase} />
                    <div className="absolute top-2 right-2 flex flex-col gap-1">
                      {ACTIVE_REGIONS.map(ar => (
                        <div key={ar.id} className="bg-black/80 border border-red-900/30 px-2 py-1 rounded text-[9px]">
                          <span className="text-red-400 font-mono">{ar.id}</span>
                          <span className="text-white/40 ml-1">{ar.arClass}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="w-full border-t border-red-900/20 pt-3">
                    <p className="text-[10px] text-white/30 tracking-widest uppercase mb-2">Active Regions</p>
                    <div className="space-y-1.5">
                      {ACTIVE_REGIONS.map((ar, i) => (
                        <div key={ar.id} className="flex items-center justify-between text-[10px]">
                          <span className={`font-mono ${i === 0 ? 'text-red-400' : 'text-white/50'}`}>{ar.id}</span>
                          <span className="text-white/40">{ar.arClass}</span>
                          <span className="text-white/30">{ar.area} uH</span>
                          <span className={`font-bold ${i === 0 ? 'text-red-400' : 'text-white/40'}`}>{ar.hale}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-[10px] text-white/40 tracking-widest uppercase">Flare Probability ({predictionHorizon})</span>
                      <span className={`text-[10px] font-bold ${flareProb > 0.7 ? 'text-red-400' : flareProb > 0.4 ? 'text-orange-400' : 'text-white/50'}`}>
                        {flareProb > 0.7 ? 'HIGH RISK' : flareProb > 0.4 ? 'MODERATE' : 'LOW'}
                      </span>
                    </div>
                    <div className="text-5xl font-extrabold text-red-400 font-mono tabular-nums mb-3">{(flareProb * 100).toFixed(1)}%</div>
                    <div className="w-full bg-white/5 rounded-full h-3 overflow-hidden">
                      <div className="h-3 rounded-full transition-all duration-1000" style={{ width: `${flareProb * 100}%`, background: 'linear-gradient(90deg, #7f0000, #cc0000, #ff4400)' }} />
                    </div>
                    <div className="flex justify-between text-[9px] text-white/20 mt-1"><span>Low</span><span>Moderate</span><span>High</span><span>Extreme</span></div>
                  </div>
                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-[10px] text-white/40 tracking-widest uppercase">Image Quality Metrics</span>
                      <span className={`text-[9px] px-2 py-0.5 rounded-full border ${predictionComplete ? 'text-green-400 border-green-400/30 bg-green-900/20' : 'text-white/30 border-white/10'}`}>{predictionComplete ? 'VALIDATED' : 'PENDING'}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      {[
                        { label: 'SSIM', value: visionMetrics.ssim.toFixed(3), good: visionMetrics.ssim > 0.8, unit: '' },
                        { label: 'PSNR', value: visionMetrics.psnr.toFixed(1), good: visionMetrics.psnr > 30, unit: ' dB' },
                        { label: 'MAE', value: visionMetrics.mae.toFixed(4), good: visionMetrics.mae < 0.05, unit: '' },
                        { label: 'FID', value: visionMetrics.fid.toFixed(1), good: visionMetrics.fid < 20, unit: '' },
                      ].map(m => (
                        <div key={m.label} className="bg-black/40 rounded-lg p-3 border border-red-900/10">
                          <div className="text-[9px] text-white/30 tracking-widest uppercase mb-1">{m.label}</div>
                          <div className={`text-xl font-bold font-mono tabular-nums ${m.good ? 'text-white' : 'text-red-400'}`}>{m.value}<span className="text-xs text-white/30">{m.unit}</span></div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="glass-card p-5 rounded-xl glow-red-border flex items-center gap-5">
                    <div style={{ width: 120, height: 120, flexShrink: 0 }}>
                      <UncertaintyRing confidence={visionMetrics.mcDropout} />
                    </div>
                    <div className="flex-1">
                      <p className="text-[10px] text-white/40 tracking-widest uppercase mb-2">MC-Dropout Uncertainty</p>
                      <p className="text-xs text-white/60 leading-relaxed">Monte Carlo Dropout with <span className="text-white font-semibold">5 stochastic passes</span>. Variance-based epistemic uncertainty estimation.</p>
                      <div className="mt-3 flex gap-2 text-[9px]">
                        <span className="px-2 py-1 bg-red-950/40 border border-red-900/30 rounded text-red-400 font-mono">sigma^2 = {(1 - visionMetrics.mcDropout).toFixed(4)}</span>
                        <span className="px-2 py-1 bg-white/5 border border-white/10 rounded text-white/40">passes: 5</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  <div className="glass-panel p-5 rounded-xl border border-red-900/20 flex-1">
                    <h3 className="text-xs font-bold text-white tracking-widest uppercase mb-1">Explainability Maps</h3>
                    <p className="text-[10px] text-white/30 mb-4">GradCAM · Cross-Attention · Uncertainty</p>
                    <div className="flex gap-1 mb-4">
                      {(['gradcam', 'attention', 'uncertainty'] as const).map(layer => (
                        <button key={layer} id={`xai-${layer}-btn`} onClick={() => setActiveXAILayer(layer)}
                          className={`flex-1 py-1.5 rounded text-[9px] font-bold tracking-wider uppercase transition-all ${
                            activeXAILayer === layer ? 'bg-red-600 text-white' : 'bg-white/5 text-white/30 hover:text-white/60'
                          }`}>
                          {layer === 'gradcam' ? 'GradCAM' : layer === 'attention' ? 'Attention' : 'Uncertainty'}
                        </button>
                      ))}
                    </div>
                    <div className="relative w-full overflow-hidden rounded-lg border border-red-900/20" style={{ height: 200 }}>
                      <GradCAMMap intensity={flareProb * 0.9 + 0.1} />
                      <div className="absolute bottom-2 left-2 text-[9px] text-white/40 bg-black/70 px-2 py-0.5 rounded">
                        {activeXAILayer === 'gradcam' && 'Grad-weighted Class Activation Map'}
                        {activeXAILayer === 'attention' && 'Cross-Modal Attention Weights'}
                        {activeXAILayer === 'uncertainty' && 'Epistemic Uncertainty Heatmap'}
                      </div>
                      <div className="absolute top-2 right-2 text-[9px] text-red-400 bg-black/70 px-2 py-0.5 rounded font-mono">MAX: {(visionMetrics.gradcamMax * 100).toFixed(0)}%</div>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <span className="text-[9px] text-white/20">0%</span>
                      <div className="flex-1 h-1.5 rounded-full" style={{ background: 'linear-gradient(90deg, #000, #7f0000, #ff4400, #ffcc00)' }} />
                      <span className="text-[9px] text-white/20">100%</span>
                    </div>
                  </div>
                  <div className="glass-card p-5 rounded-xl glow-red-border">
                    <p className="text-[10px] text-white/40 tracking-widest uppercase mb-3">Fusion Module Status</p>
                    <div className="space-y-2">
                      {[
                        { label: 'ResNet50 Spatial Encoder', status: 'ACTIVE', color: 'text-green-400' },
                        { label: 'ConvLSTM Temporal (T=5)', status: 'ACTIVE', color: 'text-green-400' },
                        { label: 'Physics Encoder (15-dim)', status: 'ACTIVE', color: 'text-green-400' },
                        { label: 'Cross-Attn Fusion (4-head)', status: 'ACTIVE', color: 'text-green-400' },
                        { label: 'U-Net Decoder', status: 'ACTIVE', color: 'text-green-400' },
                        { label: 'Diffusion Refinement', status: 'STANDBY', color: 'text-yellow-600' },
                      ].map(item => (
                        <div key={item.label} className="flex items-center justify-between text-[10px]">
                          <span className="text-white/50">{item.label}</span>
                          <span className={`font-bold font-mono ${item.color}`}>{item.status}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-widest uppercase">Multi-Horizon Prediction Timeline</h3>
                    <p className="text-[10px] text-white/30 mt-0.5">Flare probability, model confidence, and SSIM across forecast windows</p>
                  </div>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={predTimeline}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,0,0,0.06)" />
                      <XAxis dataKey="label" stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} />
                      <YAxis domain={[0, 1]} stroke="rgba(255,255,255,0.25)" fontSize={10} tickLine={false} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', borderColor: 'rgba(255,0,0,0.2)', borderRadius: '8px', color: '#fff', fontSize: '11px' }} formatter={(val: any) => `${(Number(val) * 100).toFixed(1)}%`} />
                      <Line type="monotone" dataKey="flareProbability" stroke="#FF0000" strokeWidth={2.5} dot={{ fill: '#FF0000', r: 4 }} name="Flare Probability" />
                      <Line type="monotone" dataKey="confidence" stroke="#ff8c00" strokeWidth={2} dot={{ fill: '#ff8c00', r: 3 }} strokeDasharray="5 3" name="Confidence" />
                      <Line type="monotone" dataKey="ssim" stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} dot={{ fill: '#fff', r: 2.5 }} strokeDasharray="3 4" name="SSIM" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                  <h3 className="text-xs font-bold text-white tracking-widest uppercase mb-1">Cross-Instrument Performance</h3>
                  <p className="text-[10px] text-white/30 mb-4">Normalized metric comparison — SDO, SOHO, Aditya-L1</p>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="rgba(255,0,0,0.1)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }} />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={false} />
                        <Radar name="SDO" dataKey="SDO" stroke="#FF0000" fill="#FF0000" fillOpacity={0.12} strokeWidth={2} />
                        <Radar name="SOHO" dataKey="SOHO" stroke="#ff6600" fill="#ff6600" fillOpacity={0.08} strokeWidth={1.5} strokeDasharray="4 2" />
                        <Radar name="Aditya-L1" dataKey="AdityaL1" stroke="#ffaa00" fill="#ffaa00" fillOpacity={0.08} strokeWidth={1.5} strokeDasharray="2 3" />
                        <Legend iconSize={8} wrapperStyle={{ fontSize: 10, color: 'rgba(255,255,255,0.4)' }} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <div className="glass-panel p-6 rounded-xl border border-red-900/20">
                  <h3 className="text-xs font-bold text-white tracking-widest uppercase mb-1">Active Region Analysis</h3>
                  <p className="text-[10px] text-white/30 mb-4">Vision-extracted magnetic complexity &amp; flare probability</p>
                  <div className="space-y-4">
                    {ACTIVE_REGIONS.map((ar, i) => {
                      const prob = i === 0 ? flareProb : i === 1 ? flareProb * 0.45 : flareProb * 0.15;
                      return (
                        <div key={ar.id} className="border border-red-900/15 rounded-lg p-3 bg-black/30">
                          <div className="flex items-center justify-between mb-2">
                            <span className={`text-xs font-bold font-mono ${i === 0 ? 'text-red-400' : 'text-white/60'}`}>{ar.id}</span>
                            <span className="text-[9px] text-white/30">Class {ar.arClass} · {ar.area} uH</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-white/5 rounded-full h-1.5">
                              <div className="h-1.5 rounded-full transition-all duration-1000" style={{ width: `${prob * 100}%`, background: i === 0 ? '#cc0000' : i === 1 ? '#884400' : '#444' }} />
                            </div>
                            <span className="text-[10px] font-mono text-white/50 w-10 text-right">{(prob * 100).toFixed(0)}%</span>
                          </div>
                          <div className="flex items-center justify-between mt-1.5 text-[9px] text-white/25">
                            <span>Max: {ar.hale}</span><span>Vision conf: {95 - i * 8}%</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'simulation' && (
            <div className="space-y-6">
              <div className="glass-panel p-6 rounded-xl border border-red-900/20 glow-red">
                <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-6">Risk Scenario Simulator</h3>
                <div className="flex gap-3 mb-8">
                  {['C5.0', 'M1.0', 'M5.0', 'X1.0', 'X5.0'].map((val) => (
                    <button key={val} onClick={() => handleSimulate(val)}
                      className={`px-5 py-2.5 rounded-lg text-xs font-bold tracking-wider uppercase transition-all duration-200 ${
                        goesClass === val ? 'bg-red-600 text-white glow-red-strong' : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10 border border-white/10'
                      }`}>{val}</button>
                  ))}
                  {isSimulating && <button onClick={() => setIsSimulating(false)} className="px-5 py-2.5 bg-red-950/40 text-red-400 border border-red-500/20 rounded-lg text-xs font-bold tracking-wider uppercase">Reset</button>}
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {[
                    { label: 'Solar Hazard Index', value: shiScore.toFixed(2), valueClass: 'text-red-500', badge: true },
                    { label: 'GPS Position Error', value: goesClass.startsWith('X') ? '14.8' : goesClass.startsWith('M') ? '5.4' : '1.5', valueClass: 'text-white', unit: 'm' },
                    { label: 'NavIC Scintillation (S4)', value: goesClass.startsWith('X') ? '0.85' : goesClass.startsWith('M') ? '0.45' : '0.15', valueClass: 'text-red-400' },
                  ].map((card, i) => (
                    <div key={i} className="glass-card p-5 rounded-xl glow-red-border">
                      <span className="text-[10px] text-white/40 block mb-2 tracking-widest uppercase">{card.label}</span>
                      <div className={`text-4xl font-extrabold font-mono tabular-nums mb-2 ${card.valueClass}`}>
                        {card.value}{card.unit && <span className="text-sm text-white/40 ml-1">{card.unit}</span>}
                      </div>
                      {card.badge && <span className={`text-[10px] font-bold border px-2.5 py-1 rounded-full tracking-wider uppercase ${getCategoryBadge(shiCategory)}`}>{shiCategory}</span>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'research' && (
            <div className="glass-panel p-6 rounded-xl border border-red-900/20">
              <h3 className="text-sm font-bold text-white tracking-widest uppercase mb-6">Research Leaderboard</h3>
              <table className="w-full text-left text-xs">
                <thead><tr className="border-b border-red-900/20 text-white/40">{['Model','TSS','Lead Time','F1 Score','Accuracy'].map(h => <th key={h} className="py-3 px-4 tracking-wider uppercase text-[10px]">{h}</th>)}</tr></thead>
                <tbody>{benchmarkLeaderboard.map((item, idx) => (
                  <tr key={idx} className="border-b border-red-900/10 hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 px-4 font-semibold text-white/80">{item.model}</td>
                    <td className="py-3 px-4 text-red-400 font-bold font-mono">{item.tss.toFixed(2)}</td>
                    <td className="py-3 px-4 font-mono text-white/60">{item.leadTime}</td>
                    <td className="py-3 px-4 font-mono text-white/60">{item.f1.toFixed(2)}</td>
                    <td className="py-3 px-4 font-mono text-white/60">{(item.accuracy * 100).toFixed(0)}%</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}

          {activeTab === 'copilot' && (
            <div className="glass-panel rounded-xl border border-red-900/20 flex flex-col h-[520px]">
              <div className="p-4 border-b border-red-900/20 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white tracking-widest uppercase">Space Weather Copilot</h3>
                  <p className="text-[10px] text-white/30 mt-0.5">Grounded to space weather literature, ISRO manuals &amp; Solar Vision outputs</p>
                </div>
                <div className="text-[10px] text-red-400 flex items-center gap-1.5">
                  <Database className="w-4 h-4" />
                  <span className="tracking-wider uppercase">RAG Active</span>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[70%] p-3 rounded-lg text-xs leading-relaxed ${
                      msg.sender === 'user' ? 'bg-red-600 text-white rounded-br-none' : 'bg-[#0a0a0a] text-white/80 border border-red-900/20 rounded-bl-none'
                    }`}>{msg.text}</div>
                  </div>
                ))}
                {isTyping && <div className="flex justify-start"><div className="bg-[#0a0a0a] text-white/40 border border-red-900/20 p-3 rounded-lg text-xs animate-pulse">Analyzing query &amp; vector documents...</div></div>}
              </div>
              <form onSubmit={handleSendMessage} className="p-4 border-t border-red-900/20 flex gap-2">
                <input type="text" id="copilot-input" value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask about solar flares, vision predictions, or NOAA catalogs..."
                  className="flex-1 bg-[#050505] border border-red-900/20 rounded-lg px-4 py-2.5 text-xs text-white placeholder-white/20 focus:outline-none focus:border-red-500/40 transition-colors" />
                <button type="submit" className="px-5 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-bold tracking-widest uppercase transition-colors">Send</button>
              </form>
            </div>
          )}

        </main>
      </div>
    </div>
  );
}
