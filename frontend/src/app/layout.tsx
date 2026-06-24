import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AstroNova — ISRO Space Weather Intelligence Console',
  description: 'AI-Powered Solar Flare Forecasting & Space Weather Intelligence Platform for ISRO Mission Operations. Aditya-L1 SoLEXS & HEL1OS integrated.',
  keywords: 'ISRO, AstroNova, solar flare, space weather, Aditya-L1, SoLEXS, HEL1OS, AI forecasting',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet" />
      </head>
      <body className="antialiased bg-black text-white" style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
