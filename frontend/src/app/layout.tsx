import type { Metadata } from 'next';
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
