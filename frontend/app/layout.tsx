import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Anonmaly Detection | Report Suspicious Activity',
  description:
    'Anonymous, secure tip submission platform powered by AI anomaly detection. ' +
    'Your identity is never recorded.',
  keywords: ['anomaly detection', 'anonymous tip', 'crime reporting', 'AI', 'safety'],
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-dark text-white font-sans antialiased">
        <nav className="sticky top-0 z-50 border-b border-dark-border bg-dark/80 backdrop-blur-md">
          <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
            <a href="/" className="text-xl font-bold tracking-tight">
              <span className="text-brand">Anonmaly</span>
              <span className="text-white/70"> Detection</span>
            </a>
            <div className="flex gap-6 text-sm text-white/60">
              <a href="/" className="hover:text-white transition-colors">Home</a>
              <a href="/about" className="hover:text-white transition-colors">About</a>
              <a href="/form" className="hover:text-white transition-colors">Report</a>
            </div>
          </div>
        </nav>
        <main>{children}</main>
        <footer className="border-t border-dark-border mt-24 py-8 text-center text-white/30 text-sm">
          Built with TensorFlow, FastAPI &amp; Next.js &nbsp;·&nbsp; Anonymous by design
        </footer>
      </body>
    </html>
  )
}
