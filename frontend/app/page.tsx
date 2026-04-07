import Link from 'next/link'

const features = [
  {
    icon: '🧠',
    title: 'AI-Powered Analysis',
    description:
      'DenseNet121 (8M parameters, trained on UCF Crime Dataset) classifies 14 crime and anomaly types ' +
      'from images and video with high accuracy.',
  },
  {
    icon: '🔍',
    title: 'Explainable Results',
    description:
      'Grad-CAM heatmaps visually highlight exactly which regions of the image triggered the ' +
      'AI classification — no black-box decisions.',
  },
  {
    icon: '🔒',
    title: 'Fully Anonymous',
    description:
      'No login, no account, no IP logging. Your identity is never recorded. ' +
      'Tips are processed and discarded — only the analysis is retained.',
  },
]

const classes = [
  'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
  'Fighting', 'Normal', 'Road Accidents', 'Robbery', 'Shooting',
  'Shoplifting', 'Stealing', 'Vandalism',
]

export default function LandingPage() {
  return (
    <div className="max-w-6xl mx-auto px-6">
      {/* Hero */}
      <section className="py-28 text-center animate-fade-in">
        <div className="inline-block mb-4 px-4 py-1.5 rounded-full border border-brand/30 bg-brand/10 text-brand text-sm font-medium">
          AI Anomaly Detection Platform
        </div>
        <h1 className="text-5xl md:text-6xl font-bold leading-tight tracking-tight mb-6">
          Report Suspicious Activity.
          <br />
          <span className="text-brand">Anonymously.</span>
        </h1>
        <p className="text-xl text-white/50 max-w-2xl mx-auto mb-10 leading-relaxed">
          Powered by deep learning anomaly detection. Upload an image or video —
          our AI classifies the threat and generates an incident report in seconds.
          Your identity is never recorded.
        </p>
        <div className="flex gap-4 justify-center flex-wrap">
          <Link href="/form" className="btn-brand text-lg px-8 py-4">
            Submit Anonymous Tip →
          </Link>
          <Link
            href="/about"
            className="px-8 py-4 rounded-lg border border-white/20 text-white/70 hover:text-white hover:border-white/40 transition-colors text-lg"
          >
            Learn More
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="py-16">
        <h2 className="text-3xl font-bold text-center mb-12">
          How It <span className="text-brand">Works</span>
        </h2>
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((f) => (
            <div key={f.title} className="glass p-8 hover:border-brand/30 transition-colors">
              <div className="text-4xl mb-4">{f.icon}</div>
              <h3 className="text-xl font-semibold mb-3">{f.title}</h3>
              <p className="text-white/50 leading-relaxed">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Detection Classes */}
      <section className="py-16">
        <h2 className="text-3xl font-bold text-center mb-4">
          14 Anomaly <span className="text-brand">Categories</span>
        </h2>
        <p className="text-center text-white/50 mb-10">
          Trained on the UCF Crime Dataset — 1.26M images across 14 classes
        </p>
        <div className="flex flex-wrap gap-3 justify-center">
          {classes.map((cls) => (
            <span
              key={cls}
              className="px-4 py-2 rounded-full border border-dark-border bg-dark-card text-white/60 text-sm"
            >
              {cls}
            </span>
          ))}
        </div>
      </section>

      {/* CTA Banner */}
      <section className="py-16">
        <div className="glass p-12 text-center border-brand/20">
          <h2 className="text-3xl font-bold mb-4">See Something Suspicious?</h2>
          <p className="text-white/50 mb-8 text-lg">
            Submit a tip in under 60 seconds. No account required.
          </p>
          <Link href="/form" className="btn-brand text-lg px-10 py-4">
            Report Now
          </Link>
        </div>
      </section>
    </div>
  )
}
