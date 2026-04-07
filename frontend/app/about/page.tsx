import Link from 'next/link'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'About | Anonmaly Detection',
  description: 'How our AI anomaly detection platform works and our privacy commitment.',
}

const modelSpecs = [
  { label: 'Architecture', value: 'DenseNet121 + Custom Classifier' },
  { label: 'Parameters', value: '8,095,054 (7.9M trainable)' },
  { label: 'Input size', value: '64 × 64 × 3 (RGB)' },
  { label: 'Output classes', value: '14 crime/anomaly categories' },
  { label: 'Training data', value: 'UCF Crime Dataset (1.26M images)' },
  { label: 'Backbone pre-training', value: 'ImageNet (frozen)' },
]

const privacyPoints = [
  'No account or login required — ever.',
  'No IP addresses are stored or logged.',
  'Uploaded media is processed in memory and not persisted after analysis.',
  'Only the structured analysis result is retained for review.',
  'This platform does not use cookies or tracking scripts.',
]

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-16">
      <div className="mb-12">
        <h1 className="text-4xl font-bold mb-4">
          About <span className="text-brand">Anonmaly Detection</span>
        </h1>
        <p className="text-xl text-white/50 leading-relaxed">
          An open platform that empowers citizens to report suspicious activity anonymously,
          augmented by state-of-the-art AI to help analysts triage submissions faster.
        </p>
      </div>

      {/* Mission */}
      <section className="glass p-8 mb-8">
        <h2 className="text-2xl font-bold mb-4">Our Mission</h2>
        <p className="text-white/60 leading-relaxed mb-4">
          Witnessing suspicious activity and not knowing what to do — or fearing retaliation
          for reporting it — is a barrier to public safety. Anonmaly Detection lowers that
          barrier by providing a fully anonymous, AI-augmented tip submission channel.
        </p>
        <p className="text-white/60 leading-relaxed">
          Every submission is classified by our anomaly detection models and, where a text
          description is provided, an LLM-powered incident report is generated to help
          human reviewers prioritize their response.
        </p>
      </section>

      {/* How the AI works */}
      <section className="mb-8">
        <h2 className="text-2xl font-bold mb-6">How the AI Works</h2>
        <div className="grid md:grid-cols-3 gap-4 mb-6">
          {[
            { step: '1', title: 'Upload', body: 'You upload an image or video of the suspicious activity.' },
            { step: '2', title: 'Classify', body: 'DenseNet121 classifies it into one of 14 crime/anomaly categories with a confidence score.' },
            { step: '3', title: 'Explain', body: 'Grad-CAM generates a heatmap showing which regions of the image triggered the classification.' },
          ].map((s) => (
            <div key={s.step} className="glass p-6">
              <div className="text-brand text-2xl font-bold mb-2">{s.step}</div>
              <h3 className="font-semibold mb-2">{s.title}</h3>
              <p className="text-white/50 text-sm leading-relaxed">{s.body}</p>
            </div>
          ))}
        </div>

        {/* Model spec table */}
        <div className="glass p-6">
          <h3 className="font-semibold mb-4 text-white/70 uppercase text-sm tracking-widest">
            Model Specifications
          </h3>
          <div className="divide-y divide-dark-border">
            {modelSpecs.map(({ label, value }) => (
              <div key={label} className="flex justify-between py-3">
                <span className="text-white/40 text-sm">{label}</span>
                <span className="text-white/80 text-sm font-medium">{value}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* LLM synthesis */}
      <section className="glass p-8 mb-8">
        <h2 className="text-2xl font-bold mb-4">AI Incident Report Synthesis</h2>
        <p className="text-white/60 leading-relaxed mb-4">
          When you include a text description with your tip, our platform sends it — along
          with the ML model&apos;s classification result — to Claude (claude-sonnet-4-6) to
          generate a structured incident report.
        </p>
        <p className="text-white/60 leading-relaxed">
          The report extracts location indicators, time references, estimates the number of
          individuals involved, assesses whether the visual AI result is consistent with
          the description, and provides a recommended next action for the human reviewer.
          This reduces manual triage time significantly.
        </p>
      </section>

      {/* Privacy */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Privacy Commitment</h2>
        <div className="glass p-6">
          <ul className="space-y-3">
            {privacyPoints.map((point) => (
              <li key={point} className="flex gap-3 text-white/60">
                <span className="text-brand mt-0.5 shrink-0">✓</span>
                {point}
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* CTA */}
      <div className="text-center">
        <Link href="/form" className="btn-brand text-lg px-10 py-4">
          Submit a Tip →
        </Link>
      </div>
    </div>
  )
}
