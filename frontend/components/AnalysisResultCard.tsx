'use client'

interface ClassificationResult {
  predicted_class: string
  confidence: number
  all_scores: Record<string, number>
  heatmap_url: string | null
}

interface IncidentReport {
  summary: string
  threat_level: number
  location_indicators: string[]
  time_references: string[]
  actor_count: string
  ml_consistency: string
  recommendation: string
}

interface AnalysisResponse {
  submission_id: string
  media_type: 'image' | 'video'
  classification: ClassificationResult
  incident_report: IncidentReport | null
  processing_time_ms: number
}

interface Props {
  result: AnalysisResponse
}

const VIOLENT_CLASSES = new Set([
  'Abuse', 'Arson', 'Assault', 'Explosion', 'Fighting', 'Robbery', 'Shooting',
])
const MODERATE_CLASSES = new Set([
  'Arrest', 'Burglary', 'RoadAccidents', 'Shoplifting', 'Stealing', 'Vandalism',
])

function classColor(predicted: string): string {
  if (predicted === 'Normal') return 'text-emerald-400 border-emerald-400/40 bg-emerald-400/10'
  if (VIOLENT_CLASSES.has(predicted)) return 'text-red-400 border-red-400/40 bg-red-400/10'
  if (MODERATE_CLASSES.has(predicted)) return 'text-brand border-brand/40 bg-brand/10'
  return 'text-white border-white/20 bg-white/5'
}

function ThreatStars({ level }: { level: number }) {
  return (
    <div className="flex gap-1" aria-label={`Threat level ${level} of 5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className={`text-lg ${i <= level ? 'text-brand' : 'text-white/20'}`}
        >
          ★
        </span>
      ))}
    </div>
  )
}

export default function AnalysisResultCard({ result }: Props) {
  const { classification, incident_report, processing_time_ms, submission_id, media_type } = result
  const sortedScores = Object.entries(classification.all_scores).sort(([, a], [, b]) => b - a)

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header: predicted class */}
      <div className="glass p-8">
        <div className="flex items-start justify-between flex-wrap gap-4 mb-6">
          <div>
            <p className="text-white/40 text-sm uppercase tracking-widest mb-2">
              AI Classification · {media_type}
            </p>
            <span
              className={`inline-block px-5 py-2 rounded-full border text-2xl font-bold ${classColor(classification.predicted_class)}`}
            >
              {classification.predicted_class}
            </span>
          </div>
          <div className="text-right">
            <p className="text-white/40 text-sm mb-1">Confidence</p>
            <p className="text-4xl font-bold text-brand">
              {(classification.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Confidence bar */}
        <div className="w-full bg-dark-border rounded-full h-2.5 mb-8">
          <div
            className="h-2.5 rounded-full bg-brand transition-all duration-700"
            style={{ width: `${(classification.confidence * 100).toFixed(1)}%` }}
          />
        </div>

        {/* All class scores */}
        <div>
          <p className="text-white/40 text-sm mb-4 uppercase tracking-widest">All Class Scores</p>
          <div className="space-y-2">
            {sortedScores.map(([cls, score]) => (
              <div key={cls} className="flex items-center gap-3">
                <span className="w-28 text-sm text-white/50 shrink-0">{cls}</span>
                <div className="flex-1 bg-dark-border rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      cls === classification.predicted_class ? 'bg-brand' : 'bg-white/20'
                    }`}
                    style={{ width: `${(score * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-12 text-right text-xs text-white/30">
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Grad-CAM Heatmap */}
      {classification.heatmap_url && (
        <div className="glass p-6">
          <p className="text-white/40 text-sm uppercase tracking-widest mb-4">
            AI Attention Map (Grad-CAM)
          </p>
          <p className="text-white/50 text-sm mb-4">
            Highlighted regions show where the model focused to reach its classification.
          </p>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={classification.heatmap_url}
            alt="Grad-CAM heatmap showing model attention regions"
            className="rounded-lg max-w-full max-h-80 object-contain"
          />
        </div>
      )}

      {/* Incident Report */}
      {incident_report && (
        <div className="glass p-8">
          <p className="text-white/40 text-sm uppercase tracking-widest mb-6">
            AI-Generated Incident Report
          </p>

          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/50 text-sm">Threat Level</span>
            <ThreatStars level={incident_report.threat_level} />
            <span className="text-brand font-semibold">{incident_report.threat_level}/5</span>
          </div>

          <div className="space-y-5">
            <div>
              <p className="text-xs text-white/30 uppercase tracking-widest mb-1">Summary</p>
              <p className="text-white/80 leading-relaxed">{incident_report.summary}</p>
            </div>

            {incident_report.location_indicators.length > 0 && (
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-2">
                  Location Indicators
                </p>
                <div className="flex flex-wrap gap-2">
                  {incident_report.location_indicators.map((loc) => (
                    <span key={loc} className="px-3 py-1 rounded-full bg-dark-card border border-dark-border text-sm text-white/60">
                      {loc}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {incident_report.time_references.length > 0 && (
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-2">
                  Time References
                </p>
                <div className="flex flex-wrap gap-2">
                  {incident_report.time_references.map((t) => (
                    <span key={t} className="px-3 py-1 rounded-full bg-dark-card border border-dark-border text-sm text-white/60">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-1">Individuals</p>
                <p className="text-white/70">{incident_report.actor_count}</p>
              </div>
              <div>
                <p className="text-xs text-white/30 uppercase tracking-widest mb-1">
                  ML Consistency
                </p>
                <p className="text-white/70">{incident_report.ml_consistency}</p>
              </div>
            </div>

            <div className="border-t border-dark-border pt-5">
              <p className="text-xs text-white/30 uppercase tracking-widest mb-1">Recommendation</p>
              <p className="text-brand font-medium">{incident_report.recommendation}</p>
            </div>
          </div>
        </div>
      )}

      {/* Footer metadata */}
      <div className="text-center text-white/20 text-xs space-y-1">
        <p>Submission ID: {submission_id}</p>
        <p>Processed in {processing_time_ms.toFixed(0)} ms</p>
      </div>
    </div>
  )
}
