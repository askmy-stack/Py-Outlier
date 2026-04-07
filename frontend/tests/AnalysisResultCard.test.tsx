import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import AnalysisResultCard from '../components/AnalysisResultCard'

const baseResult = {
  submission_id: 'test-uuid-1234',
  media_type: 'image' as const,
  classification: {
    predicted_class: 'Fighting',
    confidence: 0.91,
    all_scores: {
      Fighting: 0.91,
      Normal: 0.04,
      Assault: 0.03,
      Abuse: 0.01,
      Arrest: 0.005,
      Arson: 0.001,
      Burglary: 0.001,
      Explosion: 0.001,
      RoadAccidents: 0.001,
      Robbery: 0.001,
      Shooting: 0.001,
      Shoplifting: 0.001,
      Stealing: 0.001,
      Vandalism: 0.001,
    },
    heatmap_url: null,
  },
  incident_report: null,
  processing_time_ms: 312.5,
}

describe('AnalysisResultCard', () => {
  it('renders the predicted class name', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.getByText('Fighting')).toBeInTheDocument()
  })

  it('shows the confidence percentage', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.getByText('91.0%')).toBeInTheDocument()
  })

  it('renders all 14 class scores', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.getByText('Normal')).toBeInTheDocument()
    expect(screen.getByText('Assault')).toBeInTheDocument()
  })

  it('does NOT show heatmap section when heatmap_url is null', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.queryByAltText(/Grad-CAM/i)).not.toBeInTheDocument()
  })

  it('shows the heatmap image when heatmap_url is provided', () => {
    const withHeatmap = {
      ...baseResult,
      classification: { ...baseResult.classification, heatmap_url: 'data:image/png;base64,abc' },
    }
    render(<AnalysisResultCard result={withHeatmap} />)
    const img = screen.getByAltText(/Grad-CAM heatmap/i)
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute('src', 'data:image/png;base64,abc')
  })

  it('shows the incident report section when provided', () => {
    const withReport = {
      ...baseResult,
      incident_report: {
        summary: 'Two individuals engaged in physical altercation.',
        threat_level: 4,
        location_indicators: ['near the bus stop'],
        time_references: ['evening'],
        actor_count: '2',
        ml_consistency: 'Consistent — model detected Fighting.',
        recommendation: 'Dispatch patrol unit to reported location.',
      },
    }
    render(<AnalysisResultCard result={withReport} />)
    expect(screen.getByText(/physical altercation/i)).toBeInTheDocument()
    expect(screen.getByText(/Dispatch patrol/i)).toBeInTheDocument()
    expect(screen.getByText('near the bus stop')).toBeInTheDocument()
  })

  it('shows the submission ID in the footer', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.getByText(/test-uuid-1234/)).toBeInTheDocument()
  })

  it('shows the processing time', () => {
    render(<AnalysisResultCard result={baseResult} />)
    expect(screen.getByText(/312 ms/)).toBeInTheDocument()
  })
})
