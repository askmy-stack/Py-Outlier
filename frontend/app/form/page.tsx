'use client'

import { useRef, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import axios from 'axios'
import AnalysisResultCard from '@/components/AnalysisResultCard'
import Spinner from '@/components/Spinner'

// ── Types ─────────────────────────────────────────────────────────────────────
interface AnalysisResponse {
  submission_id: string
  media_type: 'image' | 'video'
  classification: {
    predicted_class: string
    confidence: number
    all_scores: Record<string, number>
    heatmap_url: string | null
  }
  incident_report: {
    summary: string
    threat_level: number
    location_indicators: string[]
    time_references: string[]
    actor_count: string
    ml_consistency: string
    recommendation: string
  } | null
  processing_time_ms: number
}

type PageState = 'idle' | 'submitting' | 'polling' | 'done' | 'error'

// ── Zod schema ────────────────────────────────────────────────────────────────
const schema = z.object({
  message: z
    .string()
    .min(10, 'Please provide at least 10 characters describing the activity.')
    .max(1000, 'Message must be under 1000 characters.'),
  mediaType: z.enum(['image', 'video']),
})

type FormValues = z.infer<typeof schema>

// ── Polling helper ─────────────────────────────────────────────────────────────
async function pollJobStatus(jobId: string, maxAttempts = 60): Promise<AnalysisResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((r) => setTimeout(r, 2000))
    const { data } = await axios.get(`/api/analyze/job/${jobId}`)
    if (data.status === 'complete') return data.result as AnalysisResponse
    if (data.status === 'failed') throw new Error(data.error ?? 'Video processing failed.')
  }
  throw new Error('Timed out waiting for video analysis.')
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function FormPage() {
  const [pageState, setPageState] = useState<PageState>('idle')
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const [errorMsg, setErrorMsg] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { mediaType: 'image' },
  })

  const mediaType = watch('mediaType')
  const messageValue = watch('message') ?? ''

  const onSubmit = async (values: FormValues) => {
    const file = fileRef.current?.files?.[0]
    if (!file) {
      setErrorMsg('Please select a file to upload.')
      return
    }

    setErrorMsg('')
    setPageState('submitting')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('message', values.message)

    try {
      if (values.mediaType === 'image') {
        const { data } = await axios.post<AnalysisResponse>('/api/analyze/image', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        })
        setResult(data)
        setPageState('done')
      } else {
        const { data: jobData } = await axios.post<{ job_id: string; status: string }>(
          '/api/analyze/video',
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        setPageState('polling')
        const analysisResult = await pollJobStatus(jobData.job_id)
        setResult(analysisResult)
        setPageState('done')
      }
    } catch (err: unknown) {
      const message =
        axios.isAxiosError(err)
          ? err.response?.data?.detail ?? err.message
          : 'An unexpected error occurred.'
      setErrorMsg(message)
      setPageState('error')
    }
  }

  const reset = () => {
    setPageState('idle')
    setResult(null)
    setErrorMsg('')
    if (fileRef.current) fileRef.current.value = ''
  }

  if (pageState === 'done' && result) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-16">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Analysis <span className="text-brand">Complete</span></h1>
          <button onClick={reset} className="text-sm text-white/40 hover:text-white transition-colors">
            ← Submit Another
          </button>
        </div>
        <AnalysisResultCard result={result} />
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-16">
      <div className="mb-10">
        <h1 className="text-4xl font-bold mb-3">
          Submit <span className="text-brand">Anonymous Tip</span>
        </h1>
        <p className="text-white/50">
          Your submission is processed by AI and reviewed by trained analysts. No personal information is collected.
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Message */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-2">
            Describe the suspicious activity *
          </label>
          <textarea
            {...register('message')}
            rows={5}
            placeholder="Describe what you witnessed: location, time, number of people, what happened..."
            className="input-field resize-none"
          />
          <div className="flex justify-between mt-1">
            {errors.message ? (
              <p className="text-red-400 text-xs">{errors.message.message}</p>
            ) : (
              <span />
            )}
            <span className="text-white/20 text-xs">{messageValue.length}/1000</span>
          </div>
        </div>

        {/* Media type toggle */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">Media type *</label>
          <div className="flex gap-4">
            {(['image', 'video'] as const).map((type) => (
              <label
                key={type}
                className={`flex items-center gap-2 px-5 py-3 rounded-lg border cursor-pointer transition-colors ${
                  mediaType === type
                    ? 'border-brand bg-brand/10 text-brand'
                    : 'border-dark-border text-white/50 hover:border-white/30'
                }`}
              >
                <input type="radio" value={type} {...register('mediaType')} className="hidden" />
                <span className="text-lg">{type === 'image' ? '🖼️' : '🎥'}</span>
                <span className="font-medium capitalize">{type}</span>
              </label>
            ))}
          </div>
        </div>

        {/* File upload */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-2">
            Upload {mediaType} *
          </label>
          <input
            ref={fileRef}
            type="file"
            accept={mediaType === 'image' ? 'image/jpeg,image/png,image/webp' : 'video/mp4,video/avi,video/quicktime'}
            className="w-full text-white/50 file:mr-4 file:py-2 file:px-4
                       file:rounded-lg file:border-0 file:text-sm file:font-semibold
                       file:bg-brand file:text-white hover:file:bg-orange-500
                       file:cursor-pointer cursor-pointer"
          />
          <p className="text-white/20 text-xs mt-1">
            Max 50 MB · {mediaType === 'image' ? 'JPEG, PNG, WebP' : 'MP4, AVI, MOV'}
          </p>
        </div>

        {/* Error message */}
        {(pageState === 'error' || errorMsg) && (
          <div className="p-4 rounded-lg border border-red-400/30 bg-red-400/10 text-red-400 text-sm">
            {errorMsg}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={pageState === 'submitting' || pageState === 'polling'}
          className="btn-brand w-full flex items-center justify-center gap-3 text-base"
        >
          {pageState === 'submitting' && (
            <>
              <Spinner size={20} color="white" />
              Analyzing…
            </>
          )}
          {pageState === 'polling' && (
            <>
              <Spinner size={20} color="white" />
              Processing video (this may take 30–60s)…
            </>
          )}
          {(pageState === 'idle' || pageState === 'error') && 'Submit Anonymous Tip →'}
        </button>

        <p className="text-center text-white/20 text-xs">
          By submitting, you confirm this report is made in good faith.
          False reports may be subject to local laws.
        </p>
      </form>
    </div>
  )
}
