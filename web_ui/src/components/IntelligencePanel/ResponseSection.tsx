import type { IntelligenceResponse } from '../../types'

type ResponseSectionProps = {
  response: IntelligenceResponse
}

export function ResponseSection({ response }: ResponseSectionProps) {
  return (
    <div className="space-y-6">
      <section className="rounded-2xl p-6 bg-blue-500/10 shadow-panel-sm">
        <h4 className="text-xs font-medium text-blue-400 uppercase tracking-wider mb-3">
          Executive Summary
        </h4>
        <p className="text-base text-text-primary leading-relaxed">
          {response.executiveSummary}
        </p>
      </section>
      <section className="bg-surface-panel rounded-2xl shadow-panel p-6">
        <h4 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3">
          Detailed Analysis
        </h4>
        <p className="text-base text-text-primary leading-relaxed whitespace-pre-wrap">
          {response.detailedAnalysis}
        </p>
      </section>
      <section className="bg-surface-panel rounded-2xl shadow-panel p-6">
        <h4 className="text-xs font-medium text-text-tertiary uppercase tracking-wider mb-3">
          Key Findings
        </h4>
        <ul className="list-disc list-inside space-y-2 text-base text-text-primary leading-relaxed">
          {response.keyFindings.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      </section>
      {response.riskNotes.length > 0 && (
        <section className="bg-amber-500/10 rounded-2xl shadow-panel-sm p-6">
          <h4 className="text-xs font-medium text-amber-400 uppercase tracking-wider mb-3">
            Risk / Notes
          </h4>
          <ul className="list-disc list-inside space-y-2 text-base text-text-secondary leading-relaxed">
            {response.riskNotes.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </section>
      )}
      <div className="flex items-center justify-between pt-4">
        <span className="text-sm text-text-tertiary">
          Confidence: {(response.confidenceScore * 100).toFixed(0)}%
        </span>
        <div className="flex items-center gap-3">
          <button
            type="button"
            className="px-4 py-2.5 text-sm text-text-secondary bg-surface-panel rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
          >
            Explain this answer
          </button>
          <button
            type="button"
            className="px-4 py-2.5 text-sm text-text-secondary bg-surface-panel rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
          >
            Export report
          </button>
          <button
            type="button"
            className="px-4 py-2.5 text-sm text-text-secondary bg-surface-panel rounded-xl hover:bg-surface-hover hover:text-text-primary transition-colors duration-200"
          >
            Open source document
          </button>
        </div>
      </div>
    </div>
  )
}
