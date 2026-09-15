import type { AccuracySummary } from '../types'

function formatPercent(value: number | null): string | null {
  return value == null ? null : `${Math.round(value * 1000) / 10}%`
}

export function AccuracyFooter({ accuracy }: { accuracy: AccuracySummary }) {
  const entries: Array<[string, number | null]> = [
    ['Input-only baseline', accuracy.inputOnly],
    ['MaleCNS hidden-state readout', accuracy.maleCns],
    ['Randomized-edge control', accuracy.randomizedControl],
  ]
  const available = entries.filter((entry): entry is [string, number] => entry[1] != null)
  if (available.length === 0) return null

  return (
    <p className="accuracy-footer">
      Experimental held-out accuracy:{' '}
      {available.map(([label, value], index) => (
        <span key={label}>
          {index > 0 && ' · '}
          {label}: {formatPercent(value)}
        </span>
      ))}
    </p>
  )
}
