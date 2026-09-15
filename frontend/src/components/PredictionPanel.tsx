const DIGITS = Array.from({ length: 10 }, (_, digit) => digit)

interface PredictionPanelProps {
  prediction: number
  confidence: number
  probabilities: Record<string, number>
}

export function PredictionPanel({ prediction, confidence, probabilities }: PredictionPanelProps) {
  return (
    <section className="panel panel--prediction">
      <p className="panel__index">04 / Readout</p>
      <h3>Classifier output</h3>
      <p className="prediction-summary">
        <strong>{prediction}</strong>
        <span>{Math.round(confidence * 100)}% confidence</span>
      </p>
      <div className="bar-chart" role="img" aria-label="Predicted probability for each digit">
        {DIGITS.map((digit) => (
          <div className="bar-chart__column" key={digit}>
            <div
              className={`bar-chart__bar${digit === prediction ? ' bar-chart__bar--active' : ''}`}
              style={{ height: `${(probabilities[String(digit)] ?? 0) * 100}%` }}
            />
            <span className="bar-chart__label">{digit}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
