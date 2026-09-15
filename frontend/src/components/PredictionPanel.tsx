const DIGITS = Array.from({ length: 10 }, (_, digit) => digit)

interface PredictionPanelProps {
  prediction: number
  confidence: number
  probabilities: Record<string, number>
}

export function PredictionPanel({ prediction, confidence, probabilities }: PredictionPanelProps) {
  return (
    <section className="panel">
      <h3>4. Prediction</h3>
      <p>
        Predicted digit: <strong>{prediction}</strong>
      </p>
      <p>
        Classifier confidence: <strong>{Math.round(confidence * 100)}%</strong>
      </p>
      <div className="bar-chart" role="img" aria-label="Predicted probability for each digit">
        {DIGITS.map((digit) => (
          <div className="bar-chart__column" key={digit}>
            <div className="bar-chart__bar" style={{ height: `${(probabilities[String(digit)] ?? 0) * 100}%` }} />
            <span className="bar-chart__label">{digit}</span>
          </div>
        ))}
      </div>
    </section>
  )
}
