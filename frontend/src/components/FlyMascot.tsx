import { useId } from 'react'

type Expression = 'neutral' | 'happy' | 'unsure'

interface FlyMascotProps {
  prediction: number | null
  confidence: number
  isLoading: boolean
  error: string | null
}

function Mouth({ expression }: { expression: Expression }) {
  if (expression === 'happy') {
    return <path d="M 88 104 Q 110 122 132 104" stroke="#3a2a1a" strokeWidth={4} fill="none" strokeLinecap="round" />
  }
  if (expression === 'unsure') {
    return (
      <path
        d="M 88 110 Q 98 100 110 110 Q 122 120 132 110"
        stroke="#3a2a1a"
        strokeWidth={4}
        fill="none"
        strokeLinecap="round"
      />
    )
  }
  return <path d="M 92 108 Q 110 108 128 108" stroke="#3a2a1a" strokeWidth={4} fill="none" strokeLinecap="round" />
}

function FlySvg({ expression }: { expression: Expression }) {
  // Unique per mount so gradient ids never collide if the mascot ever renders twice.
  const uid = useId()
  const bodyGradientId = `${uid}-body`
  const eyeGradientId = `${uid}-eye`
  const wingGradientId = `${uid}-wing`

  return (
    <svg viewBox="0 0 220 160" width={120} height={87} className="fly-mascot__svg" role="img" aria-label="Cartoon fly mascot">
      <defs>
        <radialGradient id={bodyGradientId} cx="35%" cy="30%" r="75%">
          <stop offset="0%" stopColor="#8a6a4a" />
          <stop offset="100%" stopColor="#3a2a1a" />
        </radialGradient>
        <radialGradient id={eyeGradientId} cx="35%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#e0566c" />
          <stop offset="100%" stopColor="#7a1020" />
        </radialGradient>
        <linearGradient id={wingGradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ffffff" stopOpacity={0.85} />
          <stop offset="100%" stopColor="#bfe4fa" stopOpacity={0.4} />
        </linearGradient>
      </defs>

      <g stroke="#9fd3ee" strokeWidth={1.5} fill={`url(#${wingGradientId})`}>
        <path d="M 100 80 C 60 55, 30 60, 20 95 C 45 105, 80 100, 100 80 Z">
          <animateTransform
            attributeName="transform"
            type="rotate"
            values="-6 100 80;8 100 80;-6 100 80"
            dur="0.5s"
            repeatCount="indefinite"
          />
        </path>
        <path d="M 120 80 C 160 55, 190 60, 200 95 C 175 105, 140 100, 120 80 Z">
          <animateTransform
            attributeName="transform"
            type="rotate"
            values="6 120 80;-8 120 80;6 120 80"
            dur="0.5s"
            repeatCount="indefinite"
          />
        </path>
      </g>

      <line x1="95" x2="80" y1="45" y2="18" stroke="#3a2a1a" strokeWidth={3} strokeLinecap="round" />
      <line x1="125" x2="140" y1="45" y2="18" stroke="#3a2a1a" strokeWidth={3} strokeLinecap="round" />
      <circle cx="80" cy="18" r="4" fill="#3a2a1a" />
      <circle cx="140" cy="18" r="4" fill="#3a2a1a" />

      <ellipse cx="110" cy="95" rx="45" ry="42" fill={`url(#${bodyGradientId})`} />

      <circle cx="85" cy="80" r="22" fill={`url(#${eyeGradientId})`} />
      <circle cx="135" cy="80" r="22" fill={`url(#${eyeGradientId})`} />
      <circle cx="78" cy="72" r="6" fill="#ffffff" fillOpacity={0.75} />
      <circle cx="128" cy="72" r="6" fill="#ffffff" fillOpacity={0.75} />

      <Mouth expression={expression} />

      <line x1="75" x2="55" y1="120" y2="140" stroke="#3a2a1a" strokeWidth={4} strokeLinecap="round" />
      <line x1="145" x2="165" y1="120" y2="140" stroke="#3a2a1a" strokeWidth={4} strokeLinecap="round" />
      <line x1="85" x2="72" y1="132" y2="154" stroke="#3a2a1a" strokeWidth={4} strokeLinecap="round" />
      <line x1="135" x2="148" y1="132" y2="154" stroke="#3a2a1a" strokeWidth={4} strokeLinecap="round" />
    </svg>
  )
}

export function FlyMascot({ prediction, confidence, isLoading, error }: FlyMascotProps) {
  const hasPrediction = prediction !== null
  let expression: Expression = 'neutral'
  let message: React.ReactNode = 'Draw a digit for me!'
  let sub: string | null = null

  if (error) {
    message = "I couldn't think that through."
    sub = error
  } else if (isLoading && !hasPrediction) {
    message = 'Thinking…'
  } else if (hasPrediction) {
    const percent = Math.round(confidence * 100)
    if (confidence >= 0.5) {
      expression = 'happy'
      message = (
        <>
          It's a <strong>{prediction}</strong>!
        </>
      )
      sub = `${percent}% sure`
    } else {
      expression = 'unsure'
      message = (
        <>
          Hmm… maybe a <strong>{prediction}</strong>?
        </>
      )
      sub = `only ${percent}% sure`
    }
  }

  return (
    <div className="fly-mascot">
      <FlySvg expression={expression} />
      <div className="fly-mascot__bubble">
        <div className="fly-mascot__message">{message}</div>
        {sub && <div className="fly-mascot__sub">{sub}</div>}
      </div>
    </div>
  )
}
