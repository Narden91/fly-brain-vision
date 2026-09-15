import { useId } from 'react'

type Mood = 'idle' | 'thinking' | 'happy' | 'unsure' | 'error'

interface FlyMascotProps {
  prediction: number | null
  confidence: number
  isLoading: boolean
  error: string | null
}

function Mouth({ mood }: { mood: Mood }) {
  if (mood === 'happy') return <path d="M92 111 Q110 127 128 111" className="fly-mascot__line" />
  if (mood === 'unsure') return <path d="M91 115 Q101 105 110 115 Q120 125 129 115" className="fly-mascot__line" />
  if (mood === 'error') return <path d="M94 120 Q110 105 126 120" className="fly-mascot__line" />
  return <path d="M94 113 Q110 116 126 113" className="fly-mascot__line" />
}

function FlyModel({ mood }: { mood: Mood }) {
  const uid = useId()
  const bodyId = `${uid}-body`
  const eyeId = `${uid}-eye`
  const wingId = `${uid}-wing`

  return (
    <svg className="fly-mascot__model" viewBox="0 0 220 180" role="img" aria-label="Animated fly assistant">
      <defs>
        <radialGradient id={bodyId} cx="34%" cy="24%" r="72%">
          <stop offset="0%" stopColor="#e0ad72" />
          <stop offset="48%" stopColor="#9b623d" />
          <stop offset="100%" stopColor="#3b281e" />
        </radialGradient>
        <radialGradient id={eyeId} cx="32%" cy="27%" r="74%">
          <stop offset="0%" stopColor="#ffb4b8" />
          <stop offset="50%" stopColor="#d65463" />
          <stop offset="100%" stopColor="#6c1828" />
        </radialGradient>
        <linearGradient id={wingId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#fffef8" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#a5d9e5" stopOpacity="0.35" />
        </linearGradient>
        <filter id={`${uid}-shadow`} x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="7" stdDeviation="5" floodColor="#302019" floodOpacity="0.28" />
        </filter>
      </defs>

      <ellipse className="fly-mascot__shadow" cx="110" cy="160" rx="54" ry="10" />
      <g className="fly-mascot__wings" fill={`url(#${wingId})`} stroke="#80b8c6" strokeWidth="1.4">
        <path d="M96 88 C56 54 24 63 21 99 C50 112 82 107 103 88 Z" />
        <path d="M124 88 C164 54 196 63 199 99 C170 112 138 107 117 88 Z" />
      </g>
      <g className="fly-mascot__body" filter={`url(#${uid}-shadow)`}>
        <path d="M96 52 L81 25 M124 52 L139 25" className="fly-mascot__antenna" />
        <circle cx="81" cy="25" r="4.5" className="fly-mascot__antenna-tip" />
        <circle cx="139" cy="25" r="4.5" className="fly-mascot__antenna-tip" />
        <ellipse cx="110" cy="105" rx="49" ry="48" fill={`url(#${bodyId})`} />
        <path d="M76 120 Q110 140 144 120" className="fly-mascot__abdomen" />
        <path d="M72 133 L54 151 M86 142 L75 165 M148 133 L166 151 M134 142 L145 165" className="fly-mascot__leg" />
        <circle cx="84" cy="88" r="24" fill={`url(#${eyeId})`} />
        <circle cx="136" cy="88" r="24" fill={`url(#${eyeId})`} />
        <circle cx="77" cy="80" r="7" className="fly-mascot__eye-glint" />
        <circle cx="129" cy="80" r="7" className="fly-mascot__eye-glint" />
        <circle cx="88" cy="93" r="4" className="fly-mascot__pupil" />
        <circle cx="132" cy="93" r="4" className="fly-mascot__pupil" />
        <Mouth mood={mood} />
      </g>
    </svg>
  )
}

export function FlyMascot({ prediction, confidence, isLoading, error }: FlyMascotProps) {
  let mood: Mood = 'idle'
  let message = 'Draw a digit and I will trace it through my circuit.'
  let detail = 'Ready to look.'

  if (error) {
    mood = 'error'
    message = 'My signal got tangled.'
    detail = error
  } else if (isLoading) {
    mood = 'thinking'
    message = 'Reading the signal…'
    detail = 'Sampling visual columns.'
  } else if (prediction !== null) {
    const percent = Math.round(confidence * 100)
    mood = confidence >= 0.5 ? 'happy' : 'unsure'
    message = confidence >= 0.5 ? `I see a ${prediction}!` : `Maybe a ${prediction}?`
    detail = `${percent}% classifier confidence`
  }

  return (
    <aside className={`fly-mascot fly-mascot--${mood}`} aria-live="polite">
      <FlyModel mood={mood} />
      <div className="fly-mascot__speech">
        <p>{message}</p>
        <span>{detail}</span>
      </div>
    </aside>
  )
}
