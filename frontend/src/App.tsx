import { useCallback } from 'react'
import './App.css'
import { AccuracyFooter } from './components/AccuracyFooter'
import { ColumnSampling } from './components/ColumnSampling'
import { ConnectomeStats } from './components/ConnectomeStats'
import { DrawingCanvas } from './components/DrawingCanvas'
import { FlyMascot } from './components/FlyMascot'
import { PredictionPanel } from './components/PredictionPanel'
import { TopCellTypes } from './components/TopCellTypes'
import { WhatFlySees } from './components/WhatFlySees'
import { useMeta } from './hooks/useMeta'
import { usePrediction } from './hooks/usePrediction'

const CANVAS_SIZE = 280
// Training digits are thick, blocky 8x8 strokes; a thin pen stroke downsamples to a
// faint smear. Raise/lower if handwriting on your display draws thinner/thicker.
const STROKE_WIDTH = 18

function App() {
  const { meta } = useMeta()
  const { data, isLoading, error, predict, reset } = usePrediction()

  const handleStrokeEnd = useCallback(
    (imageDataUrl: string) => {
      void predict(imageDataUrl)
    },
    [predict],
  )

  const hasFullPrediction = data?.prediction != null && data.probabilities != null

  return (
    <main className="app">
      <header className="app__header">
        <p className="app__eyebrow">Interactive connectomics experiment</p>
        <div className="app__heading">
          <div>
            <h1>How a fly brain reads a digit</h1>
            <p className="app__caption">
              A drawn numeral is sampled by optic columns in the MaleCNS connectome, propagated through its wiring,
              and read by a linear classifier.
            </p>
          </div>
          <p className="app__stamp">MaleCNS v1.0<br />Visual circuit</p>
        </div>
        <p className="app__note">
          The wiring is anatomical data. The neural dynamics and final digit readout are simplified computational
          assumptions.
        </p>
      </header>

      <section className="experiment" aria-labelledby="draw-heading">
        <div className="experiment__header">
          <div>
            <p className="section-label">01 / Input</p>
            <h2 id="draw-heading">Draw one digit</h2>
          </div>
          <p>Use a single, bold stroke. The model crops and reduces it to an 8 × 8 image.</p>
        </div>
        <div className="experiment__body">
          <DrawingCanvas size={CANVAS_SIZE} strokeWidth={STROKE_WIDTH} onStrokeEnd={handleStrokeEnd} onClear={reset} />
          <FlyMascot
            prediction={data?.prediction ?? null}
            confidence={data?.confidence ?? 0}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </section>

      {hasFullPrediction && data && (
        <section className="results" aria-labelledby="results-heading">
          <div className="results__header">
            <div>
              <p className="section-label">02 / Result</p>
              <h2 id="results-heading">Signal through the visual circuit</h2>
            </div>
            <p>Each panel shows one stage between your drawing and the final classification.</p>
          </div>
          <div className="panels">
            <WhatFlySees src={data.whatFlySeesPng!} />
            <ColumnSampling src={data.columnSamplingPng!} />
            {meta && <ConnectomeStats meta={meta} />}
            <PredictionPanel prediction={data.prediction!} confidence={data.confidence} probabilities={data.probabilities!} />
          </div>
          <TopCellTypes cellTypes={data.topCellTypes ?? []} />
        </section>
      )}

      {meta && <AccuracyFooter accuracy={meta.accuracy} />}
    </main>
  )
}

export default App
