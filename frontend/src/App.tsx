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
      <h1>MaleCNS Fly Brain Classifier</h1>
      <p className="app__caption">
        A handwritten digit is projected onto visual columns from the real MaleCNS connectome, propagated through
        measured neuron-to-neuron wiring, and classified from the resulting neural activity.
      </p>
      <p className="app__info">
        This is a toy computational model built on real anatomical connectivity. The connectome supplies the wiring;
        the dynamics and classifier are simplified software assumptions.
      </p>

      <section className="card">
        <h2>Draw a digit</h2>
        <div className="card__row">
          <DrawingCanvas size={CANVAS_SIZE} strokeWidth={STROKE_WIDTH} onStrokeEnd={handleStrokeEnd} onClear={reset} />
          <FlyMascot prediction={data?.prediction ?? null} confidence={data?.confidence ?? 0} isLoading={isLoading} error={error} />
        </div>
      </section>

      {hasFullPrediction && data && (
        <>
          <hr />
          <div className="panels">
            <WhatFlySees src={data.whatFlySeesPng!} />
            <ColumnSampling src={data.columnSamplingPng!} />
            {meta && <ConnectomeStats meta={meta} />}
            <PredictionPanel prediction={data.prediction!} confidence={data.confidence} probabilities={data.probabilities!} />
          </div>
          <hr />
          <TopCellTypes cellTypes={data.topCellTypes ?? []} />
        </>
      )}

      {meta && <AccuracyFooter accuracy={meta.accuracy} />}
    </main>
  )
}

export default App
