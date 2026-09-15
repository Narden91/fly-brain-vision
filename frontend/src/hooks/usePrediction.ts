import { useCallback, useRef, useState } from 'react'
import { predictDigit } from '../api'
import type { PredictResponse } from '../types'

interface PredictionState {
  data: PredictResponse | null
  isLoading: boolean
  error: string | null
}

const INITIAL_STATE: PredictionState = { data: null, isLoading: false, error: null }

/** Predicts a digit from a canvas data URL. Ignores a stale response if a newer
 * stroke was drawn (and a newer request started) before this one came back. */
export function usePrediction() {
  const [state, setState] = useState<PredictionState>(INITIAL_STATE)
  const latestRequestId = useRef(0)

  const predict = useCallback(async (imageDataUrl: string) => {
    const requestId = ++latestRequestId.current
    setState((prev) => ({ ...prev, isLoading: true, error: null }))
    try {
      const data = await predictDigit(imageDataUrl)
      if (requestId === latestRequestId.current) setState({ data, isLoading: false, error: null })
    } catch (err) {
      if (requestId !== latestRequestId.current) return
      const message = err instanceof Error ? err.message : 'Prediction failed.'
      setState((prev) => ({ ...prev, isLoading: false, error: message }))
    }
  }, [])

  const reset = useCallback(() => {
    latestRequestId.current += 1 // invalidate any in-flight request
    setState(INITIAL_STATE)
  }, [])

  return { ...state, predict, reset }
}
