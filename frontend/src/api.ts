import type { MetaResponse, PredictResponse } from './types'

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with status ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function getMeta(): Promise<MetaResponse> {
  return fetch('/api/meta').then((response) => parseJsonOrThrow<MetaResponse>(response))
}

export function predictDigit(imageDataUrl: string): Promise<PredictResponse> {
  return fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl }),
  }).then((response) => parseJsonOrThrow<PredictResponse>(response))
}
