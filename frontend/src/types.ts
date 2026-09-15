export interface CellTypeActivity {
  cellType: string
  activity: number
}

export interface PredictResponse {
  prediction: number | null
  confidence: number
  probabilities?: Record<string, number>
  whatFlySeesPng?: string
  columnSamplingPng?: string
  topCellTypes?: CellTypeActivity[]
}

export interface AccuracySummary {
  inputOnly: number | null
  maleCns: number | null
  randomizedControl: number | null
}

export interface MetaResponse {
  dataset: string
  nNeurons: number
  nEdges: number
  nInputNeurons: number
  simulationSteps: number
  accuracy: AccuracySummary
  model?: {
    kind: string
    version: string
    temporalAccuracy: number | null
    device: string
  }
}
