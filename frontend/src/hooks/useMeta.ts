import { useEffect, useState } from 'react'
import { getMeta } from '../api'
import type { MetaResponse } from '../types'

export function useMeta() {
  const [meta, setMeta] = useState<MetaResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getMeta()
      .then(setMeta)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load circuit info.')
      })
  }, [])

  return { meta, error }
}
