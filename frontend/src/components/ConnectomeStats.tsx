import type { MetaResponse } from '../types'

export function ConnectomeStats({ meta }: { meta: MetaResponse }) {
  return (
    <section className="panel">
      <h3>3. Connectome activity</h3>
      <p>{meta.dataset}</p>
      <p>{meta.nNeurons.toLocaleString()} biological neurons</p>
      <p>{meta.nEdges.toLocaleString()} biological connections</p>
      <p>{meta.nInputNeurons.toLocaleString()} image-input neurons</p>
      <p>{meta.simulationSteps} simulation steps</p>
    </section>
  )
}
