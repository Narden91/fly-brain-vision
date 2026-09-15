import type { MetaResponse } from '../types'

export function ConnectomeStats({ meta }: { meta: MetaResponse }) {
  return (
    <section className="panel">
      <p className="panel__index">03 / Circuit</p>
      <h3>Connectome activity</h3>
      <dl className="stats-list">
        <div><dt>Dataset</dt><dd>{meta.dataset}</dd></div>
        <div><dt>Neurons</dt><dd>{meta.nNeurons.toLocaleString()}</dd></div>
        <div><dt>Connections</dt><dd>{meta.nEdges.toLocaleString()}</dd></div>
        <div><dt>Image inputs</dt><dd>{meta.nInputNeurons.toLocaleString()}</dd></div>
        <div><dt>Simulation steps</dt><dd>{meta.simulationSteps}</dd></div>
      </dl>
    </section>
  )
}
