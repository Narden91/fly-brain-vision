export function ColumnSampling({ src }: { src: string }) {
  return (
    <section className="panel">
      <p className="panel__index">02 / Sampling</p>
      <h3>Visual-column sampling</h3>
      <img
        src={src}
        alt="Scatter plot of the sampled visual-column neuron activity at real optic-column positions"
        className="panel__image panel__image--white"
      />
    </section>
  )
}
