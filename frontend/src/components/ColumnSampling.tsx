export function ColumnSampling({ src }: { src: string }) {
  return (
    <section className="panel">
      <h3>2. MaleCNS visual-column sampling</h3>
      <img
        src={src}
        alt="Scatter plot of the sampled visual-column neuron activity at real optic-column positions"
        className="panel__image panel__image--white"
      />
    </section>
  )
}
