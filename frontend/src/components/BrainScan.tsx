export function BrainScan() {
  return (
    <section className="brain-scan" aria-labelledby="brain-scan-heading">
      <div className="brain-scan__copy">
        <p className="section-label">Reference anatomy</p>
        <h2 id="brain-scan-heading">MaleCNS brain scan</h2>
        <p>
          A 3D visual representation of the scanned fly brain, showing dense fluorescent neural tracts across the
          central brain and optic lobes.
        </p>
        <p className="brain-scan__note">Visual reference based on the supplied connectome scan.</p>
      </div>
      <figure className="brain-scan__figure">
        <img
          src="/malecns-brain-scan.png"
          alt="Frontal 3D representation of a fluorescent Drosophila brain scan with colourful neural tracts"
          loading="lazy"
        />
      </figure>
    </section>
  )
}
