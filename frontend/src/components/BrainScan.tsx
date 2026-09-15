export function BrainScan() {
  const neuroglancerScene = 'https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json'

  return (
    <section className="brain-scan" aria-labelledby="brain-scan-heading">
      <div className="brain-scan__copy">
        <p className="section-label">Official 3D anatomy</p>
        <h2 id="brain-scan-heading">MaleCNS v1.0 visual circuit</h2>
        <p>
          Explore the official MaleCNS v1.0 Neuroglancer scene. Drag to rotate, scroll to zoom, and use its layer
          controls to inspect EM imagery, segmentation, synapses, and neuropil compartments.
        </p>
        <a className="brain-scan__link" href={neuroglancerScene} target="_blank" rel="noreferrer">
          Open full-screen viewer
        </a>
      </div>
      <figure className="brain-scan__figure">
        <iframe
          className="brain-scan__viewer"
          src={neuroglancerScene}
          title="Official interactive MaleCNS v1.0 Neuroglancer scene"
          loading="lazy"
        />
        <figcaption>Official Janelia MaleCNS v1.0 scene. It requires an internet connection.</figcaption>
      </figure>
    </section>
  )
}
