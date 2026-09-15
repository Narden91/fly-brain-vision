export function WhatFlySees({ src }: { src: string }) {
  return (
    <section className="panel">
      <p className="panel__index">01 / Encoding</p>
      <h3>What the fly sees</h3>
      <img src={src} alt="The 8 by 8 grayscale grid the classifier actually sees" className="panel__image" />
    </section>
  )
}
