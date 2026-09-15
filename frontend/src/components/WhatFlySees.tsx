export function WhatFlySees({ src }: { src: string }) {
  return (
    <section className="panel">
      <h3>1. What the fly sees</h3>
      <img src={src} alt="The 8 by 8 grayscale grid the classifier actually sees" className="panel__image" />
    </section>
  )
}
