import type { CellTypeActivity } from '../types'

export function TopCellTypes({ cellTypes }: { cellTypes: CellTypeActivity[] }) {
  if (cellTypes.length === 0) return null
  const maxMagnitude = Math.max(...cellTypes.map((entry) => Math.abs(entry.activity)), 1)

  return (
    <section className="panel panel--wide">
      <h3>Top activated downstream cell types</h3>
      <div className="hbar-chart">
        {cellTypes.map(({ cellType, activity }) => (
          <div className="hbar-chart__row" key={cellType}>
            <span className="hbar-chart__label">{cellType}</span>
            <div className="hbar-chart__track">
              <div
                className={`hbar-chart__bar hbar-chart__bar--${activity >= 0 ? 'positive' : 'negative'}`}
                style={{ width: `${(Math.abs(activity) / maxMagnitude) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
