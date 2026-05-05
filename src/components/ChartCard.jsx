import { useEffect, useState } from 'react'
import TrainingChart from './TrainingChart'

export default function ChartCard({ title, history, keys, height = 200 }) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  return (
    <>
      <div className="chart-card">
        <div className="chart-card-head">
          <h3>{title}</h3>
          <button className="icon-btn" onClick={() => setOpen(true)} title="Maximize">⤢</button>
        </div>
        <TrainingChart history={history} keys={keys} height={height} />
      </div>

      {open && (
        <div className="modal-overlay" onClick={() => setOpen(false)}>
          <div className="modal-window" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>{title}</h3>
              <button className="icon-btn" onClick={() => setOpen(false)} title="Close">×</button>
            </div>
            <div className="modal-body">
              <TrainingChart history={history} keys={keys} height={520} />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
