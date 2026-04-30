import { useEffect, useRef } from 'react'

export default function Predictions({ cells }) {
  if (cells.length === 0) {
    return <div style={{ color: '#8a93a6', fontSize: 12 }}>Train a model to see predictions on test samples.</div>
  }
  return (
    <div className="predictions">
      {cells.map((c, i) => (
        <Cell key={i} cell={c} />
      ))}
    </div>
  )
}

function Cell({ cell }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const img = ctx.createImageData(28, 28)
    for (let i = 0; i < 784; i++) {
      const v = Math.max(0, Math.min(255, Math.round(cell.pixels[i] * 255)))
      img.data[i * 4] = v
      img.data[i * 4 + 1] = v
      img.data[i * 4 + 2] = v
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [cell])

  const correct = cell.predicted === cell.actual
  return (
    <div className={`pred-cell ${correct ? 'correct' : 'wrong'}`}>
      <canvas ref={canvasRef} width={28} height={28} />
      <div className="label">{cell.predicted}{correct ? '' : ` (≠${cell.actual})`}</div>
    </div>
  )
}
