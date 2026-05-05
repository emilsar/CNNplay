import { useEffect, useRef } from 'react'

const MASK_PALETTE = [
  [15, 17, 21, 0],
  [110, 168, 255, 200],
  [74, 222, 128, 200],
  [251, 191, 36, 200],
  [248, 113, 113, 200],
  [167, 139, 250, 200],
]

export default function Predictions({ cells, mode = 'classification' }) {
  if (cells.length === 0) {
    return <div className="empty">Train a model to see predictions on test samples.</div>
  }
  return (
    <div className={`predictions ${mode}`}>
      {cells.map((c, i) => (mode === 'segmentation' ? <SegCell key={i} cell={c} /> : <ClsCell key={i} cell={c} />))}
    </div>
  )
}

function ClsCell({ cell }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    drawGrayscale(canvas, cell.pixels, cell.h, cell.w)
  }, [cell])

  const correct = cell.predicted === cell.actual
  return (
    <div className={`pred-cell ${correct ? 'correct' : 'wrong'}`}>
      <canvas ref={canvasRef} width={cell.w} height={cell.h} />
      <div className="label">
        <span>pred {cell.predicted}</span>
        <span className="muted">true {cell.actual}</span>
      </div>
    </div>
  )
}

function SegCell({ cell }) {
  const inputRef = useRef(null)
  const predRef = useRef(null)
  const truthRef = useRef(null)

  useEffect(() => {
    if (inputRef.current) {
      if (cell.channels === 1) drawGrayscale(inputRef.current, cell.pixels, cell.h, cell.w)
      else drawRGB(inputRef.current, cell.pixels, cell.h, cell.w)
    }
    if (predRef.current) drawMask(predRef.current, cell.predictedMask, cell.h, cell.w)
    if (truthRef.current) drawMask(truthRef.current, cell.trueMask, cell.h, cell.w)
  }, [cell])

  return (
    <div className="seg-cell">
      <div className="seg-row">
        <div className="seg-tile">
          <canvas ref={inputRef} width={cell.w} height={cell.h} />
          <span className="seg-label">input</span>
        </div>
        <div className="seg-tile">
          <canvas ref={predRef} width={cell.w} height={cell.h} />
          <span className="seg-label">pred</span>
        </div>
        <div className="seg-tile">
          <canvas ref={truthRef} width={cell.w} height={cell.h} />
          <span className="seg-label">truth</span>
        </div>
      </div>
    </div>
  )
}

function drawGrayscale(canvas, pixels, h, w) {
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(w, h)
  for (let i = 0; i < h * w; i++) {
    const v = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)))
    img.data[i * 4] = v
    img.data[i * 4 + 1] = v
    img.data[i * 4 + 2] = v
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

function drawRGB(canvas, pixels, h, w) {
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(w, h)
  for (let i = 0; i < h * w; i++) {
    img.data[i * 4] = Math.round(pixels[i * 3] * 255)
    img.data[i * 4 + 1] = Math.round(pixels[i * 3 + 1] * 255)
    img.data[i * 4 + 2] = Math.round(pixels[i * 3 + 2] * 255)
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

function drawMask(canvas, mask, h, w) {
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(w, h)
  for (let i = 0; i < h * w; i++) {
    const cls = mask[i] | 0
    const c = MASK_PALETTE[cls % MASK_PALETTE.length]
    img.data[i * 4] = c[0]
    img.data[i * 4 + 1] = c[1]
    img.data[i * 4 + 2] = c[2]
    img.data[i * 4 + 3] = c[3]
  }
  ctx.putImageData(img, 0, 0)
}
