import { useMemo } from 'react'

const COLORS = {
  input: '#8a93a6',
  conv2d: '#6ea8ff',
  conv2dTranspose: '#a78bfa',
  maxPooling2d: '#4ade80',
  avgPooling2d: '#22d3ee',
  flatten: '#94a3b8',
  dense: '#fbbf24',
  dropout: '#f87171',
  output: '#8a93a6',
}

const LABELS = {
  input: 'Input',
  conv2d: 'Conv2D',
  conv2dTranspose: 'ConvT',
  maxPooling2d: 'MaxPool',
  avgPooling2d: 'AvgPool',
  flatten: 'Flatten',
  dense: 'Dense',
  dropout: 'Dropout',
  output: 'Output',
}

const SLOT_W = 140
const MAX_TILE = 84
const MIN_TILE = 14
const DEPTH_OFFSET = 3.5
const VISIBLE_DEPTH = 6
const VISIBLE_NODES = 7
const NODE_SPACING = 16
const NODE_R = 4.5
const TYPE_LABEL_Y = 16
const SHAPE_LABEL_Y = 32
const BASE_Y = 175
const DETAIL_LINE_H = 13
const SVG_H = 320

export default function ArchitectureDiagram({ architecture, shapes, mode, numClasses }) {
  const blocks = useMemo(() => {
    const list = [
      { kind: 'input', shape: shapes[0], detail: [] },
      ...architecture.map((layer, i) => ({
        kind: layer.type,
        shape: shapes[i + 1],
        detail: layerDetail(layer),
        skipFromId: layer.skipFromId,
        id: layer.id,
      })),
      {
        kind: 'output',
        shape: outputShape(mode, shapes[shapes.length - 1], numClasses),
        detail: mode === 'classification'
          ? [`softmax(${numClasses})`]
          : [`conv1×1(${numClasses})`, 'softmax'],
      },
    ]
    return list
  }, [architecture, shapes, mode, numClasses])

  const maxSpatial = useMemo(() => {
    let m = 1
    for (const b of blocks) {
      if (b.shape && b.shape.length === 3) m = Math.max(m, b.shape[0], b.shape[1])
    }
    return m
  }, [blocks])

  const glyphs = useMemo(() => blocks.map((b, i) => {
    const cx = SLOT_W * (i + 0.5)
    if (b.shape && b.shape.length === 3) {
      const ratio = Math.max(b.shape[0], b.shape[1]) / maxSpatial
      const size = Math.max(MIN_TILE, MAX_TILE * Math.sqrt(ratio))
      const ch = b.shape[2]
      const depth = Math.min(ch, VISIBLE_DEPTH)
      return {
        type: 'spatial',
        cx,
        size,
        depth,
        ch,
        leftX: cx - size / 2,
        rightX: cx + size / 2,
        topY: BASE_Y - size / 2 - (depth - 1) * DEPTH_OFFSET,
      }
    }
    if (b.shape && b.shape.length === 1) {
      const u = b.shape[0]
      const visible = Math.min(u, VISIBLE_NODES)
      const totalH = (visible - 1) * NODE_SPACING
      return {
        type: 'nodes',
        cx,
        units: u,
        visible,
        topY: BASE_Y - totalH / 2,
        bottomY: BASE_Y + totalH / 2,
        leftX: cx - NODE_R,
        rightX: cx + NODE_R,
      }
    }
    return { type: 'empty', cx, leftX: cx, rightX: cx, topY: BASE_Y }
  }), [blocks, maxSpatial])

  const skips = []
  architecture.forEach((layer, i) => {
    if (layer.type !== 'conv2d' || layer.skipFromId == null) return
    const srcIdx = architecture.findIndex((l) => l.id === layer.skipFromId)
    if (srcIdx >= 0) skips.push({ from: srcIdx + 1, to: i + 1 })
  })

  const totalW = SLOT_W * blocks.length

  return (
    <div className="arch-diagram-svg">
      <svg width={totalW} height={SVG_H} viewBox={`0 0 ${totalW} ${SVG_H}`}>
        <defs>
          <marker id="fwd-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="#5a6478" />
          </marker>
          <marker id="skip-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="#a78bfa" />
          </marker>
        </defs>

        {skips.map((s, i) => {
          const a = glyphs[s.from]
          const b = glyphs[s.to]
          const y1 = a.topY - 6
          const y2 = b.topY - 6
          const cy = Math.min(y1, y2) - 80
          return (
            <path
              key={`skip-${i}`}
              d={`M ${a.cx} ${y1} Q ${(a.cx + b.cx) / 2} ${cy}, ${b.cx} ${y2}`}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="2"
              strokeDasharray="5 4"
              markerEnd="url(#skip-arrow)"
            />
          )
        })}

        {glyphs.slice(0, -1).map((a, i) => {
          const b = glyphs[i + 1]
          if (a.type === 'nodes' && b.type === 'nodes') {
            const lines = []
            for (let p = 0; p < a.visible; p++) {
              for (let q = 0; q < b.visible; q++) {
                lines.push(
                  <line
                    key={`bp-${p}-${q}`}
                    x1={a.cx + NODE_R}
                    y1={a.topY + p * NODE_SPACING}
                    x2={b.cx - NODE_R}
                    y2={b.topY + q * NODE_SPACING}
                    stroke="#5a6478"
                    strokeWidth="0.5"
                    strokeOpacity="0.55"
                  />
                )
              }
            }
            return <g key={`fwd-${i}`}>{lines}</g>
          }
          return (
            <line
              key={`fwd-${i}`}
              x1={a.rightX + 4}
              y1={BASE_Y}
              x2={b.leftX - 4}
              y2={BASE_Y}
              stroke="#5a6478"
              strokeWidth="1.5"
              markerEnd="url(#fwd-arrow)"
            />
          )
        })}

        {glyphs.map((g, i) => {
          const b = blocks[i]
          if (g.type === 'spatial') return <SpatialGlyph key={i} g={g} kind={b.kind} />
          if (g.type === 'nodes') return <NodesGlyph key={i} g={g} kind={b.kind} />
          return null
        })}

        {blocks.map((b, i) => {
          const cx = SLOT_W * (i + 0.5)
          const color = COLORS[b.kind] || '#8a93a6'
          return (
            <g key={`lbl-${i}`}>
              <text
                x={cx}
                y={TYPE_LABEL_Y}
                textAnchor="middle"
                fill={color}
                fontSize="12"
                fontWeight="600"
              >
                {LABELS[b.kind] || b.kind}
              </text>
              <text
                x={cx}
                y={SHAPE_LABEL_Y}
                textAnchor="middle"
                fill="#8a93a6"
                fontSize="10"
                fontFamily="ui-monospace, monospace"
              >
                {fmtShape(b.shape)}
              </text>
              {b.detail && b.detail.map((line, j) => (
                <text
                  key={j}
                  x={cx}
                  y={SVG_H - 14 - (b.detail.length - 1 - j) * DETAIL_LINE_H}
                  textAnchor="middle"
                  fill="#8a93a6"
                  fontSize="10"
                  fontFamily="ui-monospace, monospace"
                >
                  {line}
                </text>
              ))}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function SpatialGlyph({ g, kind }) {
  const { cx, size, depth, ch } = g
  const color = COLORS[kind] || '#8a93a6'
  const tiles = []
  for (let i = depth - 1; i >= 0; i--) {
    const ox = i * DEPTH_OFFSET
    const oy = -i * DEPTH_OFFSET
    tiles.push(
      <rect
        key={i}
        x={cx - size / 2 + ox}
        y={BASE_Y - size / 2 + oy}
        width={size}
        height={size}
        fill={color}
        fillOpacity={0.14 + i * 0.05}
        stroke={color}
        strokeWidth="1"
      />
    )
  }
  return (
    <g>
      {tiles}
      {ch > 1 && (
        <text
          x={cx + size / 2 + (depth - 1) * DEPTH_OFFSET + 5}
          y={BASE_Y - size / 2 - (depth - 1) * DEPTH_OFFSET + 8}
          fill={color}
          fontSize="10"
          fontFamily="ui-monospace, monospace"
        >
          ×{ch}
        </text>
      )}
    </g>
  )
}

function NodesGlyph({ g, kind }) {
  const { cx, units, visible, topY } = g
  const color = COLORS[kind] || '#8a93a6'
  const dots = []
  for (let i = 0; i < visible; i++) {
    dots.push(
      <circle
        key={i}
        cx={cx}
        cy={topY + i * NODE_SPACING}
        r={NODE_R}
        fill={color}
        fillOpacity="0.4"
        stroke={color}
        strokeWidth="1.5"
      />
    )
  }
  return (
    <g>
      {dots}
      {units > visible && (
        <text
          x={cx}
          y={topY + visible * NODE_SPACING + 6}
          textAnchor="middle"
          fill="#8a93a6"
          fontSize="9"
          fontFamily="ui-monospace, monospace"
        >
          ⋮ {units}
        </text>
      )}
    </g>
  )
}

function fmtShape(s) {
  if (!s) return ''
  return s.join('×')
}

function layerDetail(l) {
  switch (l.type) {
    case 'conv2d':
      return [`${l.filters} @ ${l.kernelSize}×${l.kernelSize}`, `${l.padding || 'valid'} · ${l.activation}`]
    case 'conv2dTranspose':
      return [`${l.filters} @ ${l.kernelSize}×${l.kernelSize}`, `stride ${l.strides || 2} · ${l.activation}`]
    case 'maxPooling2d':
    case 'avgPooling2d':
      return [`pool ${l.poolSize}×${l.poolSize}`]
    case 'dense':
      return [`${l.units} units`, l.activation]
    case 'dropout':
      return [`rate ${l.rate}`]
    default:
      return []
  }
}

function outputShape(mode, lastShape, numClasses) {
  if (mode === 'classification') return [numClasses]
  if (lastShape && lastShape.length === 3) return [lastShape[0], lastShape[1], numClasses]
  return [numClasses]
}
