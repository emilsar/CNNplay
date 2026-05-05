import { useRef } from 'react'
import { compatibleSkipSources } from '../lib/buildModel'

const ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'elu', 'linear']

export default function ArchitectureEditor({ architecture, shapes, onChange, disabled }) {
  const idCounter = useRef(Math.max(0, ...architecture.map((l) => l.id)) + 1)

  const update = (id, patch) =>
    onChange(architecture.map((l) => (l.id === id ? { ...l, ...patch } : l)))

  const remove = (id) =>
    onChange(
      architecture
        .filter((l) => l.id !== id)
        .map((l) => (l.skipFromId === id ? { ...l, skipFromId: null } : l))
    )

  const add = (type) => {
    const newLayer = newLayerForType(type, idCounter.current++)
    onChange([...architecture, newLayer])
  }

  const fmt = (s) => '[' + s.join('×') + ']'

  return (
    <>
      <div className="layer-card fixed">
        <div className="row">
          <span className="name">Input</span>
        </div>
        <div className="shape">{fmt(shapes[0])}</div>
      </div>

      {architecture.map((layer, i) => (
        <div className="layer-card" key={layer.id}>
          <div className="row">
            <span className="name">{layerLabel(layer)}</span>
            <button className="danger" onClick={() => remove(layer.id)} disabled={disabled}>×</button>
          </div>
          <LayerParams layer={layer} update={(patch) => update(layer.id, patch)} disabled={disabled} />
          {layer.type === 'conv2d' && (
            <SkipPicker
              architecture={architecture}
              shapes={shapes}
              targetIdx={i}
              value={layer.skipFromId}
              onChange={(skipFromId) => update(layer.id, { skipFromId })}
              disabled={disabled}
            />
          )}
          <div className="shape">→ {fmt(shapes[i + 1])}</div>
        </div>
      ))}

      <div className="layer-card fixed">
        <div className="row">
          <span className="name">Output head (auto)</span>
        </div>
      </div>

      <div className="add-layer">
        <button onClick={() => add('conv2d')} disabled={disabled}>+ Conv2D</button>
        <button onClick={() => add('conv2dTranspose')} disabled={disabled}>+ ConvT</button>
        <button onClick={() => add('maxPooling2d')} disabled={disabled}>+ MaxPool</button>
        <button onClick={() => add('avgPooling2d')} disabled={disabled}>+ AvgPool</button>
        <button onClick={() => add('flatten')} disabled={disabled}>+ Flatten</button>
        <button onClick={() => add('dense')} disabled={disabled}>+ Dense</button>
        <button onClick={() => add('dropout')} disabled={disabled}>+ Dropout</button>
      </div>
    </>
  )
}

function SkipPicker({ architecture, shapes, targetIdx, value, onChange, disabled }) {
  const sources = compatibleSkipSources(architecture, shapes, targetIdx)
  return (
    <div className="params skip-row">
      <label>skip from</label>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value === '' ? null : +e.target.value)}
        disabled={disabled || sources.length === 0}
      >
        <option value="">none</option>
        {sources.map((s) => (
          <option key={s.id} value={s.id}>
            layer {s.idx + 1} [{s.shape.join('×')}]
          </option>
        ))}
      </select>
    </div>
  )
}

function layerLabel(l) {
  switch (l.type) {
    case 'conv2d': return 'Conv2D'
    case 'conv2dTranspose': return 'ConvT'
    case 'maxPooling2d': return 'MaxPool'
    case 'avgPooling2d': return 'AvgPool'
    case 'flatten': return 'Flatten'
    case 'dense': return 'Dense'
    case 'dropout': return 'Dropout'
    default: return l.type
  }
}

function LayerParams({ layer, update, disabled }) {
  switch (layer.type) {
    case 'conv2d':
      return (
        <div className="params">
          <label>filters</label>
          <input type="number" min="1" max="64" value={layer.filters}
            onChange={(e) => update({ filters: +e.target.value })} disabled={disabled} />
          <label>kernel</label>
          <input type="number" min="1" max="7" value={layer.kernelSize}
            onChange={(e) => update({ kernelSize: +e.target.value })} disabled={disabled} />
          <label>padding</label>
          <select value={layer.padding || 'valid'}
            onChange={(e) => update({ padding: e.target.value })} disabled={disabled}>
            <option value="valid">valid</option>
            <option value="same">same</option>
          </select>
          <label>activation</label>
          <select value={layer.activation}
            onChange={(e) => update({ activation: e.target.value })} disabled={disabled}>
            {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
      )
    case 'conv2dTranspose':
      return (
        <div className="params">
          <label>filters</label>
          <input type="number" min="1" max="64" value={layer.filters}
            onChange={(e) => update({ filters: +e.target.value })} disabled={disabled} />
          <label>kernel</label>
          <input type="number" min="1" max="7" value={layer.kernelSize}
            onChange={(e) => update({ kernelSize: +e.target.value })} disabled={disabled} />
          <label>strides</label>
          <input type="number" min="1" max="4" value={layer.strides}
            onChange={(e) => update({ strides: +e.target.value })} disabled={disabled} />
          <label>activation</label>
          <select value={layer.activation}
            onChange={(e) => update({ activation: e.target.value })} disabled={disabled}>
            {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
      )
    case 'maxPooling2d':
    case 'avgPooling2d':
      return (
        <div className="params">
          <label>pool size</label>
          <input type="number" min="2" max="4" value={layer.poolSize}
            onChange={(e) => update({ poolSize: +e.target.value })} disabled={disabled} />
        </div>
      )
    case 'dense':
      return (
        <div className="params">
          <label>units</label>
          <input type="number" min="1" max="512" value={layer.units}
            onChange={(e) => update({ units: +e.target.value })} disabled={disabled} />
          <label>activation</label>
          <select value={layer.activation}
            onChange={(e) => update({ activation: e.target.value })} disabled={disabled}>
            {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
      )
    case 'dropout':
      return (
        <div className="params">
          <label>rate</label>
          <input type="number" min="0" max="0.9" step="0.05" value={layer.rate}
            onChange={(e) => update({ rate: +e.target.value })} disabled={disabled} />
        </div>
      )
    default:
      return null
  }
}

function newLayerForType(type, id) {
  switch (type) {
    case 'conv2d': return { id, type, filters: 16, kernelSize: 3, activation: 'relu', padding: 'valid', skipFromId: null }
    case 'conv2dTranspose': return { id, type, filters: 16, kernelSize: 3, strides: 2, activation: 'relu', padding: 'same' }
    case 'maxPooling2d': return { id, type, poolSize: 2 }
    case 'avgPooling2d': return { id, type, poolSize: 2 }
    case 'flatten': return { id, type }
    case 'dense': return { id, type, units: 32, activation: 'relu' }
    case 'dropout': return { id, type, rate: 0.25 }
  }
}
