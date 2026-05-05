import * as tf from '@tensorflow/tfjs'

export function buildModel(architecture, inputShape, numClasses, mode = 'classification') {
  const input = tf.input({ shape: inputShape })
  const tensors = [input]

  for (let i = 0; i < architecture.length; i++) {
    const layer = architecture[i]
    const prev = tensors[tensors.length - 1]
    let out = makeLayer(layer).apply(prev)

    if (layer.type === 'conv2d' && layer.skipFromId != null) {
      const srcIdx = architecture.findIndex((l) => l.id === layer.skipFromId)
      if (srcIdx >= 0 && srcIdx < i) {
        const source = tensors[srcIdx + 1]
        const outShape = out.shape
        const srcShape = source.shape
        if (outShape[1] !== srcShape[1] || outShape[2] !== srcShape[2]) {
          throw new Error(
            `Skip from layer ${srcIdx + 1} to ${i + 1}: spatial dims ${srcShape[1]}×${srcShape[2]} ≠ ${outShape[1]}×${outShape[2]}`
          )
        }
        const projected =
          srcShape[3] === outShape[3]
            ? source
            : tf.layers
                .conv2d({ filters: outShape[3], kernelSize: 1, padding: 'same' })
                .apply(source)
        out = tf.layers.add().apply([out, projected])
      }
    }
    tensors.push(out)
  }

  let head = tensors[tensors.length - 1]
  if (mode === 'classification') {
    head = tf.layers.dense({ units: numClasses, activation: 'softmax' }).apply(head)
  } else {
    head = tf.layers
      .conv2d({ filters: numClasses, kernelSize: 1, activation: 'softmax', padding: 'same' })
      .apply(head)
  }

  return tf.model({ inputs: input, outputs: head })
}

function makeLayer(layer) {
  switch (layer.type) {
    case 'conv2d':
      return tf.layers.conv2d({
        filters: layer.filters,
        kernelSize: layer.kernelSize,
        activation: layer.activation,
        padding: layer.padding || 'valid',
      })
    case 'conv2dTranspose':
      return tf.layers.conv2dTranspose({
        filters: layer.filters,
        kernelSize: layer.kernelSize,
        strides: layer.strides || 2,
        activation: layer.activation,
        padding: layer.padding || 'same',
      })
    case 'maxPooling2d':
      return tf.layers.maxPooling2d({ poolSize: layer.poolSize })
    case 'avgPooling2d':
      return tf.layers.averagePooling2d({ poolSize: layer.poolSize })
    case 'flatten':
      return tf.layers.flatten()
    case 'dense':
      return tf.layers.dense({ units: layer.units, activation: layer.activation })
    case 'dropout':
      return tf.layers.dropout({ rate: layer.rate })
    default:
      throw new Error('Unknown layer type: ' + layer.type)
  }
}

export function computeShapes(architecture, inputShape) {
  const shapes = [inputShape.slice()]
  let s = inputShape.slice()
  for (const layer of architecture) {
    s = nextShape(layer, s)
    shapes.push(s.slice())
  }
  return shapes
}

function nextShape(layer, s) {
  switch (layer.type) {
    case 'conv2d': {
      const [h, w] = s
      const k = layer.kernelSize
      const pad = layer.padding === 'same' ? 0 : k - 1
      return [h - pad, w - pad, layer.filters]
    }
    case 'conv2dTranspose': {
      const [h, w] = s
      const stride = layer.strides || 2
      return [h * stride, w * stride, layer.filters]
    }
    case 'maxPooling2d':
    case 'avgPooling2d': {
      const [h, w, c] = s
      const p = layer.poolSize
      return [Math.floor(h / p), Math.floor(w / p), c]
    }
    case 'flatten':
      return [s.reduce((a, b) => a * b, 1)]
    case 'dense':
      return [layer.units]
    case 'dropout':
      return s.slice()
    default:
      return s.slice()
  }
}

export function compatibleSkipSources(architecture, shapes, targetIdx) {
  const targetShape = shapes[targetIdx + 1]
  const sources = []
  for (let i = 0; i < targetIdx; i++) {
    const s = shapes[i + 1]
    if (s.length === 3 && targetShape && targetShape.length === 3 &&
        s[0] === targetShape[0] && s[1] === targetShape[1]) {
      sources.push({ idx: i, id: architecture[i].id, shape: s })
    }
  }
  return sources
}
