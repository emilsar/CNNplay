import * as tf from '@tensorflow/tfjs'

export function buildModel(architecture, inputShape, numClasses) {
  const model = tf.sequential()
  let first = true

  for (const layer of architecture) {
    const config = layerConfig(layer, first ? inputShape : undefined)
    model.add(config)
    first = false
  }

  model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }))
  return model
}

function layerConfig(layer, inputShape) {
  const opts = inputShape ? { inputShape } : {}
  switch (layer.type) {
    case 'conv2d':
      return tf.layers.conv2d({
        ...opts,
        filters: layer.filters,
        kernelSize: layer.kernelSize,
        activation: layer.activation,
        padding: 'valid',
      })
    case 'maxPooling2d':
      return tf.layers.maxPooling2d({ ...opts, poolSize: layer.poolSize })
    case 'avgPooling2d':
      return tf.layers.averagePooling2d({ ...opts, poolSize: layer.poolSize })
    case 'flatten':
      return tf.layers.flatten(opts)
    case 'dense':
      return tf.layers.dense({ ...opts, units: layer.units, activation: layer.activation })
    case 'dropout':
      return tf.layers.dropout({ ...opts, rate: layer.rate })
    default:
      throw new Error('Unknown layer type: ' + layer.type)
  }
}

// Statically compute the output shape of each layer for display.
// Returns array of shapes (one per layer) plus the input shape at index 0.
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
      return [h - k + 1, w - k + 1, layer.filters]
    }
    case 'maxPooling2d':
    case 'avgPooling2d': {
      const [h, w, c] = s
      const p = layer.poolSize
      return [Math.floor(h / p), Math.floor(w / p), c]
    }
    case 'flatten': {
      const total = s.reduce((a, b) => a * b, 1)
      return [total]
    }
    case 'dense':
      return [layer.units]
    case 'dropout':
      return s.slice()
    default:
      return s.slice()
  }
}
