export const defaultArchitecture = [
  { id: 1, type: 'conv2d', filters: 8, kernelSize: 3, activation: 'relu' },
  { id: 2, type: 'maxPooling2d', poolSize: 2 },
  { id: 3, type: 'conv2d', filters: 16, kernelSize: 3, activation: 'relu' },
  { id: 4, type: 'maxPooling2d', poolSize: 2 },
  { id: 5, type: 'flatten' },
  { id: 6, type: 'dense', units: 32, activation: 'relu' },
]

export const defaultHyperparams = {
  optimizer: 'adam',
  learningRate: 0.001,
  batchSize: 64,
  epochs: 5,
}
