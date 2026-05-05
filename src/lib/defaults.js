export const defaultArchitecture = [
  { id: 1, type: 'conv2d', filters: 8, kernelSize: 3, activation: 'relu', skipFromId: null },
  { id: 2, type: 'maxPooling2d', poolSize: 2 },
  { id: 3, type: 'conv2d', filters: 16, kernelSize: 3, activation: 'relu', skipFromId: null },
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

export const datasets = {
  mnist: {
    id: 'mnist',
    label: 'MNIST',
    description: '28×28 grayscale digits, 10 classes',
    mode: 'classification',
    inputShape: [28, 28, 1],
    numClasses: 10,
    available: true,
  },
  urothelial: {
    id: 'urothelial',
    label: 'Urothelial cells',
    description: 'Cytology image segmentation (coming soon)',
    mode: 'segmentation',
    inputShape: [128, 128, 3],
    numClasses: 3,
    available: false,
  },
}
