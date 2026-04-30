import { useState, useEffect, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import ArchitectureEditor from './components/ArchitectureEditor'
import Hyperparameters from './components/Hyperparameters'
import TrainingChart from './components/TrainingChart'
import Predictions from './components/Predictions'
import { defaultArchitecture, defaultHyperparams } from './lib/defaults'
import { buildModel, computeShapes } from './lib/buildModel'
import { MnistData } from './data/mnist'

function App() {
  const [architecture, setArchitecture] = useState(defaultArchitecture)
  const [hyperparams, setHyperparams] = useState(defaultHyperparams)
  const [data, setData] = useState(null)
  const [dataStatus, setDataStatus] = useState('idle')
  const [training, setTraining] = useState(false)
  const [history, setHistory] = useState([])
  const [predictions, setPredictions] = useState([])
  const [status, setStatus] = useState('Ready. Load data to begin.')
  const [backend, setBackend] = useState('')
  const modelRef = useRef(null)
  const stopRef = useRef(false)

  useEffect(() => {
    tf.ready().then(() => setBackend(tf.getBackend()))
  }, [])

  const shapes = computeShapes(architecture, [28, 28, 1])

  async function loadData() {
    setDataStatus('loading')
    setStatus('Loading MNIST sprite (~10 MB)...')
    try {
      const d = new MnistData()
      await d.load()
      setData(d)
      setDataStatus('ready')
      setStatus('Data loaded: 55,000 train / 10,000 test samples.')
    } catch (e) {
      setDataStatus('error')
      setStatus('Error loading data: ' + e.message)
    }
  }

  async function train() {
    if (!data) return
    setTraining(true)
    stopRef.current = false
    setHistory([])

    if (modelRef.current) {
      modelRef.current.dispose()
      modelRef.current = null
    }

    let model
    try {
      model = buildModel(architecture, [28, 28, 1], 10)
    } catch (e) {
      setStatus('Model build error: ' + e.message)
      setTraining(false)
      return
    }

    const optimizer = hyperparams.optimizer === 'adam'
      ? tf.train.adam(hyperparams.learningRate)
      : hyperparams.optimizer === 'sgd'
      ? tf.train.sgd(hyperparams.learningRate)
      : tf.train.rmsprop(hyperparams.learningRate)

    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    })
    modelRef.current = model

    const TRAIN_SIZE = 5500
    const TEST_SIZE = 1000
    const trainData = data.nextTrainBatch(TRAIN_SIZE)
    const testData = data.nextTestBatch(TEST_SIZE)
    const trainXs = trainData.xs.reshape([TRAIN_SIZE, 28, 28, 1])
    const trainYs = trainData.labels
    const testXs = testData.xs.reshape([TEST_SIZE, 28, 28, 1])
    const testYs = testData.labels

    setStatus(`Training: ${TRAIN_SIZE} samples, batch ${hyperparams.batchSize}, ${hyperparams.epochs} epochs.`)

    try {
      await model.fit(trainXs, trainYs, {
        batchSize: hyperparams.batchSize,
        epochs: hyperparams.epochs,
        validationData: [testXs, testYs],
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            setHistory((h) => [
              ...h,
              {
                epoch: epoch + 1,
                loss: logs.loss,
                acc: logs.acc,
                valLoss: logs.val_loss,
                valAcc: logs.val_acc,
              },
            ])
            setStatus(
              `Epoch ${epoch + 1}/${hyperparams.epochs} — loss ${logs.loss.toFixed(4)}, acc ${(logs.acc * 100).toFixed(1)}%, val_acc ${(logs.val_acc * 100).toFixed(1)}%`
            )
            await tf.nextFrame()
            if (stopRef.current) {
              model.stopTraining = true
            }
          },
        },
      })

      const samples = data.nextTestBatch(20)
      const sampleXs = samples.xs.reshape([20, 28, 28, 1])
      const preds = model.predict(sampleXs)
      const predIdx = await preds.argMax(1).data()
      const trueIdx = await samples.labels.argMax(1).data()
      const imageData = await sampleXs.data()
      const cells = []
      for (let i = 0; i < 20; i++) {
        cells.push({
          pixels: imageData.slice(i * 784, (i + 1) * 784),
          predicted: predIdx[i],
          actual: trueIdx[i],
        })
      }
      setPredictions(cells)
      preds.dispose()
      sampleXs.dispose()
      samples.xs.dispose()
      samples.labels.dispose()

      setStatus((s) => s + '\nTraining complete.')
    } catch (e) {
      setStatus('Training error: ' + e.message)
    } finally {
      trainXs.dispose()
      trainYs.dispose()
      testXs.dispose()
      testYs.dispose()
      setTraining(false)
    }
  }

  function stop() { stopRef.current = true }

  function reset() {
    setHistory([])
    setPredictions([])
    if (modelRef.current) {
      modelRef.current.dispose()
      modelRef.current = null
    }
    setStatus('Reset.')
  }

  const last = history[history.length - 1]

  return (
    <div className="app">
      <header className="header">
        <h1>CNNplay</h1>
        <span className="tag">Phase 1 · Classification</span>
        <span className="tag">backend: {backend || '...'}</span>
        <div className="spacer" />
        <a href="https://github.com/emilsar/CNNplay" target="_blank" rel="noreferrer">GitHub</a>
      </header>

      <aside className="panel">
        <section>
          <h2>Dataset</h2>
          <div className="hyper-row">
            <label>MNIST (28×28, 10 classes)</label>
            <button onClick={loadData} disabled={dataStatus === 'loading' || dataStatus === 'ready'}>
              {dataStatus === 'ready' ? 'Loaded' : dataStatus === 'loading' ? 'Loading...' : 'Load'}
            </button>
          </div>
        </section>

        <section>
          <h2>Architecture</h2>
          <ArchitectureEditor
            architecture={architecture}
            shapes={shapes}
            onChange={setArchitecture}
            disabled={training}
          />
        </section>
      </aside>

      <main className="center-content">
        <div className="intro">
          <strong>CNNplay</strong> — design a CNN by editing the layer stack on the left, set hyperparameters on the right, then click <code>Train</code>. Training runs entirely in your browser via TensorFlow.js. This is a <strong>Phase 1 prototype</strong>: classification on MNIST. Phase 2 will add segmentation on Oxford Pets.
        </div>

        <div className="metric-row">
          <div className="metric">
            <div className="label">Epoch</div>
            <div className="value">{last ? `${last.epoch}/${hyperparams.epochs}` : '—'}</div>
          </div>
          <div className="metric">
            <div className="label">Train Acc</div>
            <div className="value">{last ? `${(last.acc * 100).toFixed(1)}%` : '—'}</div>
          </div>
          <div className="metric">
            <div className="label">Val Acc</div>
            <div className="value">{last ? `${(last.valAcc * 100).toFixed(1)}%` : '—'}</div>
          </div>
          <div className="metric">
            <div className="label">Val Loss</div>
            <div className="value">{last ? last.valLoss.toFixed(3) : '—'}</div>
          </div>
        </div>

        <div className="chart-wrap">
          <h3>Loss</h3>
          <TrainingChart history={history} keys={['loss', 'valLoss']} />
        </div>
        <div className="chart-wrap">
          <h3>Accuracy</h3>
          <TrainingChart history={history} keys={['acc', 'valAcc']} />
        </div>

        <div className="chart-wrap">
          <h3>Sample predictions (after training)</h3>
          <Predictions cells={predictions} />
        </div>
      </main>

      <aside className="panel">
        <section>
          <h2>Hyperparameters</h2>
          <Hyperparameters
            value={hyperparams}
            onChange={setHyperparams}
            disabled={training}
          />
        </section>

        <section>
          <h2>Train</h2>
          <div className="train-controls">
            {!training ? (
              <button className="primary" onClick={train} disabled={!data}>Train</button>
            ) : (
              <button onClick={stop}>Stop</button>
            )}
            <button onClick={reset} disabled={training}>Reset</button>
          </div>
          <div className="status">{status}</div>
        </section>
      </aside>
    </div>
  )
}

export default App
