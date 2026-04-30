export default function Hyperparameters({ value, onChange, disabled }) {
  const set = (patch) => onChange({ ...value, ...patch })
  return (
    <>
      <div className="hyper-row">
        <label>Optimizer</label>
        <select value={value.optimizer} onChange={(e) => set({ optimizer: e.target.value })} disabled={disabled}>
          <option value="adam">adam</option>
          <option value="sgd">sgd</option>
          <option value="rmsprop">rmsprop</option>
        </select>
      </div>
      <div className="hyper-row">
        <label>Learning rate</label>
        <input type="number" min="0.0001" max="1" step="0.0001"
          value={value.learningRate}
          onChange={(e) => set({ learningRate: +e.target.value })} disabled={disabled} />
      </div>
      <div className="hyper-row">
        <label>Batch size</label>
        <input type="number" min="1" max="512" step="1"
          value={value.batchSize}
          onChange={(e) => set({ batchSize: +e.target.value })} disabled={disabled} />
      </div>
      <div className="hyper-row">
        <label>Epochs</label>
        <input type="number" min="1" max="50" step="1"
          value={value.epochs}
          onChange={(e) => set({ epochs: +e.target.value })} disabled={disabled} />
      </div>
    </>
  )
}
