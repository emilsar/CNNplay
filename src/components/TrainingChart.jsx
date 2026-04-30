import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from 'recharts'

const COLORS = {
  loss: '#f87171',
  valLoss: '#fbbf24',
  acc: '#6ea8ff',
  valAcc: '#4ade80',
}

const LABELS = {
  loss: 'train loss',
  valLoss: 'val loss',
  acc: 'train acc',
  valAcc: 'val acc',
}

export default function TrainingChart({ history, keys }) {
  if (history.length === 0) {
    return <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#8a93a6', fontSize: 12 }}>No data yet — train a model.</div>
  }
  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={history} margin={{ top: 4, right: 12, bottom: 4, left: -16 }}>
        <CartesianGrid stroke="#2a2f3a" strokeDasharray="3 3" />
        <XAxis dataKey="epoch" stroke="#8a93a6" tick={{ fontSize: 11 }} />
        <YAxis stroke="#8a93a6" tick={{ fontSize: 11 }} />
        <Tooltip contentStyle={{ background: '#181b22', border: '1px solid #2a2f3a', fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {keys.map((k) => (
          <Line key={k} type="monotone" dataKey={k} stroke={COLORS[k]} strokeWidth={2} dot={false} name={LABELS[k]} isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
