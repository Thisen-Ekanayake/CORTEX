import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { chartData } from '../../data/placeholder'

const COLORS = {
  queries: 'rgba(88, 166, 255, 0.4)',
  queriesStroke: '#58a6ff',
  rag: 'rgba(63, 185, 80, 0.3)',
  ragStroke: '#3fb950',
}

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tooltip">
      <span className="chart-tooltip__label">{label}</span>
      {payload.map((p) => (
        <span key={p.name} className="chart-tooltip__item">
          {p.name}: <strong>{p.value}</strong>
        </span>
      ))}
    </div>
  )
}

export function OverviewChart() {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="queriesGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.queriesStroke} stopOpacity={0.4} />
              <stop offset="100%" stopColor={COLORS.queriesStroke} stopOpacity={0} />
            </linearGradient>
            <linearGradient id="ragGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={COLORS.ragStroke} stopOpacity={0.3} />
              <stop offset="100%" stopColor={COLORS.ragStroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
          <XAxis dataKey="name" stroke="var(--text-tertiary)" fontSize={12} tickLine={false} axisLine={false} />
          <YAxis stroke="var(--text-tertiary)" fontSize={12} tickLine={false} axisLine={false} width={32} />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border-subtle)', strokeWidth: 1 }} />
          <Area type="monotone" dataKey="queries" stroke={COLORS.queriesStroke} fill="url(#queriesGrad)" strokeWidth={2} isAnimationActive animationDuration={800} />
          <Area type="monotone" dataKey="rag" stroke={COLORS.ragStroke} fill="url(#ragGrad)" strokeWidth={2} isAnimationActive animationDuration={800} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
