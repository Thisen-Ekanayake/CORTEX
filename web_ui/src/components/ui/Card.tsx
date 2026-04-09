import { motion } from 'framer-motion'

type CardProps = {
  children: React.ReactNode
  className?: string
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <motion.div
      className={`glass-panel p-6 rounded-2xl ${className}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
    >
      {children}
    </motion.div>
  )
}

type StatCardProps = {
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
}

export function StatCard({ title, value, subtitle, trend, trendValue }: StatCardProps) {
  return (
    <Card>
      <div className="flex flex-col gap-2">
        <span className="text-sm font-medium text-text-secondary">{title}</span>
        <div className="flex items-end justify-between">
          <span className="text-3xl font-bold text-text-primary tracking-tight">{value}</span>
          {trend && trendValue && (
            <span className={`text-xs font-medium px-2 py-1 rounded-full flex items-center gap-1 ${trend === 'up' ? 'text-accent-success bg-accent-success/10' :
                trend === 'down' ? 'text-accent-error bg-accent-error/10' :
                  'text-text-tertiary bg-white/5'
              }`}>
              {trend === 'up' && '↑'}
              {trend === 'down' && '↓'}
              {trendValue}
            </span>
          )}
        </div>

        {subtitle && <span className="text-xs text-text-tertiary mt-1">{subtitle}</span>}
      </div>
    </Card>
  )
}
