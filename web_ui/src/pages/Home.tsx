import { motion } from 'framer-motion'
import { Card, StatCard } from '../components/ui/Card'
import { OverviewChart } from '../components/charts/OverviewChart'
import { stats, alerts } from '../data/placeholder'
import '../components/charts/charts.css'

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.06 },
  },
}

const item = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0 },
}

export function Home() {
  return (
    <motion.div
      className="p-8 pb-24 max-w-[1600px] mx-auto"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="mb-8">
        <motion.h2
          className="text-3xl font-semibold text-text-primary mb-2 tracking-tight"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          Overview
        </motion.h2>
        <motion.p
          className="text-text-secondary max-w-2xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
        >
          Real-time stats and alerts for your CORTEX RAG system.
        </motion.p>
      </div>

      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {stats.map((s) => (
          <motion.div key={s.title} variants={item}>
            <StatCard
              title={s.title}
              value={s.value}
              subtitle={s.subtitle}
              trend={s.trend}
              trendValue={s.trendValue}
            />
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <Card className="lg:col-span-2 min-h-[400px]">
          <h3 className="text-lg font-semibold text-text-primary mb-6">Query volume (7 days)</h3>
          <div className="h-[320px] w-full">
            <OverviewChart />
          </div>
        </Card>
        <Card className="h-full min-h-[400px]">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Alerts</h3>
          <ul className="space-y-3">
            {alerts.map((a) => (
              <li
                key={a.id}
                className={`p-3 rounded-lg border-l-2 bg-white/5 flex flex-col gap-1 ${a.type === 'info' ? 'border-accent-primary' :
                    a.type === 'success' ? 'border-accent-success' :
                      a.type === 'warning' ? 'border-accent-warning' : 'border-text-tertiary'
                  }`}
              >
                <div className="flex justify-between items-start gap-4">
                  <span className="text-sm text-text-primary leading-snug">{a.message}</span>
                  <span className="text-xs text-text-tertiary whitespace-nowrap">{a.time}</span>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </motion.div>
    </motion.div>
  )
}
