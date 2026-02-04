import { motion } from 'framer-motion'
import { Card, StatCard } from '../components/ui/Card'
import { OverviewChart } from '../components/charts/OverviewChart'
import { stats, alerts } from '../data/placeholder'
import '../components/charts/charts.css'
import './pages.css'

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
      className="page page--home"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.h2
        className="page__title"
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        Overview
      </motion.h2>
      <motion.p
        className="page__subtitle"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      >
        Real-time stats and alerts for your CORTEX RAG system.
      </motion.p>

      <motion.div
        className="page__grid page__grid--stats"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {stats.map((s, i) => (
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
        className="page__grid page__grid--main"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        <Card className="card--chart">
          <h3 className="card__heading">Query volume (7 days)</h3>
          <OverviewChart />
        </Card>
        <Card className="card--alerts">
          <h3 className="card__heading">Alerts</h3>
          <ul className="alerts-list">
            {alerts.map((a) => (
              <li key={a.id} className={`alerts-list__item alerts-list__item--${a.type}`}>
                <span className="alerts-list__message">{a.message}</span>
                <span className="alerts-list__time">{a.time}</span>
              </li>
            ))}
          </ul>
        </Card>
      </motion.div>
    </motion.div>
  )
}
