import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Card, StatCard } from '../components/ui/Card'
import { OverviewChart } from '../components/charts/OverviewChart'
import { getStats, type Stats } from '../lib/api'
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

type StatCardData = {
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
}

function buildCards(stats: Stats): StatCardData[] {
  const delta = stats.queriesToday - stats.queriesYesterday
  const hasHistory = stats.queriesToday > 0 || stats.queriesYesterday > 0

  return [
    {
      title: 'Documents indexed',
      value: stats.documentsIndexed.toLocaleString(),
      subtitle: 'In the document corpus',
    },
    {
      title: 'Queries today',
      value: stats.queriesToday.toLocaleString(),
      subtitle: `${stats.queriesLast7Days.toLocaleString()} in last 7 days`,
      trend: hasHistory ? (delta > 0 ? 'up' : delta < 0 ? 'down' : 'neutral') : undefined,
      trendValue: hasHistory ? (delta === 0 ? '—' : `${delta > 0 ? '+' : ''}${delta}`) : undefined,
    },
    {
      title: 'Router accuracy',
      value: stats.routerAccuracy === null ? '—' : `${Math.round(stats.routerAccuracy * 100)}%`,
      subtitle:
        stats.totalPredictions > 0
          ? `${stats.totalPredictions.toLocaleString()} predictions`
          : 'No feedback yet',
    },
    {
      title: 'Queries (7 days)',
      value: stats.queriesLast7Days.toLocaleString(),
      subtitle: `RAG: ${stats.ragQueries} · Chat: ${stats.chatQueries}`,
    },
  ]
}

export function Home() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    getStats()
      .then((data) => {
        if (!cancelled) setStats(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load stats')
      })
    return () => {
      cancelled = true
    }
  }, [])

  const cards = stats ? buildCards(stats) : []

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
          Real-time stats and activity for your CORTEX RAG system.
        </motion.p>
      </div>

      {error && (
        <div className="mb-8 p-4 rounded-lg border-l-2 border-accent-error bg-accent-error/5 text-sm text-text-secondary">
          Couldn’t reach the CORTEX API: {error}. Is <code>cortex_api</code> running on port 8001?
        </div>
      )}

      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        variants={container}
        initial="hidden"
        animate="show"
      >
        {stats
          ? cards.map((s) => (
              <motion.div key={s.title} variants={item}>
                <StatCard
                  title={s.title}
                  value={s.value}
                  subtitle={s.subtitle}
                  trend={s.trend}
                  trendValue={s.trendValue}
                />
              </motion.div>
            ))
          : !error &&
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="glass-panel p-6 rounded-2xl h-[116px] animate-pulse" />
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
            {stats ? (
              <OverviewChart data={stats.volumeSeries} />
            ) : (
              <div className="h-full w-full rounded-xl bg-white/5 animate-pulse" />
            )}
          </div>
        </Card>
        <Card className="h-full min-h-[400px]">
          <h3 className="text-lg font-semibold text-text-primary mb-4">Recent activity</h3>
          {stats ? (
            <ul className="space-y-3">
              {stats.activity.map((a) => (
                <li
                  key={a.id}
                  className={`p-3 rounded-lg border-l-2 bg-white/5 flex flex-col gap-1 ${
                    a.type === 'info'
                      ? 'border-accent-primary'
                      : a.type === 'success'
                        ? 'border-accent-success'
                        : a.type === 'warning'
                          ? 'border-accent-warning'
                          : 'border-text-tertiary'
                  }`}
                >
                  <div className="flex justify-between items-start gap-4">
                    <span className="text-sm text-text-primary leading-snug break-words">
                      {a.message}
                    </span>
                    {a.time && (
                      <span className="text-xs text-text-tertiary whitespace-nowrap">{a.time}</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="h-12 rounded-lg bg-white/5 animate-pulse" />
              ))}
            </div>
          )}
        </Card>
      </motion.div>
    </motion.div>
  )
}
