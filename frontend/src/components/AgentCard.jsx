import styles from './AgentCard.module.css'

export function AgentCard({ number, title, status, statusColor, children }) {
  const colorClass = statusColor || 'teal'
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div className={styles.agentBadge}>
          <span className={styles.numBadge}>{number}</span>
          <div>
            <div className={styles.agentNum}>Agent {number}</div>
            <div className={styles.agentTitle}>{title}</div>
          </div>
        </div>
        {status && (
          <span className={`${styles.status} ${styles['status_' + colorClass]}`}>
            {status}
          </span>
        )}
      </div>
      <div className={styles.body}>{children}</div>
    </div>
  )
}

export function Metric({ label, value, sub, color }) {
  return (
    <div className={styles.metric}>
      <div className={`${styles.metricValue} ${color ? styles['metric_' + color] : ''}`}>{value}</div>
      <div className={styles.metricLabel}>{label}</div>
      {sub && <div className={styles.metricSub}>{sub}</div>}
    </div>
  )
}

export function MetricsRow({ children }) {
  return <div className={styles.metricsRow}>{children}</div>
}

export function InfoRow({ label, value, mono }) {
  return (
    <div className={styles.infoRow}>
      <span className={styles.infoLabel}>{label}</span>
      <span className={`${styles.infoValue} ${mono ? styles.mono : ''}`}>{value}</span>
    </div>
  )
}

export function ProgressBar({ label, value, max = 1, colorClass }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className={styles.progressRow}>
      <div className={styles.progressHeader}>
        <span className={styles.progressLabel}>{label}</span>
        <span className={styles.progressVal}>{typeof value === 'number' ? value.toFixed(3) : value}</span>
      </div>
      <div className={styles.progressTrack}>
        <div
          className={`${styles.progressFill} ${styles['fill_' + (colorClass || 'teal')]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function Tag({ children, color }) {
  return <span className={`${styles.tag} ${styles['tag_' + (color || 'neutral')]}`}>{children}</span>
}

export function TagList({ tags, color }) {
  if (!tags || tags.length === 0) return <span className={styles.none}>None</span>
  return (
    <div className={styles.tagList}>
      {tags.map((t, i) => <Tag key={i} color={color}>{t}</Tag>)}
    </div>
  )
}
