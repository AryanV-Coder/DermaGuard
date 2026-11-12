import { AgentCard, InfoRow } from './AgentCard'
import styles from './Agent6Card.module.css'

function decisionColor(d) {
  if (d === 'AUTO_TRIAGE') return 'green'
  if (d === 'REFER') return 'red'
  return 'amber'
}

function urgencyColor(u) {
  if (u === 'URGENT') return 'red'
  if (u === 'SOON') return 'amber'
  if (u === 'ROUTINE') return 'green'
  return 'amber'
}

function tierColor(t) {
  if (t === 'HIGH') return 'green'
  if (t === 'MODERATE') return 'amber'
  return 'red'
}

export default function Agent6Card({ data, decision, urgency }) {
  if (!data && !decision) return null

  const d = decision || data?.decision || '—'
  const u = urgency || data?.urgency || '—'
  const rule = data?.decision_rule || '—'
  const tier = data?.confidence_tier || '—'

  return (
    <AgentCard
      number={6}
      title="Coordinator Agent"
      icon="🎯"
      status={d}
      statusColor={decisionColor(d)}
    >
      <div className={styles.bigDecision}>
        <span className={`${styles.bigLabel} ${styles['col_' + decisionColor(d)]}`}>{d.replace('_', ' ')}</span>
      </div>

      <InfoRow label="Clinical Urgency" value={
        <span className={styles[`urgency_${urgencyColor(u)}`]}>{u}</span>
      } />
      <InfoRow label="Decision Rule" value={rule} mono />
      <InfoRow label="Confidence Tier" value={
        <span className={styles[`tier_${tierColor(tier)}`]}>{tier}</span>
      } />
    </AgentCard>
  )
}
