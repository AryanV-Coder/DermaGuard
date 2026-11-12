import { AgentCard, InfoRow, TagList } from './AgentCard'
import styles from './Agent5Card.module.css'

function severityColor(s) {
  if (s === 'VETO') return 'red'
  if (s === 'WARN') return 'amber'
  return 'green'
}

export default function Agent5Card({ data }) {
  if (!data) return null

  const veto = data.veto || false
  const mustRefer = data.must_refer || veto
  const severity = data.severity || (veto ? 'VETO' : 'CLEAR')
  const triggers = data.safety_flags || data.triggers || []
  const justification = data.justification || ''

  return (
    <AgentCard
      number={5}
      title="Safety & Veto Engine"
      icon="🛡️"
      status={severity}
      statusColor={severityColor(severity)}
    >
      {veto && (
        <div className={styles.vetoBanner}>
          <span>🚫</span>
          <strong>VETO ACTIVE — Mandatory specialist referral</strong>
        </div>
      )}
      {!veto && severity === 'CLEAR' && (
        <div className={styles.clearBanner}>
          <span>✅</span>
          <strong>All safety checks passed — no conflicts detected</strong>
        </div>
      )}
      {severity === 'WARN' && (
        <div className={styles.warnBanner}>
          <span>⚠️</span>
          <strong>Safety warnings raised</strong>
        </div>
      )}

      <InfoRow label="Must Refer" value={mustRefer ? '🔴 Yes' : '🟢 No'} />
      <InfoRow label="Veto Triggered" value={veto ? '🔴 Yes' : '🟢 No'} />
      <InfoRow label="Safety Severity" value={severity} />

      <div>
        <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-dim)', fontWeight: 600, marginBottom: '0.4rem' }}>Safety Triggers</div>
        <TagList tags={triggers} color={triggers.length > 0 ? 'red' : 'neutral'} />
      </div>

      {justification && (
        <div className={styles.justification}>
          <span>📋</span> {justification}
        </div>
      )}
    </AgentCard>
  )
}
