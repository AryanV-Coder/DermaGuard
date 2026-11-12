import { AgentCard, Metric, MetricsRow, InfoRow, TagList, ProgressBar } from './AgentCard'

function riskColor(level) {
  if (!level) return 'teal'
  const l = level.toUpperCase()
  if (l === 'HIGH') return 'red'
  if (l === 'MODERATE') return 'amber'
  return 'green'
}

export default function Agent4Card({ data }) {
  if (!data) return null

  const riskLevel = data.risk_level || data.risk_indicator || '—'
  const riskScore = data.risk_score ?? 0
  const factors = data.risk_factors || data.clinical_notes || []
  const missing = data.missing_fields || []

  return (
    <AgentCard
      number={4}
      title="Patient Context"
      status={riskLevel}
      statusColor={riskColor(riskLevel)}
    >
      <MetricsRow>
        <Metric label="Risk Score" value={(riskScore * 100).toFixed(0) + '%'} color={riskColor(riskLevel)} />
        <Metric label="Risk Level" value={riskLevel} color={riskColor(riskLevel)} />
      </MetricsRow>

      <ProgressBar label="Patient Risk Score" value={riskScore} colorClass={riskScore > 0.6 ? 'red' : riskScore > 0.3 ? 'amber' : 'green'} />

      {factors.length > 0 && (
        <div>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-dim)', fontWeight: 600, marginBottom: '0.4rem' }}>Clinical Notes</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {factors.map((f, i) => (
              <div key={i} style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', gap: '0.4rem' }}>
                <span style={{ color: 'var(--teal)', flexShrink: 0, fontWeight: 700 }}>›</span> {f}
              </div>
            ))}
          </div>
        </div>
      )}

      {missing.length > 0 && (
        <div>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-dim)', fontWeight: 600, marginBottom: '0.4rem' }}>Missing Fields</div>
          <TagList tags={missing} color="amber" />
        </div>
      )}
    </AgentCard>
  )
}
