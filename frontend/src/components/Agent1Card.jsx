import { AgentCard, Metric, MetricsRow, InfoRow, TagList } from './AgentCard'

export default function Agent1Card({ data }) {
  if (!data) return null
  const status = data.status || (data.quality_pass ? 'PASS' : 'FAIL')
  const score = data.score ?? data.quality_score ?? 0
  const failedChecks = data.failed_checks || []
  const metrics = data.metrics || {}

  const statusColor = status === 'PASS' ? 'green' : status === 'FAIL' ? 'red' : 'amber'

  return (
    <AgentCard number={1} title="Quality Supervisor" status={status} statusColor={statusColor}>
      <MetricsRow>
        <Metric label="Quality Score" value={(score * 100).toFixed(0) + '%'} color={status === 'PASS' ? 'green' : 'red'} />
        {metrics.blur !== undefined && <Metric label="Blur (Lap.)" value={metrics.blur?.toFixed(1) ?? '—'} sub="higher = sharper" />}
        {metrics.glare_ratio !== undefined && <Metric label="Glare Ratio" value={(metrics.glare_ratio * 100).toFixed(1) + '%'} />}
        {metrics.occlude_ratio !== undefined && <Metric label="Occlusion" value={(metrics.occlude_ratio * 100).toFixed(1) + '%'} />}
      </MetricsRow>

      {metrics.brightness !== undefined && (
        <InfoRow label="Brightness" value={metrics.brightness?.toFixed(1) + ' lum'} />
      )}

      <div>
        <span style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-dim)', fontWeight: 600 }}>
          Failed Checks
        </span>
        <div style={{ marginTop: '0.4rem' }}>
          <TagList tags={failedChecks} color={failedChecks.length > 0 ? 'red' : 'neutral'} />
        </div>
      </div>

      {data.recommendation && (
        <InfoRow label="Recommendation" value={data.recommendation} />
      )}
    </AgentCard>
  )
}
