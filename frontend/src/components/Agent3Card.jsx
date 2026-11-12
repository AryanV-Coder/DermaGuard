import { AgentCard, Metric, MetricsRow, InfoRow, ProgressBar } from './AgentCard'

function suspicionColor(level) {
  if (level === 'HIGH') return 'red'
  if (level === 'MODERATE') return 'amber'
  return 'green'
}

export default function Agent3Card({ data }) {
  if (!data) return null

  const suspicion = data.suspicion_level || data.lesion_risk || 'LOW'
  const suspicionScore = data.suspicion_score ?? 0
  const asymmetry = data.asymmetry ?? data.asymmetry_index ?? 0
  const border = data.border_score ?? data.border_irregularity ?? 0
  const colourVar = data.colour_var ?? data.color_variation ?? 0
  const diameter = data.diameter_px ?? data.equivalent_diameter_px ?? 0
  const abcd = data.abcd_score ?? 0
  const seg = data.segmentation_backend || 'unknown'
  const flags = data.reliability_flags || []

  return (
    <AgentCard
      number={3}
      title="Lesion Analysis (ABCD)"
      icon="📐"
      status={suspicion}
      statusColor={suspicionColor(suspicion)}
    >
      <MetricsRow>
        <Metric label="Suspicion" value={(suspicionScore * 100).toFixed(0) + '%'} color={suspicionColor(suspicion)} sub={suspicion} />
        <Metric label="Diameter" value={diameter ? diameter.toFixed(0) + 'px' : '—'} />
        {abcd > 0 && <Metric label="ABCD Score" value={abcd.toFixed(2)} />}
      </MetricsRow>

      <ProgressBar label="Asymmetry Index (A)" value={asymmetry} colorClass="amber" />
      <ProgressBar label={`Border Irregularity (B)${border > 1 ? ' — raw' : ''}`} value={Math.min(1, border)} colorClass="amber" />
      <ProgressBar label="Colour Variation (C)" value={typeof colourVar === 'number' && colourVar > 1 ? colourVar / 100 : colourVar} colorClass="amber" />

      <InfoRow label="Segmentation Backend" value={seg} mono />

      {flags.length > 0 && (
        <div>
          <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-dim)', fontWeight: 600, marginBottom: '0.4rem' }}>Reliability Flags</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
            {flags.map((f, i) => (
              <span key={i} style={{ background: 'rgba(245,158,11,0.1)', border: '1px solid rgba(245,158,11,0.25)', color: '#fcd34d', fontSize: '0.68rem', padding: '0.2rem 0.5rem', borderRadius: '999px' }}>
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </AgentCard>
  )
}

