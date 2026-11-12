import { AgentCard, Metric, MetricsRow, InfoRow, ProgressBar } from './AgentCard'
import styles from './Agent2Card.module.css'

const CLASS_FULL = {
  nv: 'Melanocytic Nevus',
  mel: 'Melanoma',
  bkl: 'Benign Keratosis',
  bcc: 'Basal Cell Carcinoma',
  akiec: "Actinic Keratosis",
  vasc: 'Vascular Lesion',
  df: 'Dermatofibroma',
}

const CLASS_MALIGNANT = { mel: true, bcc: true, akiec: true }

const CLASS_COLORS = {
  nv: 'teal', mel: 'red', bkl: 'teal', bcc: 'red', akiec: 'amber', vasc: 'green', df: 'green'
}

export default function Agent2Card({ data }) {
  if (!data) return null

  const predicted = data.predicted_class || data.predicted_label || '—'
  const confidence = data.confidence ?? 0
  const probs = data.class_probabilities || data.probabilities || {}
  const melFlag = data.melanoma_flag || false
  const melProb = data.melanoma_prob ?? 0
  const malProb = data.malignant_prob ?? 0

  const isMalignant = CLASS_MALIGNANT[predicted]

  // Sort probabilities descending
  const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1])

  return (
    <AgentCard
      number={2}
      title="Diagnosis Agent"
      status={predicted.toUpperCase()}
      statusColor={isMalignant ? 'red' : 'teal'}
    >
      <MetricsRow>
        <Metric
          label="Confidence"
          value={(confidence * 100).toFixed(1) + '%'}
          color={confidence > 0.7 ? 'green' : confidence > 0.4 ? 'amber' : 'red'}
        />
        <Metric label="Melanoma Prob" value={(melProb * 100).toFixed(1) + '%'} color={melFlag ? 'red' : 'teal'} />
        <Metric label="Malignant Prob" value={(malProb * 100).toFixed(1) + '%'} color={malProb > 0.3 ? 'red' : 'teal'} />
      </MetricsRow>

      <InfoRow label="Diagnosis" value={CLASS_FULL[predicted] || predicted} />

      {melFlag && (
        <div className={styles.flagBanner}>
          <strong>Melanoma flag active</strong> — specialist review mandatory
        </div>
      )}

      {/* Class probability bars */}
      <div className={styles.probsSection}>
        <div className={styles.probsLabel}>Class Probabilities</div>
        <div className={styles.probsList}>
          {sortedProbs.map(([cls, prob]) => (
            <ProgressBar
              key={cls}
              label={`${cls} · ${CLASS_FULL[cls] || cls}`}
              value={prob}
              colorClass={CLASS_COLORS[cls] || 'teal'}
            />
          ))}
        </div>
      </div>
    </AgentCard>
  )
}
