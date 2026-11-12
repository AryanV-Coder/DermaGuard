import styles from './DecisionBanner.module.css'

const DECISION_CONFIG = {
  AUTO_TRIAGE: {
    icon: '🟢', label: 'AUTO TRIAGE', color: 'green',
    desc: 'High-confidence benign lesion cleared for routine primary care.',
  },
  REFER: {
    icon: '🔴', label: 'REFER', color: 'red',
    desc: 'Specialist referral required. Potential malignancy or safety veto triggered.',
  },
  RE_IMAGE: {
    icon: '🟡', label: 'RE-IMAGE', color: 'amber',
    desc: 'Image quality insufficient for safe diagnosis. Please recapture.',
  },
}

const URGENCY_LABELS = {
  ROUTINE: { label: 'Routine', color: 'green' },
  SOON: { label: 'Soon', color: 'amber' },
  URGENT: { label: 'Urgent', color: 'red' },
  REPEAT_IMAGE: { label: 'Repeat Image', color: 'amber' },
}

export default function DecisionBanner({ decision, urgency, rationale }) {
  const cfg = DECISION_CONFIG[decision] || DECISION_CONFIG.RE_IMAGE
  const urg = URGENCY_LABELS[urgency] || { label: urgency, color: 'amber' }

  return (
    <div className={`${styles.banner} ${styles[cfg.color]}`}>
      <div className={styles.top}>
        <div className={styles.left}>
          <span className={styles.icon}>{cfg.icon}</span>
          <div>
            <h2 className={styles.decision}>{cfg.label}</h2>
            <p className={styles.desc}>{cfg.desc}</p>
          </div>
        </div>
        <div className={`${styles.urgencyBadge} ${styles['urg_' + urg.color]}`}>
          Urgency: <strong>{urg.label}</strong>
        </div>
      </div>
      {rationale && (
        <div className={styles.rationale}>
          <span className={styles.rationaleIcon}>📝</span>
          <p>{rationale}</p>
        </div>
      )}
    </div>
  )
}

