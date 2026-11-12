import styles from './DecisionBanner.module.css'

const DECISION_CONFIG = {
  AUTO_TRIAGE: {
    label: 'AUTO TRIAGE', color: 'green',
    desc: 'High-confidence benign lesion cleared for routine primary care.',
    iconSvg: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </svg>
    ),
  },
  REFER: {
    label: 'REFER', color: 'red',
    desc: 'Specialist referral required. Potential malignancy or safety veto triggered.',
    iconSvg: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    ),
  },
  RE_IMAGE: {
    label: 'RE-IMAGE', color: 'amber',
    desc: 'Image quality insufficient for safe diagnosis. Please recapture.',
    iconSvg: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="1 4 1 10 7 10" />
        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
      </svg>
    ),
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
          <div className={`${styles.iconWrap} ${styles['icon_' + cfg.color]}`}>
            {cfg.iconSvg}
          </div>
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
          <p>{rationale}</p>
        </div>
      )}
    </div>
  )
}

