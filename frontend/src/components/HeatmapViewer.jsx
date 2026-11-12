import { useState } from 'react'
import styles from './HeatmapViewer.module.css'

const CLASS_FULL_NAMES = {
  nv: 'Melanocytic Nevus',
  mel: 'Melanoma',
  bkl: 'Benign Keratosis',
  bcc: 'Basal Cell Carcinoma',
  akiec: "Actinic Keratosis / Bowen's",
  vasc: 'Vascular Lesion',
  df: 'Dermatofibroma',
}

const VIEWS = [
  { key: 'gradcam_overlay', label: '🌡️ Grad-CAM Overlay', desc: 'CAM fused with original' },
  { key: 'original_image', label: '📸 Original', desc: 'Input dermoscopy image' },
  { key: 'pure_heatmap', label: '🔥 Pure Heatmap', desc: 'Raw activation map' },
]

export default function HeatmapViewer({ visualizations, predictedLabel }) {
  const [activeView, setActiveView] = useState('gradcam_overlay')

  const src = visualizations?.[activeView]
  const focusPct = visualizations?.focus_inside_lesion_pct
  const description = visualizations?.description
  const targetClass = visualizations?.target_class || predictedLabel
  const targetClassFull = CLASS_FULL_NAMES[targetClass] || targetClass

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.headerLeft}>
          <span className={styles.headerIcon}>🌡️</span>
          <div>
            <h3 className={styles.cardTitle}>Grad-CAM Visual Attention</h3>
            <p className={styles.cardSub}>
              Target class: <strong className={styles.classLabel}>{targetClass}</strong>
              {' · '}{targetClassFull}
            </p>
          </div>
        </div>
        {focusPct !== undefined && (
          <div className={styles.focusChip}>
            <span className={styles.focusPct}>{Number(focusPct).toFixed(1)}%</span>
            <span className={styles.focusLabel}>attention in lesion</span>
          </div>
        )}
      </div>

      {/* View Switcher */}
      <div className={styles.viewTabs}>
        {VIEWS.map(v => (
          <button
            key={v.key}
            className={`${styles.tab} ${activeView === v.key ? styles.tabActive : ''}`}
            onClick={() => setActiveView(v.key)}
            disabled={!visualizations?.[v.key]}
          >
            <span className={styles.tabLabel}>{v.label}</span>
            <span className={styles.tabDesc}>{v.desc}</span>
          </button>
        ))}
      </div>

      {/* Image Display */}
      <div className={styles.imageWrapper}>
        {src ? (
          <img src={src} alt={activeView} className={styles.heatmapImg} />
        ) : (
          <div className={styles.noImage}>
            <span>🚫</span>
            <p>Visualization not available</p>
          </div>
        )}
      </div>

      {/* Focus progress bar */}
      {focusPct !== undefined && (
        <div className={styles.focusBar}>
          <div className={styles.focusBarLabel}>
            <span>Attention focus inside lesion boundary</span>
            <span className={styles.focusBarPct}>{Number(focusPct).toFixed(1)}%</span>
          </div>
          <div className={styles.focusTrack}>
            <div
              className={styles.focusFill}
              style={{ width: `${Math.min(100, focusPct)}%` }}
            />
          </div>
        </div>
      )}

      {/* Description */}
      {description && (
        <p className={styles.description}>
          <span className={styles.descIcon}>💡</span>
          {description}
        </p>
      )}
    </div>
  )
}

