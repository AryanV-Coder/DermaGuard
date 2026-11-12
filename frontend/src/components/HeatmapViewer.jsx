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

export default function HeatmapViewer({ visualizations, predictedLabel }) {
  const overlaySrc = visualizations?.gradcam_overlay
  const originalSrc = visualizations?.original_image
  const focusPct = visualizations?.focus_inside_lesion_pct
  const description = visualizations?.description
  const targetClass = visualizations?.target_class || predictedLabel
  const targetClassFull = CLASS_FULL_NAMES[targetClass] || targetClass

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.headerLeft}>
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

      {/* Side-by-Side Images: Original vs Grad-CAM Overlay */}
      <div className={styles.comparisonGrid}>
        <div className={styles.imageColumn}>
          <div className={styles.columnHeader}>
            <span className={styles.columnTitle}>Original Image</span>
            <span className={styles.columnSubtitle}>Input dermoscopy</span>
          </div>
          <div className={styles.imageWrapper}>
            {originalSrc ? (
              <img src={originalSrc} alt="Original dermoscopy lesion" className={styles.previewImg} />
            ) : (
              <div className={styles.noImage}>
                <p>Original image not available</p>
              </div>
            )}
          </div>
        </div>

        <div className={styles.imageColumn}>
          <div className={styles.columnHeader}>
            <span className={styles.columnTitle}>Grad-CAM Heatmap</span>
            <span className={styles.columnSubtitle}>Visual attention overlay</span>
          </div>
          <div className={styles.imageWrapper}>
            {overlaySrc ? (
              <img src={overlaySrc} alt="Grad-CAM visual heatmap overlay" className={styles.previewImg} />
            ) : (
              <div className={styles.noImage}>
                <p>Heatmap overlay not available</p>
              </div>
            )}
          </div>
        </div>
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
          {description}
        </p>
      )}
    </div>
  )
}

