import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>🩺</span>
          </div>
          <div>
            <h1 className={styles.title}>DermaGuard</h1>
            <p className={styles.subtitle}>Autonomous 6-Agent Clinical Triage · Jaypee Agentic AI Summit 2026</p>
          </div>
        </div>
        <div className={styles.badges}>
          <span className={styles.badge}>Track 2: Healthcare</span>
          <span className={styles.badgeAlt}>Grad-CAM Explainability</span>
          <a className={styles.docsLink} href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            API Docs ↗
          </a>
        </div>
      </div>
    </header>
  )
}
