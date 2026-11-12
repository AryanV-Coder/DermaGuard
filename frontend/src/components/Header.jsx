import styles from './Header.module.css'

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <div className={styles.logo}>🩺</div>
          <div>
            <h1 className={styles.title}>DermaGuard</h1>
            <p className={styles.subtitle}>Autonomous Dermatology Clinical Triage</p>
          </div>
        </div>
      </div>
    </header>
  )
}
