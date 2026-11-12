import { useState, useCallback } from 'react'
import UploadPanel from './components/UploadPanel'
import ResultsPanel from './components/ResultsPanel'
import Header from './components/Header'
import styles from './App.module.css'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleAnalysis = useCallback((data) => {
    setResult(data)
    setError(null)
  }, [])

  const handleError = useCallback((msg) => {
    setError(msg)
    setResult(null)
  }, [])

  const handleReset = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return (
    <div className={styles.root}>
      <Header />
      <main className={styles.main}>
        <div className={styles.layout}>
          <UploadPanel
            onAnalysis={handleAnalysis}
            onError={handleError}
            onReset={handleReset}
            loading={loading}
            setLoading={setLoading}
          />
          <div className={styles.rightPane}>
            {error && (
              <div className={styles.errorBanner}>
                <span className={styles.errorIcon}>⚠️</span>
                <div>
                  <strong>Analysis Error</strong>
                  <p>{error}</p>
                </div>
              </div>
            )}
            {!result && !error && !loading && (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>🩺</div>
                <h2>Awaiting Dermoscopy Image</h2>
                <p>Upload a lesion image and configure patient parameters on the left panel to begin the 6-agent clinical triage pipeline.</p>
                <div className={styles.agentPills}>
                  {['Quality Supervisor','Diagnosis Agent','Lesion Analysis','Patient Context','Safety Veto','Coordinator'].map((a, i) => (
                    <span key={a} className={styles.pill}>Agent {i+1}: {a}</span>
                  ))}
                </div>
              </div>
            )}
            {loading && (
              <div className={styles.loadingState}>
                <div className={styles.spinner} />
                <h2>Running 6-Agent Pipeline…</h2>
                <p>Processing image quality, diagnosis, ABCD morphometrics, patient context, safety review, and final coordination.</p>
                <div className={styles.loadingSteps}>
                  {['🔬 Quality Check','🧠 Deep Learning Diagnosis','📐 ABCD Morphometrics','👤 Patient Risk Context','🛡️ Safety Veto Review','🎯 Coordinator Decision','🌡️ Grad-CAM Heatmap'].map(s => (
                    <div key={s} className={styles.loadingStep}><span className={styles.loadingDot} />{s}</div>
                  ))}
                </div>
              </div>
            )}
            {result && !loading && <ResultsPanel result={result} />}
          </div>
        </div>
      </main>
    </div>
  )
}
