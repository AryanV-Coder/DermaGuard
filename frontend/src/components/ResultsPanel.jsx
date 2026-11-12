import DecisionBanner from './DecisionBanner'
import HeatmapViewer from './HeatmapViewer'
import Agent1Card from './Agent1Card'
import Agent2Card from './Agent2Card'
import Agent3Card from './Agent3Card'
import Agent4Card from './Agent4Card'
import Agent5Card from './Agent5Card'
import Agent6Card from './Agent6Card'
import styles from './ResultsPanel.module.css'

export default function ResultsPanel({ result }) {
  if (!result) return null

  const {
    decision, urgency, rationale, clinical_explanation,
    visualizations,
    agent1_quality,
    agent2_diagnosis,
    agent3_lesion,
    agent4_patient_risk,
    agent5_safety,
    agent6_coordinator,
  } = result

  return (
    <div className={styles.container}>
      {/* Decision Banner */}
      <DecisionBanner decision={decision} urgency={urgency} rationale={rationale} />

      {/* Grad-CAM Heatmap Viewer */}
      {visualizations && (
        <HeatmapViewer
          visualizations={visualizations}
          predictedLabel={result.predicted_label}
        />
      )}

      {/* Agent Cards Grid */}
      <div className={styles.agentsGrid}>
        <Agent1Card data={agent1_quality} />
        <Agent2Card data={agent2_diagnosis} />
        <Agent3Card data={agent3_lesion} />
        <Agent4Card data={agent4_patient_risk} />
        <Agent5Card data={agent5_safety} />
        <Agent6Card data={agent6_coordinator} decision={decision} urgency={urgency} />
      </div>

      {/* Clinical Explanation */}
      {clinical_explanation && (
        <div className={styles.explanationCard}>
          <div className={styles.explanationHeader}>
            <span>✨</span>
            <h3>Gemini Clinical Explanation</h3>
          </div>
          <p className={styles.explanationText}>{clinical_explanation}</p>
        </div>
      )}

      {/* Raw JSON toggle */}
      <details className={styles.rawJson}>
        <summary className={styles.rawSummary}>🔍 Raw Pipeline JSON Response</summary>
        <pre className={styles.rawPre}>{JSON.stringify(result, null, 2)}</pre>
      </details>
    </div>
  )
}

