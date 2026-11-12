import { useState, useRef, useCallback } from 'react'
import styles from './UploadPanel.module.css'

const API_BASE = 'http://127.0.0.1:8000'

const LOCALIZATIONS = [
  'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
  'foot', 'genital', 'hand', 'lower extremity', 'neck',
  'scalp', 'trunk', 'upper extremity',
]

export default function UploadPanel({ onAnalysis, onError, onReset, loading, setLoading }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [localization, setLocalization] = useState('')
  const [colormap, setColormap] = useState('jet')
  const fileRef = useRef()

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      onError('Please upload an image file (JPEG, PNG, or WebP).')
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
    onReset()
  }, [onError, onReset])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) { onError('Please select an image before running analysis.'); return }

    setLoading(true)
    onReset()

    const form = new FormData()
    form.append('file', file)
    if (age) form.append('age', age)
    if (sex) form.append('sex', sex)
    if (localization) form.append('localization', localization)
    form.append('include_gradcam', 'true')
    form.append('colormap', colormap)

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || `HTTP ${res.status}`)
      }
      const data = await res.json()
      onAnalysis(data)
    } catch (err) {
      onError(err.message || 'Network error. Is the DermaGuard API running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const clear = () => {
    setFile(null)
    setPreview(null)
    setAge('')
    setSex('')
    setLocalization('')
    onReset()
  }

  return (
    <form className={styles.panel} onSubmit={handleSubmit}>
      <div className={styles.panelHeader}>
        <span className={styles.panelIcon}>📋</span>
        <h2 className={styles.panelTitle}>Case Ingestion</h2>
      </div>

      {/* Dropzone */}
      <div
        className={`${styles.dropzone} ${dragging ? styles.dragging : ''} ${preview ? styles.hasImage : ''}`}
        onClick={() => !preview && fileRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') fileRef.current?.click() }}
        aria-label="Upload dermoscopy image"
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className={styles.hiddenInput}
          onChange={e => handleFile(e.target.files[0])}
        />
        {preview ? (
          <div className={styles.previewWrapper}>
            <img src={preview} alt="Selected lesion" className={styles.preview} />
            <div className={styles.previewOverlay}>
              <span className={styles.fileName}>{file?.name}</span>
            </div>
          </div>
        ) : (
          <div className={styles.dropContent}>
            <div className={styles.dropIcon}>🔬</div>
            <p className={styles.dropTitle}>Drop dermoscopy image here</p>
            <p className={styles.dropSub}>or click to browse · JPEG · PNG · WebP</p>
          </div>
        )}
      </div>

      {preview && (
        <button type="button" className={styles.clearBtn} onClick={clear}>
          ✕ Remove image
        </button>
      )}

      {/* Patient Parameters */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>
          <span>👤</span> Patient Parameters
        </h3>

        <div className={styles.row2}>
          <div className={styles.field}>
            <label>Age (years)</label>
            <input
              type="number"
              min="0" max="120"
              placeholder="e.g. 52"
              value={age}
              onChange={e => setAge(e.target.value)}
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label>Biological Sex</label>
            <select value={sex} onChange={e => setSex(e.target.value)} className={styles.input}>
              <option value="">Unknown</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>
        </div>

        <div className={styles.field}>
          <label>Anatomical Site</label>
          <select value={localization} onChange={e => setLocalization(e.target.value)} className={styles.input}>
            <option value="">Unknown / Not specified</option>
            {LOCALIZATIONS.map(l => (
              <option key={l} value={l}>{l.charAt(0).toUpperCase() + l.slice(1)}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Visualization Options */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>
          <span>🌡️</span> Grad-CAM Options
        </h3>
        <div className={styles.colormapRow}>
          {[
            { val: 'jet', label: 'Jet', desc: 'Clinical classic' },
            { val: 'turbo', label: 'Turbo', desc: 'High contrast' },
          ].map(o => (
            <label key={o.val} className={`${styles.colormapOption} ${colormap === o.val ? styles.colormapSelected : ''}`}>
              <input type="radio" name="colormap" value={o.val} checked={colormap === o.val} onChange={() => setColormap(o.val)} className={styles.hiddenInput} />
              <span className={styles.colormapLabel}>{o.label}</span>
              <span className={styles.colormapDesc}>{o.desc}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        className={styles.submitBtn}
        disabled={!file || loading}
      >
        {loading
          ? <><span className={styles.btnSpinner} /> Analyzing…</>
          : <><span>⚡</span> Run 6-Agent Analysis</>
        }
      </button>

      {/* Disclaimer */}
      <p className={styles.disclaimer}>
        ⚠️ For research & demo purposes only. Not a clinical diagnostic device.
      </p>
    </form>
  )
}
