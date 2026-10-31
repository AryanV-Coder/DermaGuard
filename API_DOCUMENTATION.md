# DermaGuard FastAPI & Grad-CAM Heatmap Integration Guide

## 1. Quick Start

### Launching the FastAPI Server
To start the backend server, run:
```bash
# In the workspace directory
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
or simply:
```bash
python app.py
```

### URLs
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **Built-in Interactive Testing Dashboard**: `http://127.0.0.1:8000/demo`

---

## 2. API Endpoints

### 1. `POST /api/analyze` (Main Clinical Triage & Grad-CAM)
Uploads a dermoscopy image and optional patient metadata. Runs the full 6-agent DermaGuard pipeline and generates the Grad-CAM heatmap.

- **Request Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Image file (`JPEG`, `PNG`, `WebP`) *(Required)*
  - `age`: Patient age as a float/int (e.g. `52`) *(Optional)*
  - `sex`: `'male'` | `'female'` | `'unknown'` *(Optional)*
  - `localization`: Body location, e.g. `'back'`, `'face'`, `'lower extremity'` *(Optional)*
  - `include_gradcam`: `true` | `false` (Defaults to `true`)
  - `colormap`: `'jet'` (clinical classic) | `'turbo'` (high-contrast)

#### JavaScript / React Integration Example:
```javascript
const formData = new FormData();
formData.append('file', imageFile); // File from <input type="file">
formData.append('age', 52);
formData.append('sex', 'female');
formData.append('localization', 'back');
formData.append('include_gradcam', 'true');

const response = await fetch('http://127.0.0.1:8000/api/analyze', {
  method: 'POST',
  body: formData,
});

const data = await response.json();
console.log('Track 2 Decision:', data.decision);       // "AUTO_TRIAGE" | "REFER" | "RE_IMAGE"
console.log('Prediction:', data.predicted_label);      // "nv", "mel", "bcc", etc.
console.log('Urgency:', data.urgency);                 // "ROUTINE" | "SOON" | "URGENT" | "REPEAT_IMAGE"

// Display Grad-CAM directly in <img> tag:
document.getElementById('heatmapImg').src = data.visualizations.gradcam_overlay;
document.getElementById('originalImg').src = data.visualizations.original_image;
```

#### JSON Response Structure:
```json
{
  "image_id": "ISIC_0024317",
  "predicted_label": "nv",
  "decision": "AUTO_TRIAGE",
  "urgency": "ROUTINE",
  "rationale": "Benign nv (conf=0.92), low suspicion, no safety conflicts. Cleared for auto-triage.",
  "visualizations": {
    "gradcam_overlay": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "original_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "pure_heatmap": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "focus_inside_lesion_pct": 96.2,
    "peak_attention_coords": {"x": 112, "y": 110},
    "target_class": "nv",
    "target_class_name": "Melanocytic Nevus",
    "description": "Grad-CAM visual attention (96.2% inside lesion) is distributed across the middle-center uniform pigment network, consistent with benign Melanocytic Nevus patterns."
  },
  "agent1_quality": {
    "quality_pass": true,
    "status": "PASS",
    "score": 1.0,
    "failed_checks": [],
    "metrics": {
      "blur": 142.5,
      "brightness": 128.2,
      "glare_ratio": 0.012,
      "occlude_ratio": 0.045
    }
  },
  "agent2_diagnosis": {
    "predicted_class": "nv",
    "predicted_label": "nv",
    "confidence": 0.924,
    "class_probabilities": {
      "nv": 0.924,
      "mel": 0.021,
      "bkl": 0.035,
      "bcc": 0.008,
      "akiec": 0.005,
      "vasc": 0.004,
      "df": 0.003
    },
    "melanoma_flag": false,
    "melanoma_prob": 0.021,
    "malignant_prob": 0.034
  },
  "agent3_lesion": {
    "asymmetry": 0.12,
    "border_score": 1.15,
    "colour_var": 18.4,
    "diameter_px": 85.0,
    "abcd_score": 0.88,
    "lesion_risk": "LOW"
  },
  "agent4_patient_risk": {
    "risk_level": "LOW",
    "risk_score": 0.20,
    "risk_factors": []
  },
  "agent5_safety": {
    "must_refer": false,
    "veto": false,
    "severity": "CLEAR",
    "safety_flags": []
  },
  "agent6_coordinator": {
    "decision": "AUTO_TRIAGE",
    "decision_rule": "RULE_3_BENIGN_ALL_CLEAR",
    "urgency": "ROUTINE",
    "confidence_tier": "HIGH"
  }
}
```

---

### 2. `POST /api/gradcam` (Standalone Grad-CAM Heatmap)
Allows generating a Grad-CAM heatmap for an arbitrary target class without running the full coordination graph.

- **Parameters**:
  - `file`: Image file *(Required)*
  - `target_class`: `'mel'` | `'bcc'` | `'akiec'` | `'nv'` | `'bkl'` *(Optional, default: highest predicted)*
  - `colormap`: `'jet'` | `'turbo'` *(Optional, default: `'jet'`)*
  - `alpha`: Blend opacity float `0.0` to `1.0` *(Optional, default: `0.55`)*

---

### 3. `POST /api/quality-check` (Technical Quality Screening)
Fast pre-screening endpoint (Agent 1) to check if an image is blurry, too dark, or has glare before running the heavy model inference.

---

### 4. `GET /api/classes`
Returns the 7 ISIC classification metadata:
`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`.
