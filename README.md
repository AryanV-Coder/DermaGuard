# DermaGuard

DermaGuard is an explainable, multi-agent skin-lesion triage system. It combines
technical image-quality screening, lesion classification, ABCD-style morphology,
patient context, safety guardrails, and Grad-CAM visual explanations into one
LangGraph workflow.

> **Medical safety notice:** DermaGuard is a research and decision-support
> prototype. It is not a medical device and must not be used as a diagnosis or
> as a substitute for a qualified clinician. Do not use its output to delay
> professional care.

## What it does

The pipeline processes dermoscopy images through six specialized agents:

1. **Quality Supervisor** — checks blur, lighting, glare, and obstructions.
2. **Diagnosis Agent** — predicts one of seven ISIC-style lesion classes when a
   compatible checkpoint is available.
3. **Lesion Analysis Agent** — segments the lesion and calculates
   asymmetry, border, colour, diameter, and ABCD-related features.
4. **Patient Context Agent** — incorporates optional age, sex, and anatomical
   localization metadata.
5. **Safety Agent** — applies conservative referral and veto guardrails.
6. **Coordinator Agent** — combines the evidence into a final triage decision.

The workflow can also generate a Grad-CAM overlay and a clinical explanation.
Quality failures short-circuit the expensive diagnostic path and produce a
`RE_IMAGE`-style outcome.

## Repository layout

```text
.
├── app.py                         # FastAPI application and demo dashboard
├── run_dermaguard.py              # Single-image CLI for the full graph
├── pipeline.py                    # Earlier single/batch pipeline
├── six_agent_pipeline.py          # Explicit six-agent evaluation pipeline
├── coordinator_agent/             # LangGraph orchestration and decisions
├── image_quality_agent/           # Blur, lighting, glare, obstruction checks
├── diagnosis_agent/               # Seven-class model inference
├── lesion_analysis_agent/         # Segmentation and ABCD morphometrics
├── reasoning_context_agent/       # Patient-context risk reasoning
├── safety_agent/                  # Conservative safety review and veto logic
├── explanation/                   # Gemini explanations and Grad-CAM
├── schema/                        # Shared typed pipeline state
├── results/                       # Example/evaluation outputs
└── API_DOCUMENTATION.md           # Endpoint details and response examples
```

## Requirements

- Python 3.10 or newer
- A compatible PyTorch installation for model inference
- Optional diagnosis and U-Net checkpoints (`.pth` or `.onnx`)
- Optional `GEMINI_API_KEY` for live Gemini explanations

The repository does not currently contain a root-level dependency lockfile.
Install the runtime dependencies in an environment appropriate for your
platform, for example:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install \
  fastapi uvicorn python-multipart pydantic \
  langgraph httpx pillow numpy opencv-python \
  torch torchvision pytest
```

The image-quality test package also documents its core requirements in
[`image_quality_agent/requirements.txt`](image_quality_agent/requirements.txt).

## Run the API

Start the service from the repository root:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- Interactive API docs: <http://127.0.0.1:8000/docs>
- ReDoc: <http://127.0.0.1:8000/redoc>
- Interactive demo dashboard: <http://127.0.0.1:8000/demo>
- Health check: <http://127.0.0.1:8000/health>

### Analyze an image

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@path/to/lesion.jpg" \
  -F "age=52" \
  -F "sex=female" \
  -F "localization=back" \
  -F "include_gradcam=true"
```

The response includes the final decision (`AUTO_TRIAGE`, `REFER`, or
`RE_IMAGE`), urgency, rationale, class probabilities, quality metrics,
morphology metrics, safety flags, and optional base64-encoded visualizations.

Other endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/classes` | Seven supported lesion-class definitions |
| `POST` | `/api/quality-check` | Fast technical image-quality screening |
| `POST` | `/api/gradcam` | Standalone Grad-CAM generation |
| `GET` | `/` | Service metadata and endpoint summary |

See [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) for request fields and
response examples.

## Run the CLI

```bash
python run_dermaguard.py \
  --image path/to/lesion.jpg \
  --id CASE_001 \
  --age 52 \
  --sex female \
  --site back
```

Use `--json` for machine-readable output:

```bash
python run_dermaguard.py --image path/to/lesion.jpg --json
```

Optional model and explanation configuration:

```bash
export GEMINI_API_KEY="your-api-key"
python run_dermaguard.py \
  --image path/to/lesion.jpg \
  --model /path/to/diagnosis_checkpoint.pth \
  --unet /path/to/segmentation_model.pth
```

If no diagnosis checkpoint is supplied, the coordinator uses its documented
heuristic baseline rather than failing to start. This is useful for exercising
the pipeline, but it is not a substitute for a trained diagnostic model.

## Test

Run the image-quality unit tests:

```bash
python -m pytest image_quality_agent/tests/test_quality.py -v
```

The tests use synthetic images and do not require a clinical dataset.

## Supported classes

The diagnosis response uses these seven labels:

| Label | Meaning |
|---|---|
| `nv` | Melanocytic nevus |
| `mel` | Melanoma |
| `bkl` | Benign keratosis |
| `bcc` | Basal cell carcinoma |
| `akiec` | Actinic keratosis / Bowen's disease |
| `vasc` | Vascular lesion |
| `df` | Dermatofibroma |

## Model and data assets

Large model weights and datasets are intentionally ignored by Git. The API
looks for a diagnosis checkpoint in these locations:

1. `checkpoints/efficientnet_b0_best.pth`
2. `model_extracted/efficientnet_b0.pth`
3. `~/dermaguard_checkpoints/efficientnet_b0_best.pth`

Pass custom paths with the CLI `--model` and `--unet` options. Do not commit
weights, datasets, API keys, or patient data to the repository.

## Development notes

- The API lazily initializes the model engine on first request.
- Uploaded files are written to a temporary directory and removed after each
  request.
- Gemini explanations fall back to a deterministic local explanation when
  `GEMINI_API_KEY` is absent or the network is unavailable.
- CORS is currently configured permissively for frontend integration; restrict
  `allow_origins` before deploying this service outside a trusted environment.

## License

No license file is currently included. Add an explicit license before
redistributing the project.
