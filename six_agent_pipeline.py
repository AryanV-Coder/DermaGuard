"""
DermaGuard — Level 2: Full Six-Agent System Evaluator & CLI
============================================================
Runs all 6 agents collaboratively on skin-lesion images:
  Agent 1: Quality Supervisor (Blur, Glare, Lighting, Obstruction)
  Agent 2: Diagnosis Agent (EfficientNet-B0 HAM10000 7-class prediction)
  Agent 3: Lesion Analysis Agent (ABCD Morphometrics: Asymmetry, Border, Color)
  Agent 4: Reasoning / Context Agent (Patient metadata: Age, Sex, Site)
  Agent 5: Safety Agent (Multi-agent conflict checking, Safety Flags & Veto)
  Agent 6: Coordinator Agent (Final Disposition: AUTO_TRIAGE / REFER / RE_IMAGE)

Meets Track 2 Day 2 Submission Specs:
  - Output CSV format: image_id, prediction, decision
  - Supports batch validation evaluation & single image inspection
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "image_quality_agent"))
sys.path.insert(0, str(ROOT))

from image_quality_agent.supervisor.quality_supervisor import QualitySupervisor
from diagnosis_agent.diagnosis_agent import DiagnosisAgent, CLASS_NAMES
from lesion_analysis_agent.lesion_analysis_agent import LesionAnalysisAgent
from reasoning_context_agent.reasoning_context_agent import ReasoningContextAgent
from safety_agent.safety_agent import SafetyAgent
from coordinator_agent.coordinator_agent import CoordinatorAgent


def build_image_lookup(image_dirs: List[Path]) -> Dict[str, str]:
    """Index all images into a dictionary for O(1) resolution."""
    lookup = {}
    for d in image_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in p.glob(ext):
                lookup[f.stem] = str(f)
    return lookup


def run_six_agent_pipeline(
    val_csv: Path,
    image_dir: Path,
    model_path: Path,
    out_csv: Path,
    api_submission_csv: Optional[Path] = None,
    limit: Optional[int] = None,
) -> dict:
    print("\n[Initializing Six-Agent System...]")
    
    # 1. Index images
    p1 = image_dir / "HAM10000_images_part_1"
    p2 = image_dir / "HAM10000_images_part_2"
    lookup = build_image_lookup([p1, p2, image_dir])
    print(f"  Indexed {len(lookup)} image files.")

    # 2. Instantiate all 6 Agents
    a1 = QualitySupervisor()
    a2 = DiagnosisAgent(model_path=str(model_path))
    a3 = LesionAnalysisAgent()
    a4 = ReasoningContextAgent()
    a5 = SafetyAgent()
    a6 = CoordinatorAgent()
    print("  All 6 agents loaded and ready.\n")

    # 3. Read dataset metadata
    with open(val_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    print("=" * 80)
    print(f"  DermaGuard Six-Agent System  |  Evaluating {total} Validation Cases")
    print("=" * 80)

    stats = {
        "total": total,
        "AUTO_TRIAGE": 0,
        "REFER": 0,
        "RE_IMAGE": 0,
        "auto_triage_correct": 0,
        "auto_triage_incorrect": 0,
        "refer_malignant_caught": 0,
        "refer_benign": 0,
    }

    full_results = []
    submission_rows = []

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if api_submission_csv:
        api_submission_csv.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for idx, row in enumerate(rows, 1):
        image_id = row.get("image_id", "").strip()
        true_dx = row.get("dx", "").strip()
        age = float(row["age"]) if row.get("age") and row["age"].strip() else None
        sex = row.get("sex")
        loc = row.get("localization")

        img_path = lookup.get(image_id)

        # Agent 1: Image Quality
        if not img_path:
            r1 = {"overall": {"status": "ERROR", "score": 0.0, "recommendation": "Missing image file"}}
        else:
            r1 = a1.evaluate(img_path)

        q_status = r1["overall"]["status"]

        # Run diagnosis & visual analysis if quality PASS, else fallback
        if q_status == "PASS" and img_path:
            r2 = a2.predict(img_path)
            r3 = a3.analyze(img_path)
        else:
            r2 = {"predicted_class": "nv", "confidence": 0.0, "probabilities": {}}
            r3 = {"suspicion_score": 0.0, "suspicion_level": "UNKNOWN", "reliability_flags": ["Quality check failed"]}

        # Agent 4: Patient Context
        r4 = a4.evaluate(age=age, sex=sex, localization=loc)

        # Agent 5: Safety Review
        r5 = a5.review(r1, r2, r3, r4)

        # Agent 6: Coordinator Decision
        r6 = a6.decide(image_id, r1, r2, r3, r4, r5)

        dec = r6["decision"]
        pred = r6["prediction"]
        stats[dec] += 1

        is_match = (pred == true_dx)
        if dec == "AUTO_TRIAGE":
            if is_match:
                stats["auto_triage_correct"] += 1
            else:
                stats["auto_triage_incorrect"] += 1
        elif dec == "REFER":
            if true_dx in {"mel", "bcc", "akiec"}:
                stats["refer_malignant_caught"] += 1
            else:
                stats["refer_benign"] += 1

        submission_rows.append({
            "image_id": image_id,
            "prediction": pred,
            "decision": dec,
        })

        full_results.append({
            "image_id": image_id,
            "true_dx": true_dx,
            "prediction": pred,
            "decision": dec,
            "decision_rule": r6["decision_rule"],
            "confidence_tier": r6["confidence_tier"],
            "a1_quality_status": q_status,
            "a1_quality_score": r1["overall"]["score"],
            "a2_confidence": r2["confidence"],
            "a3_suspicion_level": r3["suspicion_level"],
            "a3_suspicion_score": r3["suspicion_score"],
            "a4_risk_indicator": r4["risk_indicator"],
            "a4_risk_score": r4["risk_score"],
            "a5_severity": r5["severity"],
            "a5_veto": r5["veto"],
            "a5_triggers": "; ".join(r5["triggers"]),
        })

        if idx % 100 == 0 or idx == total:
            elapsed = time.time() - t_start
            speed = idx / elapsed
            print(f"[{idx:4d}/{total}] {idx/total*100:5.1f}% | AUTO: {stats['AUTO_TRIAGE']} | REFER: {stats['REFER']} | RE_IMAGE: {stats['RE_IMAGE']} | Speed: {speed:.1f} img/s")

    # Write detailed CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_results[0].keys()))
        writer.writeheader()
        writer.writerows(full_results)

    # Write official submission format CSV: image_id, prediction, decision
    if api_submission_csv:
        with open(api_submission_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "prediction", "decision"])
            writer.writeheader()
            writer.writerows(submission_rows)

    # Print summary
    print("\n" + "=" * 80)
    print("  SIX-AGENT SYSTEM SUMMARY DISPOSITION")
    print("=" * 80)
    print(f"  Total Images Processed            : {total}")
    print(f"  RE_IMAGE (Quality rejected)       : {stats['RE_IMAGE']} ({stats['RE_IMAGE']/total*100:.2f}%)")
    print(f"  REFER (Escalated to Specialist)   : {stats['REFER']} ({stats['REFER']/total*100:.2f}%)")
    print(f"    - Malignant lesions caught      : {stats['refer_malignant_caught']}")
    print(f"    - Benign / complex referred     : {stats['refer_benign']}")
    print(f"  AUTO_TRIAGE (Automated clearance) : {stats['AUTO_TRIAGE']} ({stats['AUTO_TRIAGE']/total*100:.2f}%)")
    print(f"    - Correct Auto-Triage           : {stats['auto_triage_correct']}")
    print(f"    - Incorrect Auto-Triage         : {stats['auto_triage_incorrect']}")
    auto_acc = (stats['auto_triage_correct'] / stats['AUTO_TRIAGE'] * 100) if stats['AUTO_TRIAGE'] > 0 else 0.0
    print(f"    - Auto-Triage Accuracy          : {auto_acc:.2f}%")
    print("=" * 80)
    print(f"  Detailed output saved to : {out_csv}")
    if api_submission_csv:
        print(f"  API Submission CSV saved : {api_submission_csv}")
    print("=" * 80 + "\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", default=r"c:\Users\goelt\Downloads\image_quality_agent\val_split.csv")
    parser.add_argument("--image-dir", default=r"C:\Users\goelt\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2")
    parser.add_argument("--model", default=r"c:\Users\goelt\Downloads\image_quality_agent\diagnosis_model.pth.zip")
    parser.add_argument("--out", default=r"c:\Users\goelt\Downloads\image_quality_agent\results\six_agent_evaluation.csv")
    parser.add_argument("--submission-csv", default=r"c:\Users\goelt\Downloads\image_quality_agent\results\final_submission.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_six_agent_pipeline(
        val_csv=Path(args.val_csv),
        image_dir=Path(args.image_dir),
        model_path=Path(args.model),
        out_csv=Path(args.out),
        api_submission_csv=Path(args.submission_csv),
        limit=args.limit,
    )

