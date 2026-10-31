"""
DermaGuard — Full Two-Agent Pipeline Runner
============================================
Connects Agent 1 (Image Quality) and Agent 2 (Diagnosis).

Workflow:
  1. For each validation image:
     - Run Agent 1 (QualitySupervisor) assessing blur, lighting, glare, obstruction
     - Report quality status (PASS / FAIL / ERROR), score, and reasons
  2. If Agent 1 PASSES the image:
     - Pass to Agent 2 (DiagnosisAgent) for 7-class lesion prediction
     - Compare predicted class with ground truth validation label
  3. If Agent 1 FAILS the image:
     - Flag as RE_IMAGE (rejected due to quality)
  4. Also compute baseline metrics on all images and accepted subset
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "image_quality_agent"))
sys.path.insert(0, str(ROOT))

from image_quality_agent.supervisor.quality_supervisor import QualitySupervisor
from diagnosis_agent.diagnosis_agent import DiagnosisAgent, CLASS_NAMES


def build_image_lookup(image_dirs) -> Dict[str, str]:
    """Index all images once into a dictionary for O(1) file access."""
    lookup = {}
    for d in image_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for f in p.glob(ext):
                lookup[f.stem] = str(f)
    return lookup


def run_pipeline(
    val_csv: Path,
    image_dir: Path,
    model_path: Path,
    out_csv: Path,
    limit: Optional[int] = None,
) -> dict:
    print(f"Initializing agents and indexing images...")
    
    # Check possible image folders
    p1 = image_dir / "HAM10000_images_part_1"
    p2 = image_dir / "HAM10000_images_part_2"
    dirs_to_search = [p1, p2, image_dir]
    lookup = build_image_lookup(dirs_to_search)
    print(f"Indexed {len(lookup)} images.")

    # Load Agents
    q_agent = QualitySupervisor()
    d_agent = DiagnosisAgent(model_path=str(model_path))

    # Read validation CSV
    with open(val_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    total = len(rows)
    print(f"\n{'='*70}")
    print(f"  DermaGuard Pipeline  |  Evaluating {total} Validation Images")
    print(f"{'='*70}")

    counts = {
        "quality_pass": 0,
        "quality_fail": 0,
        "quality_error": 0,
        "diag_correct": 0,
        "diag_incorrect": 0,
        "diag_skipped": 0,
    }

    results = []
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id", "true_label",
        "quality_status", "quality_score", "quality_reason",
        "decision",             # AUTO_TRIAGE | RE_IMAGE
        "predicted_class", "confidence",
        "is_correct",
        "p_akiec", "p_bcc", "p_bkl", "p_df", "p_mel", "p_nv", "p_vasc"
    ]

    t_start_all = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            image_id = row.get("image_id", "").strip()
            true_label = row.get("dx", "").strip()

            img_path = lookup.get(image_id)
            if not img_path:
                counts["quality_error"] += 1
                counts["diag_skipped"] += 1
                rec = {
                    "image_id": image_id, "true_label": true_label,
                    "quality_status": "ERROR", "quality_score": 0.0,
                    "quality_reason": "Image not found on disk",
                    "decision": "RE_IMAGE",
                    "predicted_class": None, "confidence": 0.0,
                    "is_correct": False,
                    **{f"p_{c}": 0.0 for c in CLASS_NAMES}
                }
                results.append(rec)
                writer.writerow(rec)
                continue

            # 1. Agent 1: Image Quality Assessment
            q_report = q_agent.evaluate(img_path)
            q_status = q_report["overall"]["status"]
            q_score = q_report["overall"]["score"]
            q_reason = q_report["overall"]["recommendation"]

            if q_status == "PASS":
                counts["quality_pass"] += 1
                # 2. Agent 2: Diagnosis Model
                d_result = d_agent.predict(img_path)
                pred_class = d_result["predicted_class"]
                confidence = d_result["confidence"]
                probs = d_result["probabilities"]
                is_correct = (pred_class == true_label)
                decision = "AUTO_TRIAGE"

                if is_correct:
                    counts["diag_correct"] += 1
                else:
                    counts["diag_incorrect"] += 1
            else:
                counts["quality_fail"] += 1
                pred_class = None
                confidence = 0.0
                probs = {c: 0.0 for c in CLASS_NAMES}
                is_correct = False
                decision = "RE_IMAGE"
                counts["diag_skipped"] += 1

            rec = {
                "image_id": image_id,
                "true_label": true_label,
                "quality_status": q_status,
                "quality_score": round(q_score, 4),
                "quality_reason": q_reason,
                "decision": decision,
                "predicted_class": pred_class,
                "confidence": round(confidence, 4) if confidence else 0.0,
                "is_correct": is_correct,
                **{f"p_{c}": round(probs.get(c, 0.0), 6) for c in CLASS_NAMES}
            }
            results.append(rec)
            writer.writerow(rec)

            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - t_start_all
                speed = idx / elapsed
                print(f"[{idx:4d}/{total}] {idx/total*100:5.1f}% | PASS: {counts['quality_pass']} | FAIL: {counts['quality_fail']} | Correct: {counts['diag_correct']} | Incorrect: {counts['diag_incorrect']} | Speed: {speed:.1f} img/s")

    diagnosed = counts["diag_correct"] + counts["diag_incorrect"]
    accuracy = (counts["diag_correct"] / diagnosed * 100) if diagnosed > 0 else 0.0

    summary = {
        "total_images": total,
        "quality_pass": counts["quality_pass"],
        "quality_fail": counts["quality_fail"],
        "quality_error": counts["quality_error"],
        "diagnosed_images": diagnosed,
        "correct_predictions": counts["diag_correct"],
        "incorrect_predictions": counts["diag_incorrect"],
        "skipped_re_image": counts["diag_skipped"],
        "accuracy_on_accepted": round(accuracy, 2),
    }

    print(f"\n{'='*70}")
    print(f"  DERMAGUARD EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Validation Images Evaluated : {total}")
    print(f"  Agent 1 Quality PASS (Accepted)   : {counts['quality_pass']} ({counts['quality_pass']/total*100:.2f}%)")
    print(f"  Agent 1 Quality FAIL (Rejected)   : {counts['quality_fail']} ({counts['quality_fail']/total*100:.2f}%)")
    print(f"  Agent 1 Errors                    : {counts['quality_error']}")
    print(f"  ----------------------------------------------------------------------")
    print(f"  Images Forwarded to Agent 2       : {diagnosed}")
    print(f"  Correct Diagnosis Predictions     : {counts['diag_correct']} ({counts['diag_correct']/diagnosed*100:.2f}% of accepted)")
    print(f"  Incorrect Diagnosis Predictions   : {counts['diag_incorrect']} ({counts['diag_incorrect']/diagnosed*100:.2f}% of accepted)")
    print(f"  Agent 2 Accuracy on Accepted      : {accuracy:.2f}%")
    print(f"{'='*70}")

    summary_file = out_csv.with_suffix(".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full predictions to: {out_csv}")
    print(f"Saved summary metrics to:  {summary_file}\n")

    # Per-class breakdown
    print_per_class(results)
    return summary


def print_per_class(results):
    from collections import defaultdict
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["decision"] == "AUTO_TRIAGE":
            dx = r["true_label"]
            stats[dx]["total"] += 1
            if r["is_correct"]:
                stats[dx]["correct"] += 1

    print("Per-class performance on Accepted (Agent 1 PASS) images:")
    print(f"{'Class':<8} {'Correct':>10} {'Total':>10} {'Accuracy (%)':>15}")
    print("-" * 47)
    for c in CLASS_NAMES:
        item = stats[c]
        tot = item["total"]
        cor = item["correct"]
        acc = (cor / tot * 100) if tot > 0 else 0.0
        print(f"{c:<8} {cor:>10} {tot:>10} {acc:>15.2f}%")
    print("-" * 47)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", default=r"c:\Users\goelt\Downloads\image_quality_agent\val_split.csv")
    parser.add_argument("--image-dir", default=r"C:\Users\goelt\.cache\kagglehub\datasets\kmader\skin-cancer-mnist-ham10000\versions\2")
    parser.add_argument("--model", default=r"c:\Users\goelt\Downloads\image_quality_agent\diagnosis_model.pth.zip")
    parser.add_argument("--out", default=r"c:\Users\goelt\Downloads\image_quality_agent\results\pipeline_results.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        val_csv=Path(args.val_csv),
        image_dir=Path(args.image_dir),
        model_path=Path(args.model),
        out_csv=Path(args.out),
        limit=args.limit,
    )
