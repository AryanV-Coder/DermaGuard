#!/usr/bin/env python3
"""
DermaGuard — Image Quality Agent
CLI entry point for single-image and batch evaluation.

Usage
-----
Single image:
    python main.py path/to/image.jpg

Single image with visualisation overlay:
    python main.py path/to/image.jpg --visualize

Batch evaluation of a directory:
    python main.py --batch path/to/image_folder/
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running from the project root (python main.py) or from inside
# image_quality_agent/ (python -m main).
sys.path.insert(0, str(Path(__file__).parent))

from supervisor.quality_supervisor import QualitySupervisor


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_single(path: str, visualize: bool = False) -> dict:
    supervisor = QualitySupervisor()
    report = supervisor.evaluate(path)
    print(json.dumps(report, indent=2))

    if visualize:
        _show_annotated(path, report)

    return report


def evaluate_batch(directory: str) -> None:
    _SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    root  = Path(directory)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _SUPPORTED_EXTS)

    if not paths:
        print(f"[ERROR] No supported images found in: {directory}", file=sys.stderr)
        sys.exit(1)

    supervisor = QualitySupervisor()
    results    = []
    counts     = {"PASS": 0, "FAIL": 0, "ERROR": 0}

    for idx, path in enumerate(paths, 1):
        report = supervisor.evaluate(str(path))
        status = report["overall"]["status"]
        score  = report["overall"]["score"]
        counts[status] = counts.get(status, 0) + 1

        flag = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "!")
        print(f"[{idx:4d}/{len(paths)}] {flag}  {path.name:<45s}  {status:<5s}  score={score:.3f}")

        results.append({
            "file":   str(path),
            "status": status,
            "score":  score,
            "recommendation": report["overall"]["recommendation"],
        })

    total = len(paths)
    print("\n── Batch Summary ─────────────────────────────────")
    print(f"   Total  : {total}")
    print(f"   PASS   : {counts.get('PASS',  0)}  ({counts.get('PASS',  0) / total * 100:.1f}%)")
    print(f"   FAIL   : {counts.get('FAIL',  0)}  ({counts.get('FAIL',  0) / total * 100:.1f}%)")
    print(f"   ERROR  : {counts.get('ERROR', 0)}")

    out_path = root / "quality_report.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n   Full report saved → {out_path}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def _show_annotated(path: str, report: dict) -> None:
    try:
        from utils.visualization     import draw_quality_report
        from preprocessing.image_processor import ImageProcessor

        img_data = ImageProcessor().load(path)
        if not img_data["valid"]:
            print("[WARN] Could not reload image for visualisation.", file=sys.stderr)
            return

        annotated = draw_quality_report(img_data["rgb"], report)
        cv2.imshow("DermaGuard — Quality Report", annotated)
        print("Press any key to close the window …")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as exc:
        print(f"[WARN] Visualisation failed: {exc}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="image_quality_agent",
        description="DermaGuard — Image Quality Agent\n"
                    "Evaluates dermoscopic image quality without any model or LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "image",
        nargs="?",
        help="Path to a single image file.",
    )
    p.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Display annotated result image (requires a display).",
    )
    p.add_argument(
        "--batch", "-b",
        metavar="DIR",
        help="Evaluate all images in DIR and write quality_report.json.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.batch:
        evaluate_batch(args.batch)
    elif args.image:
        evaluate_single(args.image, visualize=args.visualize)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
