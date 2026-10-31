"""
DermaGuard CLI & Unified Runner
===============================
Runs the full 6-agent LangGraph system on single images or batch datasets.

Usage:
  # Single image evaluation
  python run_dermaguard.py --image path/to/image.jpg --age 48 --sex male --site back

  # With Gemini API key for live explanations
  export GEMINI_API_KEY="your-api-key"
  python run_dermaguard.py --image path/to/image.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coordinator_agent.langgraph_coordinator import DermaGuardGraph


def main():
    parser = argparse.ArgumentParser(description="DermaGuard: 6-Agent Dermatological AI Triage System")
    parser.add_argument("--image", type=str, required=True, help="Path to input skin lesion image")
    parser.add_argument("--id", type=str, default="CASE_001", help="Image identifier")
    parser.add_argument("--age", type=float, default=None, help="Patient age in years")
    parser.add_argument("--sex", type=str, default=None, choices=["male", "female", "unknown"], help="Patient sex")
    parser.add_argument("--site", type=str, default=None, help="Anatomical site (e.g., face, back, chest, arm)")
    parser.add_argument("--model", type=str, default=None, help="Path to diagnosis CNN weights (.pth)")
    parser.add_argument("--unet", type=str, default=None, help="Path to U-Net segmentation weights (.pth / .onnx)")
    parser.add_argument("--key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize LangGraph system
    graph = DermaGuardGraph(
        model_path=args.model,
        unet_path=args.unet,
        gemini_api_key=args.key,
    )

    print(f"\nProcessing case: {args.id} ({image_path.name})...")
    result = graph.run(
        image_path=image_path,
        image_id=args.id,
        age=args.age,
        sex=args.sex,
        localization=args.site,
    )

    if args.json:
        # Filter non-serializable fields if any
        clean_out = {k: v for k, v in result.items() if not k.endswith("_report")}
        print(json.dumps(clean_out, indent=2))
        return

    # Print clinical triage summary
    print("\n" + "=" * 70)
    print("  DERMAGUARD MULTI-AGENT CLINICAL TRIAGE REPORT")
    print("=" * 70)
    print(f"  Case ID               : {result.get('image_id')}")
    print(f"  Technical Quality     : {result.get('quality_status')} (Score: {result.get('quality_score', 0.0):.2f})")
    if result.get("quality_failed_checks"):
        print(f"    - Failed Checks     : {', '.join(result.get('quality_failed_checks'))}")

    if result.get("quality_status") == "PASS":
        print(f"  Diagnostic Prediction : {result.get('predicted_label')} ({result.get('predicted_class')})")
        print(f"    - Model Confidence  : {result.get('diagnosis_confidence', 0.0):.1%}")
        if result.get("melanoma_flag"):
            print(f"    - Melanoma Risk Flag: ALERT (Melanoma prob: {result.get('melanoma_prob', 0.0):.1%})")

        print(f"  Visual ABCD Analysis  : Suspicion Level: {result.get('morphology_suspicion_level')} (Score: {result.get('morphology_suspicion_score', 0.0):.2f})")
        print(f"    - Asymmetry Index   : {result.get('asymmetry_index', 0.0):.3f}")
        print(f"    - Border Irregular  : {result.get('border_irregularity', 0.0):.3f}")
        print(f"    - Color Variegation : {result.get('color_variation', 0.0):.3f}")
        print(f"    - Equiv Diameter    : {result.get('equivalent_diameter_px', 0.0):.1f} px")
        print(f"    - Segmenter Backend : {result.get('segmentation_backend')}")

    print(f"  Patient Context Risk  : {result.get('patient_risk_indicator')} (Score: {result.get('patient_risk_score', 0.0):.2f})")
    print(f"  Safety Engine Severity: {result.get('safety_severity')} (Veto: {result.get('safety_veto')})")
    if result.get("safety_triggers"):
        print(f"    - Triggers          : {'; '.join(result.get('safety_triggers'))}")

    print("-" * 70)
    print(f"  FINAL DISPOSITION     : >>> {result.get('final_decision')} <<<")
    print(f"  Decision Rule Trigger : {result.get('decision_rule')}")
    print(f"  Confidence Tier       : {result.get('confidence_tier')}")
    print("=" * 70)

    print("\n[Clinical Explanation & Patient Communication]")
    print(result.get("clinical_explanation"))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
