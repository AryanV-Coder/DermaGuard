"""
Utility: Extract diagnosis_model.pth from the zip and test loading.
Run this once after the zip is in place.
"""
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def extract_model(zip_path: Path, out_dir: Path) -> Path:
    """Extract diagnosis_model.pth.zip → returns path to .pth file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pth_out = out_dir / "diagnosis_model.pth"

    if pth_out.exists():
        print(f"[extract] Already extracted: {pth_out}")
        return pth_out

    print(f"[extract] Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    # The zip contains a folder 'diagnosis_model/' (PyTorch 2.x format)
    extracted_dir = out_dir / "diagnosis_model"
    if extracted_dir.is_dir():
        print(f"[extract] Found PyTorch 2.x save format at {extracted_dir}")
        # torch.load can load directly from the directory path
        return extracted_dir
    elif pth_out.exists():
        return pth_out
    else:
        # Find any .pth file
        hits = list(out_dir.rglob("*.pth"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Could not find .pth after extraction in {out_dir}")


if __name__ == "__main__":
    zip_path = ROOT / "diagnosis_model.pth.zip"
    out_dir  = ROOT / "model_extracted"

    model_path = extract_model(zip_path, out_dir)
    print(f"[extract] Model path: {model_path}")

    # Quick load test
    sys.path.insert(0, str(ROOT))
    try:
        from diagnosis_agent.diagnosis_agent import DiagnosisAgent
        agent = DiagnosisAgent(model_path=str(model_path))
        print("[test] DiagnosisAgent loaded successfully!")
        print(f"[test] Device: {agent.device}")
    except Exception as e:
        print(f"[test] FAILED: {e}")
        sys.exit(1)

