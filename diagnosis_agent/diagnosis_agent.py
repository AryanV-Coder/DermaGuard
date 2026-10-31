"""
Diagnosis Agent (Agent 2) — DermaGuard Multi-Agent Pipeline
============================================================
Loads the trained PyTorch model (diagnosis_model.pth) and predicts
one of seven HAM10000 skin-lesion classes for a given image.

Classes (index → abbreviation → full name):
  0 akiec  Actinic keratoses / Bowen's disease
  1 bcc    Basal cell carcinoma
  2 bkl    Benign keratosis-like lesions
  3 df     Dermatofibroma
  4 mel    Melanoma
  5 nv     Melanocytic nevi
  6 vasc   Vascular lesions

Usage
-----
    from diagnosis_agent.diagnosis_agent import DiagnosisAgent
    agent = DiagnosisAgent(model_path="diagnosis_model.pth")
    result = agent.predict("path/to/image.jpg")
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Class mapping ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL = {
    "akiec": "Actinic Keratoses / Bowen's Disease",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# ── Image preprocessing (must match training) ──────────────────────────────────
_TRANSFORM = None
if TORCH_AVAILABLE:
    _TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats (standard for fine-tuning)
            std =[0.229, 0.224, 0.225],
        ),
    ])


class DiagnosisAgent:
    """
    Wraps the trained CNN to provide per-image lesion-class predictions.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model file (.pth).  Can be a plain state-dict
        or a full torch.save checkpoint.
    device : str
        'cpu', 'cuda', or 'auto' (picks CUDA if available).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed.  Run: pip install torch torchvision"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # ── Device ─────────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[DiagnosisAgent] Loading model from {self.model_path} on {self.device} …")
        self.model = self._load_model()
        self.model.eval()
        print(f"[DiagnosisAgent] Model ready. Classes: {CLASS_NAMES}")

    # ── Public API ──────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Parameters
        ----------
        source : str | Path | np.ndarray
            File path to a JPEG/PNG image, or a pre-loaded numpy RGB array.

        Returns
        -------
        {
            predicted_class  : str    e.g. "mel"
            predicted_label  : str    e.g. "Melanoma"
            confidence       : float  0.0–1.0
            probabilities    : {class_name: prob, …}
            processing_time_s: float
            error            : str | None
        }
        """
        t0 = time.perf_counter()
        try:
            tensor = self._preprocess(source)
        except Exception as exc:
            return self._error_result(str(exc))

        with torch.no_grad():
            logits = self.model(tensor.to(self.device))   # (1, 7)
            probs  = torch.softmax(logits, dim=1)[0]      # (7,)

        probs_np   = probs.cpu().numpy()
        pred_idx   = int(probs_np.argmax())
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs_np[pred_idx])

        return {
            "predicted_class":   pred_class,
            "predicted_label":   CLASS_FULL[pred_class],
            "confidence":        round(confidence, 6),
            "probabilities":     {c: round(float(p), 6) for c, p in zip(CLASS_NAMES, probs_np)},
            "processing_time_s": round(time.perf_counter() - t0, 4),
            "error":             None,
        }

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _load_model(self) -> nn.Module:
        """
        Build an EfficientNet-B0 (same as training) and load weights.
        Supports checkpoint format: {'model': <state_dict>, 'classes': [...]}
        as well as bare state-dicts and full model saves.
        """
        checkpoint = torch.load(
            str(self.model_path), map_location=self.device, weights_only=False
        )

        # ── Full model object ─────────────────────────────────────────────────
        if not isinstance(checkpoint, dict):
            return checkpoint.to(self.device)

        # ── Dict with 'model' + optional 'classes' keys ───────────────────────
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            # Respect class list saved alongside the weights
            saved_classes = checkpoint.get("classes", CLASS_NAMES)
            n_classes = len(saved_classes)
        else:
            # Bare state dict or other key names
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
            n_classes = len(CLASS_NAMES)

        # Build backbone with correct number of output classes
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)

        # Strip 'module.' prefix (DataParallel saves)
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        return model.to(self.device)

    @staticmethod
    def _preprocess(source: Union[str, Path, np.ndarray]) -> "torch.Tensor":
        """Convert image to a normalised (1, 3, 224, 224) tensor."""
        if isinstance(source, np.ndarray):
            # Assume RGB numpy array (H, W, 3)
            pil_img = Image.fromarray(source.astype(np.uint8))
        else:
            pil_img = Image.open(str(source)).convert("RGB")

        tensor = _TRANSFORM(pil_img)    # (3, 224, 224)
        return tensor.unsqueeze(0)      # (1, 3, 224, 224)

    @staticmethod
    def _error_result(error: str) -> Dict[str, Any]:
        return {
            "predicted_class":   None,
            "predicted_label":   None,
            "confidence":        0.0,
            "probabilities":     {},
            "processing_time_s": 0.0,
            "error":             error,
        }

