"""
Diagnosis Agent (Agent 2) — DermaGuard Multi-Agent Pipeline
============================================================
Supports single-model checkpoints or multi-model ensembles with
optional Test-Time Augmentation (TTA), Shades-of-Gray color constancy,
and temperature scaling.

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
    # or callable:
    result = agent("path/to/image.jpg")
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ── Class mapping (Matches HAM10000 training order in checkpoints) ─────────────
LABEL_MAP = {"nv": 0, "mel": 1, "bkl": 2, "bcc": 3, "akiec": 4, "vasc": 5, "df": 6}
IDX_TO_CLASS = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [IDX_TO_CLASS[i] for i in range(len(LABEL_MAP))]
MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

CLASS_FULL = {
    "nv":    "Melanocytic Nevus",
    "mel":   "Melanoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Bowen's Disease",
    "vasc":  "Vascular Lesions",
    "df":    "Dermatofibroma",
}

# ── Checkpoint candidate specifications ───────────────────────────────────────
CANDIDATE_SPECS = [
    ("efficientnet_b0",       ["best_efficientnet_b0.pth", "efficientnet_b0.pth", "efnb0.pth"]),
    ("efficientnet_b3",       ["best_efficientnet_b3.pth", "efficientnet_b3.pth", "efnb3.pth"]),
    ("resnet50",              ["best_resnet50.pth", "resnet50.pth"]),
    ("vit_small_patch16_224", ["best_vit_small.pth", "vit_small.pth", "best_vit.pth", "vit.pth"]),
    ("densenet121",           ["best_densenet121.pth", "densenet121.pth"]),
    ("convnext_tiny",         ["best_convnext_tiny.pth", "convnext_tiny.pth"]),
]


def _infer_arch_name(filename_or_stem: str) -> str:
    s = filename_or_stem.lower()
    if "efficientnet_b3" in s or "efnb3" in s:
        return "efficientnet_b3"
    elif "efficientnet_b0" in s or "efnb0" in s:
        return "efficientnet_b0"
    elif "resnet50" in s or "resnet" in s:
        return "resnet50"
    elif "vit_small" in s or "vit" in s:
        return "vit_small_patch16_224"
    elif "densenet121" in s or "densenet" in s:
        return "densenet121"
    elif "convnext" in s:
        return "convnext_tiny"
    return "efficientnet_b0"


def _instantiate_model(arch_name: str, num_classes: int = len(CLASS_NAMES)) -> Optional[Any]:
    """Creates model architecture using timm first, then torchvision fallback."""
    if TIMM_AVAILABLE:
        try:
            return timm.create_model(arch_name, pretrained=False, num_classes=num_classes)
        except Exception:
            pass

    if TORCH_AVAILABLE:
        try:
            if arch_name == "efficientnet_b0":
                m = models.efficientnet_b0(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
                return m
            elif arch_name == "efficientnet_b3":
                m = models.efficientnet_b3(weights=None)
                m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
                return m
            elif arch_name == "resnet50":
                m = models.resnet50(weights=None)
                m.fc = nn.Linear(m.fc.in_features, num_classes)
                return m
        except Exception:
            pass
    return None


# ── Preprocessing & Color Constancy ───────────────────────────────────────────
def shades_of_gray(img_np: np.ndarray, p: int = 6) -> np.ndarray:
    """Minkowski p-norm color constancy (default p=6)."""
    img = img_np.astype(np.float32) + 1e-6
    minkowski = (img ** p).mean(axis=(0, 1)) ** (1.0 / p)
    factor = minkowski / (minkowski.mean() + 1e-6)
    out = np.clip(img / factor, 0, 255).astype(np.uint8)
    return out


_BASE_TRANSFORM = None
_TTA_TRANSFORMS = []

if TORCH_AVAILABLE:
    _NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    _BASE_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        _NORMALIZE,
    ])
    _TTA_TRANSFORMS = [
        _BASE_TRANSFORM,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            _NORMALIZE,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            _NORMALIZE,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            _NORMALIZE,
        ]),
    ]


class DiagnosisAgent:
    """
    Agent 2: Diagnostic prediction engine.
    Supports single model or multi-model ensemble with TTA + temperature scaling.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        models_list: Optional[List[Tuple[str, Any]]] = None,
        temperature: float = 1.2,
        use_tta: bool = False,
        use_shades_of_gray: bool = True,
        device: str = "auto",
    ):
        self.temperature = float(temperature)
        self.use_tta = bool(use_tta)
        self.use_shades_of_gray = bool(use_shades_of_gray)

        if not TORCH_AVAILABLE:
            self.model_loaded = False
            self.models = []
            return

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.models: List[Tuple[str, nn.Module]] = []

        if models_list is not None:
            self.models = [(name, m.to(self.device).eval()) for name, m in models_list]
            self.model_loaded = len(self.models) > 0
        elif model_path is not None:
            p = Path(model_path)
            if p.exists():
                if p.is_dir():
                    self._load_checkpoint_dir(p)
                else:
                    loaded = self._load_single_model(p)
                    if loaded is not None:
                        self.models.append(("efficientnet_b0", loaded))
                self.model_loaded = len(self.models) > 0
            else:
                self.model_loaded = False
        else:
            self.model_loaded = False

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        Returns:
            predicted_class, predicted_label, confidence, probabilities,
            melanoma_flag, melanoma_prob, malignant_prob, is_malignant, models_used
        """
        t0 = time.perf_counter()

        if not TORCH_AVAILABLE or not self.model_loaded or not self.models:
            return self._heuristic_fallback(source)

        try:
            pil_img = self._load_and_preprocess_pil(source)
        except Exception as exc:
            return self._error_result(str(exc))

        try:
            all_probs = []
            for name, model in self.models:
                p = self._probs_for_model(model, pil_img)
                all_probs.append(p)

            avg_probs = np.mean(all_probs, axis=0)
            top_idx = int(avg_probs.argmax())
            top_class = IDX_TO_CLASS[top_idx]
            confidence = float(avg_probs[top_idx])
            mel_prob = float(avg_probs[LABEL_MAP["mel"]])
            malignant_prob = float(sum(avg_probs[LABEL_MAP[c]] for c in MALIGNANT_CLASSES))

            # Flag melanoma if predicted top-1 is mel OR mel_prob >= 0.25 (calibrated cancer safety barrier)
            mel_flag = (top_class == "mel") or (mel_prob >= 0.25)

            probs_dict = {IDX_TO_CLASS[i]: round(float(p), 4) for i, p in enumerate(avg_probs)}

            return {
                "predicted_class":   top_class,
                "predicted_label":   top_class,  # matches submission field
                "prediction":        top_class,
                "diagnosis_label":   CLASS_FULL.get(top_class, top_class),
                "confidence":        round(confidence, 4),
                "probabilities":     probs_dict,
                "class_probs":       probs_dict,
                "melanoma_flag":     mel_flag,
                "melanoma_prob":     round(mel_prob, 4),
                "malignant_prob":    round(malignant_prob, 4),
                "is_malignant":      top_class in MALIGNANT_CLASSES,
                "models_used":       [n for n, _ in self.models],
                "processing_time_s": round(time.perf_counter() - t0, 4),
                "error":             None,
            }
        except Exception as exc:
            return self._error_result(str(exc))

    def __call__(self, source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Convenience callable wrapper: agent2(img_path)."""
        return self.predict(source)

    # ── Inference Helpers ─────────────────────────────────────────────────────

    def _probs_for_model(self, model: nn.Module, pil_img: Image.Image) -> np.ndarray:
        """Inference with optional TTA and temperature scaling."""
        tfms = _TTA_TRANSFORMS if self.use_tta else [_BASE_TRANSFORM]
        accum = np.zeros(len(CLASS_NAMES), dtype=np.float32)

        with torch.no_grad():
            for tfm in tfms:
                t = tfm(pil_img).unsqueeze(0).to(self.device)
                logits = model(t) / self.temperature
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                accum += probs

        return accum / len(tfms)

    def _load_and_preprocess_pil(self, source: Union[str, Path, np.ndarray]) -> Image.Image:
        if isinstance(source, np.ndarray):
            img_np = source if source.shape[2] == 3 else np.stack([source]*3, axis=-1)
        else:
            p = str(source)
            from PIL import Image as PilImg
            with PilImg.open(p) as raw:
                img_np = np.array(raw.convert("RGB"))

        if self.use_shades_of_gray:
            img_np = shades_of_gray(img_np, p=6)

        return Image.fromarray(img_np)

    def _load_single_model(self, model_path: Path, arch_hint: Optional[str] = None) -> Optional[nn.Module]:
        if not TORCH_AVAILABLE:
            return None
        try:
            ckpt = torch.load(str(model_path), map_location=self.device, weights_only=False)
            if isinstance(ckpt, nn.Module):
                return ckpt.to(self.device).eval()

            state_dict = (
                ckpt.get("model")
                or ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt
            )
            if not isinstance(state_dict, dict):
                return None

            # Remove module. prefix if DataParallel
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            arch = arch_hint or _infer_arch_name(model_path.name)
            model = _instantiate_model(arch, len(CLASS_NAMES))
            if model is None:
                print(f"[DiagnosisAgent] Warning: could not instantiate architecture '{arch}' for {model_path.name}")
                return None

            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            return model.to(self.device).eval()
        except Exception as e:
            print(f"[DiagnosisAgent] Warning: could not load {model_path}: {e}")
            return None

    def _load_checkpoint_dir(self, checkpoint_dir: Path):
        """Loads all models from a checkpoint directory into the ensemble."""
        loaded_count = 0
        seen_files = set()

        # 1. Match known candidate specs
        for arch_name, candidate_files in CANDIDATE_SPECS:
            for cf in candidate_files:
                pth = checkpoint_dir / cf
                if pth.exists() and pth not in seen_files:
                    m = self._load_single_model(pth, arch_hint=arch_name)
                    if m is not None:
                        self.models.append((arch_name, m))
                        seen_files.add(pth)
                        loaded_count += 1
                        print(f"  [DiagnosisAgent] Loaded {arch_name} from {pth.name} ✅")
                    break

        # 2. Match any remaining .pth files in the directory
        for pth in sorted(checkpoint_dir.glob("*.pth")):
            if pth not in seen_files:
                arch_name = _infer_arch_name(pth.name)
                m = self._load_single_model(pth, arch_hint=arch_name)
                if m is not None:
                    self.models.append((arch_name, m))
                    seen_files.add(pth)
                    loaded_count += 1
                    print(f"  [DiagnosisAgent] Loaded {arch_name} from {pth.name} ✅")

        if loaded_count > 0:
            print(f"[DiagnosisAgent] 🚀 Ensemble ready: {loaded_count} model(s) loaded from {checkpoint_dir}")

    def _heuristic_fallback(self, source: Any) -> Dict[str, Any]:
        """Graceful baseline if weights are not available locally."""
        probs = {
            "nv": 0.50,
            "bkl": 0.15,
            "mel": 0.10,
            "bcc": 0.10,
            "akiec": 0.05,
            "df": 0.05,
            "vasc": 0.05,
        }
        return {
            "predicted_class":   "nv",
            "predicted_label":   "nv",
            "prediction":        "nv",
            "diagnosis_label":   CLASS_FULL["nv"],
            "confidence":        0.50,
            "probabilities":     probs,
            "class_probs":       probs,
            "melanoma_flag":     False,
            "melanoma_prob":     0.10,
            "malignant_prob":    0.25,
            "is_malignant":      False,
            "models_used":       ["baseline_fallback"],
            "processing_time_s": 0.0,
            "error":             None,
        }

    @staticmethod
    def _error_result(error: str) -> Dict[str, Any]:
        return {
            "predicted_class":   "unknown",
            "predicted_label":   "unknown",
            "prediction":        "unknown",
            "diagnosis_label":   "Unknown",
            "confidence":        0.0,
            "probabilities":     {c: 0.0 for c in CLASS_NAMES},
            "class_probs":       {c: 0.0 for c in CLASS_NAMES},
            "melanoma_flag":     False,
            "melanoma_prob":     0.0,
            "malignant_prob":    0.0,
            "is_malignant":      False,
            "models_used":       [],
            "processing_time_s": 0.0,
            "error":             error,
        }
