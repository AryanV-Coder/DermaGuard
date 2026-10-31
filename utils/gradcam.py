"""
Explainability Module: Grad-CAM for Diagnosis & Quality Inspection
==================================================================
Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps
for visual explainability on skin lesion images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class GradCAM:
    """
    Computes Grad-CAM heatmaps for an EfficientNet or CNN model.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # If target layer is not specified, pick the last Conv2d in model.features
        if target_layer is None:
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            self.target_layer = last_conv
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self._hook_handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        img_source: Union[str, Path, np.ndarray],
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate a Grad-CAM heatmap and blended visualization.

        Returns:
            heatmap (224, 224) float 0..1
            cam_overlay_bgr (H, W, 3) uint8 image with jet heatmap overlay
            predicted_idx (int)
        """
        if isinstance(img_source, np.ndarray):
            orig_bgr = img_source.copy()
            pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        else:
            orig_bgr = cv2.imread(str(img_source))
            pil_img = Image.open(str(img_source)).convert("RGB")

        orig_h, orig_w = orig_bgr.shape[:2]

        input_tensor = _TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 224, 224)
        input_tensor.requires_grad = True

        # Forward pass
        logits = self.model(input_tensor)
        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        # Backward pass for the target class
        self.model.zero_grad()
        score = logits[0, target_class_idx]
        score.backward()

        # Pool gradients across channels
        # gradients: (1, C, H, W), activations: (1, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # Apply ReLU to keep only positive influence

        cam_np = cam.squeeze().cpu().numpy()
        cam_np = cv2.resize(cam_np, (orig_w, orig_h))
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Create colored overlay
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        return cam_norm, overlay, target_class_idx

    def save_cam(
        self,
        img_source: Union[str, Path, np.ndarray],
        out_path: Union[str, Path],
        target_class_idx: Optional[int] = None,
    ) -> Path:
        """Convenience method to save CAM heatmap image to disk."""
        _, overlay, _ = self.generate(img_source, target_class_idx)
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
        return out_p

