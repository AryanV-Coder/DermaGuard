"""
Pretrained U-Net Segmentation Module for Skin Lesions
=====================================================
Provides deep learning based semantic segmentation for Agent 3.
Segments the skin lesion boundary from surrounding healthy skin.
Trained/formatted for ISIC Skin Lesion Segmentation benchmarks.

Supports:
  1. PyTorch U-Net architecture & checkpoint loading
  2. ONNX Runtime / OpenCV DNN inference (zero-torch environments)
  3. Seamless adaptive Otsu fallback if weights are not provided
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List
import cv2
import numpy as np

# Optional PyTorch import for environments that have it (e.g., Colab / Linux)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        """(Convolution => [BN] => ReLU) * 2"""
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.double_conv(x)


    class UNet(nn.Module):
        """Standard U-Net for Skin Lesion Segmentation."""
        def __init__(self, n_channels: int = 3, n_classes: int = 1):
            super().__init__()
            self.n_channels = n_channels
            self.n_classes = n_classes

            self.inc = DoubleConv(n_channels, 32)
            self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
            self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
            self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
            self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

            self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.conv_up1 = DoubleConv(512, 256)

            self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.conv_up2 = DoubleConv(256, 128)

            self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.conv_up3 = DoubleConv(128, 64)

            self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.conv_up4 = DoubleConv(64, 32)

            self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5)
            x = torch.cat([x4, x], dim=1)
            x = self.conv_up1(x)

            x = self.up2(x)
            x = torch.cat([x3, x], dim=1)
            x = self.conv_up2(x)

            x = self.up3(x)
            x = torch.cat([x2, x], dim=1)
            x = self.conv_up3(x)

            x = self.up4(x)
            x = torch.cat([x1, x], dim=1)
            x = self.conv_up4(x)

            logits = self.outc(x)
            return logits


class UNetSegmenter:
    """
    Manages loading and inference for the Lesion Segmentation model.
    Supports PyTorch, OpenCV DNN (ONNX), and robust Otsu morphological fallback.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        input_size: Tuple[int, int] = (224, 224),
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.input_size = input_size
        self.threshold = threshold
        self.model = None
        self.backend = "otsu_fallback"

        if self.model_path and self.model_path.exists():
            self._init_model(device)

    def _init_model(self, device_str: str):
        suffix = self.model_path.suffix.lower()

        # 1. PyTorch weights (.pth / .pt)
        if suffix in (".pth", ".pt") and TORCH_AVAILABLE:
            try:
                if device_str == "auto":
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.device = torch.device(device_str)

                self.model = UNet(n_channels=3, n_classes=1)
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device).eval()
                self.backend = "pytorch_unet"
                print(f"[UNetSegmenter] Loaded PyTorch U-Net from {self.model_path}")
                return
            except Exception as e:
                print(f"[UNetSegmenter] Warning: Failed to load PyTorch weights ({e}). Falling back.")

        # 2. ONNX model via OpenCV DNN (requires no torch)
        if suffix == ".onnx":
            try:
                self.model = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.backend = "opencv_onnx"
                print(f"[UNetSegmenter] Loaded ONNX U-Net via OpenCV DNN from {self.model_path}")
                return
            except Exception as e:
                print(f"[UNetSegmenter] Warning: Failed to load ONNX model ({e}). Falling back.")

        self.backend = "otsu_fallback"

    def segment(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[str], str]:
        """
        Segments lesion from healthy skin.
        Returns:
            binary_mask: np.ndarray (uint8, 0 or 255) of same size as img_bgr (H, W)
            reliability_flags: List[str]
            backend_used: str ("pytorch_unet", "opencv_onnx", or "otsu_fallback")
        """
        h_orig, w_orig = img_bgr.shape[:2]
        flags: List[str] = []

        # Attempt neural network segmentation if loaded
        if self.backend == "pytorch_unet" and self.model is not None and TORCH_AVAILABLE:
            try:
                # Preprocess: RGB, Resize, Normalize
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(img_rgb, self.input_size)
                tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                tensor = (tensor - mean) / std
                tensor = tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    prob = torch.sigmoid(logits).squeeze().cpu().numpy()

                prob_resized = cv2.resize(prob, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                binary_mask = (prob_resized >= self.threshold).astype(np.uint8) * 255

                # Post-processing: remove tiny islands, keep primary contour
                cleaned_mask = self._postprocess_mask(binary_mask)
                return cleaned_mask, flags, "pytorch_unet"
            except Exception as e:
                flags.append(f"UNet inference error: {e}")

        elif self.backend == "opencv_onnx" and self.model is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    img_bgr,
                    scalefactor=1.0 / 255.0,
                    size=self.input_size,
                    mean=(123.675, 116.28, 103.53),
                    swapRB=True,
                    crop=False
                )
                self.model.setInput(blob)
                out = self.model.forward()
                prob = 1.0 / (1.0 + np.exp(-out[0, 0]))
                prob_resized = cv2.resize(prob, (w_orig, h_orig))
                binary_mask = (prob_resized >= self.threshold).astype(np.uint8) * 255
                cleaned_mask = self._postprocess_mask(binary_mask)
                return cleaned_mask, flags, "opencv_onnx"
            except Exception as e:
                flags.append(f"ONNX inference error: {e}")

        # ── Otsu Morphological Fallback ───────────────────────────────────────
        flags.append("Segmented via adaptive Otsu morphological fallback")
        binary_mask, fallback_flags = self._otsu_segment(img_bgr)
        flags.extend(fallback_flags)
        return binary_mask, flags, "otsu_fallback"

    def _otsu_segment(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        flags = []
        b_channel = img_bgr[:, :, 0]
        blurred = cv2.GaussianBlur(b_channel, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = self._postprocess_mask(closed)
        return cleaned, flags

    @staticmethod
    def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
        """Keep the central/largest connected component and fill holes."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return mask

        h, w = mask.shape
        center = np.array([w / 2.0, h / 2.0])

        best_label = 1
        best_score = -1.0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50:
                continue
            dist_to_center = np.linalg.norm(centroids[i] - center)
            dist_penalty = 1.0 / (1.0 + dist_to_center / (w / 3.0))
            score = area * dist_penalty
            if score > best_score:
                best_score = score
                best_label = i

        clean_mask = (labels == best_label).astype(np.uint8) * 255

        # Fill internal holes
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(clean_mask, contours, -1, 255, thickness=cv2.FILLED)

        return clean_mask
