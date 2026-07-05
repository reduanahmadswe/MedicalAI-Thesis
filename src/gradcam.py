"""Grad-CAM and Grad-CAM++ explainability for Chest X-ray disease classification.

Provides activation heatmaps, overlay visualizations, and human-readable
prediction explanations for multi-label models.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from src.config import Config, GradCAMConfig, NIH_DISEASE_LABELS, NUM_CLASSES, get_default_config
from src.dataset import load_single_image
from src.models import get_gradcam_target_layer, load_model_from_checkpoint, move_model_to_device
from src.transforms import tensor_to_pil_image


logger = logging.getLogger(__name__)


@dataclass
class GradCAMResult:
    """Explainability output for a single Chest X-ray prediction.

    Attributes:
        image_path: Source image path.
        target_class_index: Class index used for CAM computation.
        target_class_name: Human-readable disease label.
        probability: Predicted probability for the target class.
        predicted: Binary prediction for the target class.
        heatmap: Normalized CAM heatmap array of shape ``(H, W)`` in ``[0, 1]``.
        overlay: RGB overlay image as uint8 array of shape ``(H, W, 3)``.
        predicted_diseases: All diseases predicted above threshold.
        probabilities: Full probability vector for all classes.
        method: CAM method used ('gradcam' or 'gradcam++').
    """

    image_path: str
    target_class_index: int
    target_class_name: str
    probability: float
    predicted: int
    heatmap: np.ndarray
    overlay: np.ndarray
    predicted_diseases: List[str]
    probabilities: Dict[str, float]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation result to a JSON-serializable dictionary."""
        return {
            "image_path": self.image_path,
            "target_class_index": self.target_class_index,
            "target_class_name": self.target_class_name,
            "probability": self.probability,
            "predicted": self.predicted,
            "predicted_diseases": self.predicted_diseases,
            "probabilities": self.probabilities,
            "method": self.method,
        }


class _ActivationHookManager:
    """Manage forward/backward hooks for target layer activations and gradients."""

    def __init__(self) -> None:
        """Initialize empty hook storage."""
        self.activations: Optional[Tensor] = None
        self.gradients: Optional[Tensor] = None
        self._forward_handle = None
        self._backward_handle = None

    def _forward_hook(self, module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        """Store forward activations."""
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]) -> None:
        """Store backward gradients."""
        self.gradients = grad_output[0].detach()

    def register(self, target_layer: nn.Module) -> None:
        """Register hooks on the target layer.

        Args:
            target_layer: Layer to attach hooks to.
        """
        self.remove()
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def remove(self) -> None:
        """Remove registered hooks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def get_tensors(self) -> Tuple[Tensor, Tensor]:
        """Return captured activations and gradients.

        Returns:
            Tuple of activation and gradient tensors.

        Raises:
            RuntimeError: If hooks did not capture required tensors.
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        return self.activations, self.gradients


class BaseGradCAM(ABC):
    """Abstract base class for Grad-CAM explainers.

    Args:
        model: Trained classification model.
        target_layer: Convolutional target layer for CAM computation.
        device: Device used for inference and backpropagation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize the Grad-CAM explainer."""
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.hooks = _ActivationHookManager()
        self.hooks.register(target_layer)

    @abstractmethod
    def compute_cam(
        self,
        activations: Tensor,
        gradients: Tensor,
    ) -> Tensor:
        """Compute class activation map from activations and gradients."""

    def generate(
        self,
        input_tensor: Tensor,
        target_class_index: int,
    ) -> Tensor:
        """Generate a Grad-CAM heatmap for a target class.

        Args:
            input_tensor: Model input tensor of shape ``(1, C, H, W)``.
            target_class_index: Target disease class index.

        Returns:
            CAM heatmap tensor of shape ``(H, W)`` normalized to ``[0, 1]``.

        Raises:
            ValueError: If class index is invalid.
            RuntimeError: If CAM generation fails.
        """
        if not 0 <= target_class_index < NUM_CLASSES:
            raise ValueError(
                f"target_class_index must be in [0, {NUM_CLASSES - 1}], got {target_class_index}."
            )

        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)
            score = logits[:, target_class_index].sum()
            score.backward(retain_graph=False)

            activations, gradients = self.hooks.get_tensors()
            cam = self.compute_cam(activations=activations, gradients=gradients)

            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu()
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            return cam
        except Exception as exc:
            logger.exception("Grad-CAM generation failed for class index %d.", target_class_index)
            raise RuntimeError("Grad-CAM generation failed.") from exc

    def close(self) -> None:
        """Remove hooks and release resources."""
        self.hooks.remove()


class GradCAM(BaseGradCAM):
    """Standard Grad-CAM implementation."""

    def compute_cam(self, activations: Tensor, gradients: Tensor) -> Tensor:
        """Compute Grad-CAM heatmap.

        Args:
            activations: Forward activation tensor of shape ``(N, C, H, W)``.
            gradients: Gradient tensor of shape ``(N, C, H, W)``.

        Returns:
            CAM tensor of shape ``(N, H, W)``.
        """
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        return cam


class GradCAMPlusPlus(BaseGradCAM):
    """Grad-CAM++ implementation with improved localization for multiple instances."""

    def compute_cam(self, activations: Tensor, gradients: Tensor) -> Tensor:
        """Compute Grad-CAM++ heatmap.

        Args:
            activations: Forward activation tensor of shape ``(N, C, H, W)``.
            gradients: Gradient tensor of shape ``(N, C, H, W)``.

        Returns:
            CAM tensor of shape ``(N, H, W)``.
        """
        gradients_2 = gradients ** 2
        gradients_3 = gradients ** 3

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha_denominator = 2.0 * gradients_2 + sum_activations * gradients_3
        alpha_denominator = torch.where(
            alpha_denominator != 0.0,
            alpha_denominator,
            torch.ones_like(alpha_denominator),
        )
        alpha = gradients_2 / alpha_denominator
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        return cam


def create_gradcam(
    model: nn.Module,
    config: Optional[Config] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseGradCAM:
    """Create a Grad-CAM or Grad-CAM++ explainer from configuration.

    Args:
        model: Trained model instance.
        config: Optional project configuration.
        device: Optional compute device.

    Returns:
        Initialized Grad-CAM explainer.

    Raises:
        ValueError: If the configured method is unsupported.
    """
    config = config or get_default_config()
    device_obj = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    target_layer = get_gradcam_target_layer(
        model=model,
        model_name=config.model.model_name,
        target_layer_name=config.gradcam.target_layer_name,
    )

    if config.gradcam.method == "gradcam":
        explainer: BaseGradCAM = GradCAM(model=model, target_layer=target_layer, device=device_obj)
    elif config.gradcam.method == "gradcam++":
        explainer = GradCAMPlusPlus(model=model, target_layer=target_layer, device=device_obj)
    else:
        raise ValueError(f"Unsupported Grad-CAM method '{config.gradcam.method}'.")

    logger.info(
        "Created %s explainer on layer '%s'.",
        config.gradcam.method,
        target_layer.__class__.__name__,
    )
    return explainer


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """Apply a matplotlib colormap to a normalized heatmap.

    Args:
        heatmap: Normalized heatmap array of shape ``(H, W)`` in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        RGB heatmap array of shape ``(H, W, 3)`` with values in ``[0, 255]``.

    Raises:
        ValueError: If heatmap shape is invalid.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}.")

    cmap = cm.get_cmap(colormap)
    colored = cmap(np.clip(heatmap, 0.0, 1.0))
    rgb = (colored[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def overlay_heatmap_on_image(
    base_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a CAM heatmap over an RGB image.

    Args:
        base_image: Base RGB image as PIL Image or uint8 numpy array.
        heatmap: Normalized heatmap array of shape ``(H, W)``.
        alpha: Heatmap blending weight in ``[0, 1]``.
        colormap: Matplotlib colormap name.

    Returns:
        Overlay image as uint8 numpy array of shape ``(H, W, 3)``.

    Raises:
        ValueError: If alpha is invalid or shapes are incompatible.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0.0, 1.0].")

    if isinstance(base_image, Image.Image):
        base_rgb = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    else:
        base_rgb = np.asarray(base_image, dtype=np.float32)
        if base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
            raise ValueError("base_image array must have shape (H, W, 3).")

    if base_rgb.max() > 1.0:
        base_rgb = base_rgb / 255.0

    heatmap_resized = _resize_heatmap(heatmap, target_size=(base_rgb.shape[0], base_rgb.shape[1]))
    heatmap_rgb = apply_colormap(heatmap_resized, colormap=colormap).astype(np.float32) / 255.0

    overlay = (1.0 - alpha) * base_rgb + alpha * heatmap_rgb
    overlay = np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8)
    return overlay


def _resize_heatmap(
    heatmap: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Resize a heatmap to match the base image size.

    Args:
        heatmap: Input heatmap array of shape ``(H, W)``.
        target_size: Target ``(height, width)``.

    Returns:
        Resized heatmap array.
    """
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    resized = F.interpolate(
        heatmap_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze().numpy()


@torch.no_grad()
def _predict_probabilities(
    model: nn.Module,
    input_tensor: Tensor,
) -> np.ndarray:
    """Compute sigmoid probabilities for a single input batch.

    Args:
        model: Trained model.
        input_tensor: Input tensor of shape ``(1, C, H, W)``.

    Returns:
        Probability vector of shape ``(C,)``.
    """
    logits = model(input_tensor)
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return probabilities


def _resolve_target_class_index(
    probabilities: np.ndarray,
    config: GradCAMConfig,
    threshold: float,
) -> int:
    """Resolve the target class index for CAM generation.

    Args:
        probabilities: Predicted probability vector.
        config: Grad-CAM configuration.
        threshold: Decision threshold for predicted classes.

    Returns:
        Target class index.

    Raises:
        ValueError: If no valid target class can be resolved.
    """
    if config.target_class_index is not None:
        return int(config.target_class_index)

    if config.use_predicted_class:
        predicted_indices = np.where(probabilities >= threshold)[0]
        if predicted_indices.size > 0:
            return int(predicted_indices[np.argmax(probabilities[predicted_indices])])
        return int(np.argmax(probabilities))

    raise ValueError(
        "Unable to resolve target class. Set target_class_index or enable use_predicted_class."
    )


class GradCAMExplainer:
    """High-level explainability pipeline for Chest X-ray predictions.

    Args:
        config: Full project configuration.
        model: Optional pre-loaded model.
        checkpoint_path: Optional checkpoint path for model loading.
        device: Optional compute device.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the explainer."""
        self.config = config or get_default_config()
        self.config.setup()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else self.config.paths.best_model_path

        if model is None:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            self.model, _ = load_model_from_checkpoint(
                checkpoint_path=str(checkpoint),
                config=self.config,
                map_location=self.device,
            )
        else:
            self.model = model

        self.model = move_model_to_device(self.model, device=self.device, enable_multi_gpu=False)
        self.model.eval()
        self.gradcam = create_gradcam(model=self.model, config=self.config, device=self.device)

    def explain_image(
        self,
        image_path: Union[str, Path],
        target_class_index: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> GradCAMResult:
        """Generate Grad-CAM explanation for a single image.

        Args:
            image_path: Path to the Chest X-ray image.
            target_class_index: Optional explicit target class index.
            threshold: Optional decision threshold for predicted disease selection.

        Returns:
            ``GradCAMResult`` containing heatmap and explanation metadata.
        """
        image_path = Path(image_path)
        threshold_value = threshold if threshold is not None else self.config.evaluation.threshold

        input_tensor = load_single_image(image_path=image_path, config=self.config).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)

        probabilities = _predict_probabilities(self.model, input_tensor)
        gradcam_config = self.config.gradcam

        if target_class_index is not None:
            class_index = int(target_class_index)
        else:
            class_index = _resolve_target_class_index(
                probabilities=probabilities,
                config=gradcam_config,
                threshold=threshold_value,
            )

        heatmap_tensor = self.gradcam.generate(
            input_tensor=input_tensor,
            target_class_index=class_index,
        )
        heatmap = heatmap_tensor.numpy()

        base_image = tensor_to_pil_image(input_tensor.squeeze(0).cpu(), data_config=self.config.data)
        overlay = overlay_heatmap_on_image(
            base_image=base_image,
            heatmap=heatmap,
            alpha=gradcam_config.alpha,
            colormap=gradcam_config.colormap,
        )

        predicted_diseases = [
            NIH_DISEASE_LABELS[index]
            for index, probability in enumerate(probabilities)
            if probability >= threshold_value
        ]

        result = GradCAMResult(
            image_path=str(image_path),
            target_class_index=class_index,
            target_class_name=NIH_DISEASE_LABELS[class_index],
            probability=float(probabilities[class_index]),
            predicted=int(probabilities[class_index] >= threshold_value),
            heatmap=heatmap,
            overlay=overlay,
            predicted_diseases=predicted_diseases,
            probabilities={
                class_name: float(probabilities[index])
                for index, class_name in enumerate(NIH_DISEASE_LABELS)
            },
            method=gradcam_config.method,
        )
        return result

    def explain_and_save(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        target_class_index: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> GradCAMResult:
        """Generate and persist Grad-CAM artifacts for an image.

        Saves:
        - ``*_heatmap.png``
        - ``*_overlay.png``
        - ``*_explanation.json``

        Args:
            image_path: Path to the Chest X-ray image.
            output_dir: Optional output directory.
            target_class_index: Optional explicit target class index.
            threshold: Optional decision threshold.

        Returns:
            ``GradCAMResult`` for the explained image.
        """
        result = self.explain_image(
            image_path=image_path,
            target_class_index=target_class_index,
            threshold=threshold,
        )

        output_directory = Path(output_dir) if output_dir is not None else (
            self.config.paths.gradcam_dir
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        image_stem = Path(result.image_path).stem
        heatmap_path = output_directory / f"{image_stem}_heatmap.png"
        overlay_path = output_directory / f"{image_stem}_overlay.png"
        json_path = output_directory / f"{image_stem}_explanation.json"

        try:
            _save_heatmap_image(result.heatmap, heatmap_path, colormap=self.config.gradcam.colormap)
            Image.fromarray(result.overlay).save(overlay_path)
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump(result.to_dict(), json_file, indent=2)
        except Exception as exc:
            logger.exception("Failed to save Grad-CAM artifacts for %s.", image_path)
            raise OSError(f"Unable to save Grad-CAM artifacts to {output_directory}") from exc

        logger.info(
            "Saved Grad-CAM explanation for '%s' to %s.",
            result.target_class_name,
            output_directory,
        )
        return result

    def explain_predicted_diseases(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[GradCAMResult]:
        """Generate CAM explanations for all diseases predicted above threshold.

        Args:
            image_path: Path to the Chest X-ray image.
            threshold: Optional decision threshold.
            output_dir: Optional output directory for saved artifacts.

        Returns:
            List of ``GradCAMResult`` objects, one per predicted disease.
        """
        threshold_value = threshold if threshold is not None else self.config.evaluation.threshold
        input_tensor = load_single_image(image_path=image_path, config=self.config).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        probabilities = _predict_probabilities(self.model, input_tensor)

        predicted_indices = np.where(probabilities >= threshold_value)[0]
        if predicted_indices.size == 0:
            predicted_indices = np.array([int(np.argmax(probabilities))])

        results: List[GradCAMResult] = []
        for class_index in predicted_indices:
            if output_dir is not None:
                class_output = Path(output_dir) / NIH_DISEASE_LABELS[class_index]
                result = self.explain_and_save(
                    image_path=image_path,
                    output_dir=class_output,
                    target_class_index=int(class_index),
                    threshold=threshold_value,
                )
            else:
                result = self.explain_image(
                    image_path=image_path,
                    target_class_index=int(class_index),
                    threshold=threshold_value,
                )
            results.append(result)

        return results

    def explain_batch_and_save(
        self,
        image_paths: Sequence[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
    ) -> List[GradCAMResult]:
        """Generate and save Grad-CAM explanations for multiple images.

        Args:
            image_paths: Sequence of Chest X-ray image paths.
            output_dir: Optional base output directory.
            threshold: Optional decision threshold for target class selection.

        Returns:
            List of ``GradCAMResult`` objects, one per image.
        """
        if not image_paths:
            raise ValueError("image_paths must contain at least one path.")

        output_directory = Path(output_dir) if output_dir is not None else (
            self.config.paths.gradcam_dir / "batch"
        )
        output_directory.mkdir(parents=True, exist_ok=True)

        results: List[GradCAMResult] = []
        for image_path in image_paths:
            result = self.explain_and_save(
                image_path=image_path,
                output_dir=output_directory / Path(image_path).stem,
                threshold=threshold,
            )
            results.append(result)

        logger.info(
            "Saved %d Grad-CAM explainability outputs to %s.",
            len(results),
            output_directory,
        )
        return results

    def close(self) -> None:
        """Release Grad-CAM hooks and resources."""
        self.gradcam.close()


def _save_heatmap_image(
    heatmap: np.ndarray,
    output_path: Union[str, Path],
    colormap: str = "jet",
) -> None:
    """Save a heatmap array as a color PNG image.

    Args:
        heatmap: Normalized heatmap array of shape ``(H, W)``.
        output_path: Destination PNG path.
        colormap: Matplotlib colormap name.
    """
    heatmap_rgb = apply_colormap(heatmap, colormap=colormap)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heatmap_rgb).save(output_path)


def explain_prediction(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    target_class_index: Optional[int] = None,
    save: bool = True,
    output_dir: Optional[Union[str, Path]] = None,
) -> GradCAMResult:
    """Explain a model prediction for a single Chest X-ray image.

    Args:
        image_path: Path to the input image.
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.
        target_class_index: Optional explicit target class index.
        save: Whether to save heatmap, overlay, and JSON explanation files.
        output_dir: Optional output directory when ``save=True``.

    Returns:
        Grad-CAM explanation result.
    """
    explainer = GradCAMExplainer(
        config=config,
        checkpoint_path=checkpoint_path,
    )
    try:
        if save:
            return explainer.explain_and_save(
                image_path=image_path,
                output_dir=output_dir,
                target_class_index=target_class_index,
            )
        return explainer.explain_image(
            image_path=image_path,
            target_class_index=target_class_index,
        )
    finally:
        explainer.close()


def generate_gradcam_for_image(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    method: Optional[str] = None,
    target_class_index: Optional[int] = None,
) -> GradCAMResult:
    """Convenience wrapper to generate Grad-CAM for one image.

    Args:
        image_path: Path to the Chest X-ray image.
        config: Optional project configuration.
        checkpoint_path: Optional checkpoint path.
        method: Optional CAM method override ('gradcam' or 'gradcam++').
        target_class_index: Optional target class index.

    Returns:
        Grad-CAM explanation result.
    """
    config = config or get_default_config()
    if method is not None:
        from dataclasses import replace

        config = replace(config, gradcam=replace(config.gradcam, method=method))

    return explain_prediction(
        image_path=image_path,
        config=config,
        checkpoint_path=checkpoint_path,
        target_class_index=target_class_index,
        save=True,
    )


def explain_batch_and_save(
    image_paths: Sequence[Union[str, Path]],
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    method: Optional[str] = None,
    threshold: Optional[float] = None,
) -> List[GradCAMResult]:
    """Generate and save Grad-CAM explanations for a batch of images.

    Args:
        image_paths: Sequence of Chest X-ray image paths.
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.
        output_dir: Optional output directory.
        method: Optional CAM method override (``gradcam`` or ``gradcam++``).
        threshold: Optional decision threshold.

    Returns:
        List of ``GradCAMResult`` objects.
    """
    config = config or get_default_config()
    if method is not None:
        from dataclasses import replace

        config = replace(config, gradcam=replace(config.gradcam, method=method))

    explainer = GradCAMExplainer(config=config, checkpoint_path=checkpoint_path)
    try:
        return explainer.explain_batch_and_save(
            image_paths=image_paths,
            output_dir=output_dir,
            threshold=threshold,
        )
    finally:
        explainer.close()
