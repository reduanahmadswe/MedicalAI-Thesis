"""Loss functions for multi-label Chest X-ray disease classification.

Provides weighted binary cross-entropy and optional focal loss through a
registry-based factory integrated with ``LossConfig`` and dataset-derived
class weights.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import Config, LossConfig, LossName, NUM_CLASSES, get_default_config


logger = logging.getLogger(__name__)

LossBuilder = Callable[[LossConfig, Optional[Tensor], Optional[torch.device]], nn.Module]

LOSS_REGISTRY: Dict[LossName, LossBuilder] = {}


def register_loss(loss_name: LossName) -> Callable[[LossBuilder], LossBuilder]:
    """Register a loss builder function in the global loss registry.

    Args:
        loss_name: Unique loss identifier.

    Returns:
        Decorator that registers the builder function.
    """

    def decorator(builder: LossBuilder) -> LossBuilder:
        if loss_name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{loss_name.value}' is already registered.")
        LOSS_REGISTRY[loss_name] = builder
        logger.debug("Registered loss builder: %s", loss_name.value)
        return builder

    return decorator


def _resolve_device(device: Optional[torch.device]) -> torch.device:
    """Resolve a target device for loss tensors.

    Args:
        device: Optional explicit device.

    Returns:
        Resolved torch device.
    """
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_pos_weight(
    loss_config: LossConfig,
    pos_weight: Optional[Tensor],
    device: torch.device,
) -> Optional[Tensor]:
    """Resolve positive class weights from runtime or configuration values.

    Args:
        loss_config: Loss configuration object.
        pos_weight: Optional runtime tensor from the training dataset.
        device: Device on which the loss module will run.

    Returns:
        Positive weight tensor or ``None`` when weighting is disabled.

    Raises:
        ValueError: If configured weight length does not match ``NUM_CLASSES``.
        TypeError: If ``pos_weight`` is not a tensor when provided.
    """
    if pos_weight is not None:
        if not isinstance(pos_weight, Tensor):
            raise TypeError(
                f"pos_weight must be a torch.Tensor, got {type(pos_weight).__name__}."
            )
        if pos_weight.numel() != NUM_CLASSES:
            raise ValueError(
                f"pos_weight must have {NUM_CLASSES} elements, got {pos_weight.numel()}."
            )
        return pos_weight.to(device=device, dtype=torch.float32)

    if loss_config.pos_weight is not None:
        weight_tensor = torch.tensor(loss_config.pos_weight, dtype=torch.float32, device=device)
        if weight_tensor.numel() != NUM_CLASSES:
            raise ValueError(
                f"Configured pos_weight must have {NUM_CLASSES} elements, "
                f"got {weight_tensor.numel()}."
            )
        return weight_tensor

    return None


def apply_label_smoothing(targets: Tensor, smoothing: float) -> Tensor:
    """Apply label smoothing to multi-label binary targets.

    Smoothed targets are computed as:
    ``targets * (1 - smoothing) + 0.5 * smoothing``.

    Args:
        targets: Binary label tensor of shape ``(N, C)``.
        smoothing: Label smoothing factor in ``[0, 1)``.

    Returns:
        Smoothed target tensor with the same shape as the input.

    Raises:
        ValueError: If smoothing is outside ``[0, 1)`` or targets are invalid.
    """
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must be in the range [0.0, 1.0).")
    if targets.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        targets = targets.float()
    if torch.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("targets must contain values in [0, 1] for label smoothing.")

    if smoothing == 0.0:
        return targets

    return targets * (1.0 - smoothing) + 0.5 * smoothing


class WeightedBCEWithLogitsLoss(nn.Module):
    """Weighted binary cross-entropy with optional label smoothing.

    Wraps ``nn.BCEWithLogitsLoss`` for multi-label classification with
    per-class positive weights and optional target smoothing.

    Args:
        pos_weight: Optional per-class positive weights of shape ``(C,)``.
        label_smoothing: Label smoothing factor in ``[0, 1)``.
        reduction: Loss reduction mode ('mean', 'sum', or 'none').
    """

    def __init__(
        self,
        pos_weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize the weighted BCE loss module."""
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")

        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

        self._bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight,
            reduction=reduction,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Model output logits of shape ``(N, C)``.
            targets: Ground-truth labels of shape ``(N, C)``.

        Returns:
            Scalar or tensor loss depending on ``reduction``.

        Raises:
            ValueError: If logits and targets shapes do not match.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        smoothed_targets = apply_label_smoothing(targets.float(), self.label_smoothing)
        return self._bce_loss(logits, smoothed_targets)


class MultiLabelFocalLoss(nn.Module):
    """Focal loss for multi-label classification with optional class weighting.

    Applies focal modulation to per-label binary cross-entropy terms:
    ``FL = alpha_t * (1 - p_t) ** gamma * BCE(logits, targets)``.

    Args:
        alpha: Balancing factor for positive/negative labels.
        gamma: Focusing parameter for hard example down-weighting.
        pos_weight: Optional per-class positive weights of shape ``(C,)``.
        label_smoothing: Label smoothing factor in ``[0, 1)``.
        reduction: Loss reduction mode ('mean', 'sum', or 'none').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize the focal loss module."""
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in the range [0.0, 1.0].")
        if gamma < 0.0:
            raise ValueError("gamma must be >= 0.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}.")

        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute multi-label focal loss.

        Args:
            logits: Model output logits of shape ``(N, C)``.
            targets: Ground-truth labels of shape ``(N, C)``.

        Returns:
            Scalar or tensor loss depending on ``reduction``.

        Raises:
            ValueError: If logits and targets shapes do not match.
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got "
                f"{tuple(logits.shape)} vs {tuple(targets.shape)}."
            )

        targets = apply_label_smoothing(targets.float(), self.label_smoothing)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight,
        )

        probabilities = torch.sigmoid(logits)
        positive_term = targets * (1.0 - probabilities)
        negative_term = (1.0 - targets) * probabilities
        modulating_factor = torch.pow(positive_term + negative_term, self.gamma)

        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "none":
            return focal_loss
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss.mean()


@register_loss(LossName.WEIGHTED_BCE)
def build_weighted_bce_loss(
    loss_config: LossConfig,
    pos_weight: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Build weighted BCEWithLogitsLoss from configuration.

    Args:
        loss_config: Loss configuration object.
        pos_weight: Optional runtime positive class weights.
        device: Optional device for weight tensors.

    Returns:
        Initialized weighted BCE loss module.
    """
    device_obj = _resolve_device(device)
    resolved_pos_weight = _resolve_pos_weight(loss_config, pos_weight, device_obj)

    loss_module = WeightedBCEWithLogitsLoss(
        pos_weight=resolved_pos_weight,
        label_smoothing=loss_config.label_smoothing,
        reduction="mean",
    )
    logger.info(
        "Built WeightedBCEWithLogitsLoss with class_weighting=%s, label_smoothing=%.4f.",
        resolved_pos_weight is not None,
        loss_config.label_smoothing,
    )
    return loss_module


@register_loss(LossName.FOCAL)
def build_focal_loss(
    loss_config: LossConfig,
    pos_weight: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Build multi-label focal loss from configuration.

    Args:
        loss_config: Loss configuration object.
        pos_weight: Optional runtime positive class weights.
        device: Optional device for weight tensors.

    Returns:
        Initialized focal loss module.
    """
    device_obj = _resolve_device(device)
    resolved_pos_weight = _resolve_pos_weight(loss_config, pos_weight, device_obj)

    loss_module = MultiLabelFocalLoss(
        alpha=loss_config.focal_alpha,
        gamma=loss_config.focal_gamma,
        pos_weight=resolved_pos_weight,
        label_smoothing=loss_config.label_smoothing,
        reduction="mean",
    )
    logger.info(
        "Built MultiLabelFocalLoss with alpha=%.4f, gamma=%.4f, class_weighting=%s.",
        loss_config.focal_alpha,
        loss_config.focal_gamma,
        resolved_pos_weight is not None,
    )
    return loss_module


def build_loss(
    config: Optional[Config] = None,
    loss_config: Optional[LossConfig] = None,
    pos_weight: Optional[Tensor] = None,
    device: Optional[torch.device | str] = None,
) -> nn.Module:
    """Build a loss function from configuration using the registry.

    When ``loss_config.compute_class_weights`` is ``True`` and ``pos_weight`` is
    provided (typically from ``dataset.compute_pos_weight``), class imbalance
    weighting is applied automatically.

    Args:
        config: Optional full project configuration.
        loss_config: Optional loss configuration override.
        pos_weight: Optional per-class positive weights tensor of shape ``(C,)``.
        device: Optional device for loss buffers.

    Returns:
        Initialized loss module.

    Raises:
        ValueError: If the requested loss is not registered.
        RuntimeError: If loss construction fails.
    """
    config = config or get_default_config()
    loss_config = loss_config or config.loss
    device_obj = torch.device(device) if isinstance(device, str) else _resolve_device(device)

    builder = LOSS_REGISTRY.get(loss_config.loss_name)
    if builder is None:
        supported = [name.value for name in LOSS_REGISTRY]
        raise ValueError(
            f"Unsupported loss '{loss_config.loss_name.value}'. "
            f"Supported losses: {supported}."
        )

    runtime_pos_weight = pos_weight
    if not loss_config.compute_class_weights:
        runtime_pos_weight = None
        if loss_config.pos_weight is None:
            logger.info(
                "Class weight computation disabled; using unweighted loss unless "
                "static pos_weight is configured."
            )

    try:
        loss_module = builder(loss_config, runtime_pos_weight, device_obj)
    except Exception as exc:
        logger.exception("Loss build failed for '%s'.", loss_config.loss_name.value)
        raise RuntimeError(
            f"Failed to build loss '{loss_config.loss_name.value}'."
        ) from exc

    return loss_module.to(device_obj)


def get_supported_losses() -> List[str]:
    """Return supported loss function identifiers.

    Returns:
        Sorted list of registered loss names.
    """
    return sorted(name.value for name in LOSS_REGISTRY)


def compute_batch_loss(
    loss_fn: nn.Module,
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Compute loss for a single batch with validation and error handling.

    Args:
        loss_fn: Loss module returned by ``build_loss``.
        logits: Model logits of shape ``(N, C)``.
        targets: Ground-truth labels of shape ``(N, C)``.

    Returns:
        Scalar batch loss tensor.

    Raises:
        TypeError: If inputs are not torch tensors.
        RuntimeError: If loss computation fails.
    """
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("logits and targets must be torch.Tensor instances.")

    try:
        loss = loss_fn(logits, targets)
    except Exception as exc:
        logger.exception("Batch loss computation failed.")
        raise RuntimeError("Failed to compute batch loss.") from exc

    if not isinstance(loss, Tensor):
        raise RuntimeError(
            f"Loss function must return a torch.Tensor, got {type(loss).__name__}."
        )
    return loss
