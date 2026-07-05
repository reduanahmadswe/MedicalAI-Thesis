"""Model architectures and factory utilities for multi-label Chest X-ray classification.

Supports DenseNet121, EfficientNet-B0, and ConvNeXt-Tiny through a registry-based
factory so new backbones can be added with a single registration line.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.config import Config, ModelConfig, ModelName, get_default_config


logger = logging.getLogger(__name__)

ModelBuilder = Callable[[ModelConfig], nn.Module]

MODEL_REGISTRY: Dict[ModelName, ModelBuilder] = {}


def register_model(model_name: ModelName) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder function in the global model registry.

    Args:
        model_name: Unique model identifier.

    Returns:
        Decorator that registers the builder function.
    """

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if model_name in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name.value}' is already registered.")
        MODEL_REGISTRY[model_name] = builder
        logger.debug("Registered model builder: %s", model_name.value)
        return builder

    return decorator


def _resolve_pretrained_weights(
    weights_enum: object,
    pretrained: bool,
) -> Optional[object]:
    """Resolve torchvision weight enums from a boolean pretrained flag.

    Args:
        weights_enum: Torchvision weights enum class (e.g., ``DenseNet121_Weights``).
        pretrained: Whether to load default pretrained weights.

    Returns:
        Weights enum member or ``None`` when pretrained is disabled.
    """
    if not pretrained:
        return None
    return weights_enum.DEFAULT  # type: ignore[attr-defined]


class MultiLabelClassifierHead(nn.Module):
    """Dropout-regularized linear head for multi-label disease classification.

    Args:
        in_features: Number of input features from the backbone.
        num_classes: Number of output logits.
        dropout: Dropout probability applied before the final linear layer.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float) -> None:
        """Initialize the classifier head."""
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be a positive integer.")
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Apply dropout and linear projection.

        Args:
            features: Backbone feature tensor of shape ``(B, in_features)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.linear(self.dropout(features))


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Toggle gradient updates for backbone feature extractor parameters.

    Args:
        model: Model containing a ``features`` attribute.
        trainable: Whether backbone parameters should require gradients.
    """
    if not hasattr(model, "features"):
        raise AttributeError("Model does not expose a 'features' attribute for freezing.")

    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


@register_model(ModelName.DENSENET121)
def build_densenet121(model_config: ModelConfig) -> nn.Module:
    """Build a DenseNet121 model with a multi-label classification head.

    Args:
        model_config: Model configuration object.

    Returns:
        Initialized DenseNet121 model.

    Raises:
        ImportError: If torchvision is unavailable.
        RuntimeError: If model construction fails.
    """
    try:
        from torchvision.models import densenet121
        from torchvision.models import DenseNet121_Weights
    except ImportError as exc:
        raise ImportError("torchvision is required to build DenseNet121.") from exc

    weights = _resolve_pretrained_weights(DenseNet121_Weights, model_config.pretrained)

    try:
        model = densenet121(weights=weights)
    except Exception as exc:
        logger.exception("Failed to initialize DenseNet121.")
        raise RuntimeError("DenseNet121 initialization failed.") from exc

    in_features = model.classifier.in_features
    model.classifier = MultiLabelClassifierHead(
        in_features=in_features,
        num_classes=model_config.num_classes,
        dropout=model_config.dropout,
    )

    if model_config.freeze_backbone:
        _set_backbone_trainable(model, trainable=False)

    logger.info(
        "Built DenseNet121 with num_classes=%d, pretrained=%s, freeze_backbone=%s.",
        model_config.num_classes,
        model_config.pretrained,
        model_config.freeze_backbone,
    )
    return model


@register_model(ModelName.EFFICIENTNET_B0)
def build_efficientnet_b0(model_config: ModelConfig) -> nn.Module:
    """Build an EfficientNet-B0 model with a multi-label classification head.

    Args:
        model_config: Model configuration object.

    Returns:
        Initialized EfficientNet-B0 model.

    Raises:
        ImportError: If torchvision is unavailable.
        RuntimeError: If model construction fails.
    """
    try:
        from torchvision.models import efficientnet_b0
        from torchvision.models import EfficientNet_B0_Weights
    except ImportError as exc:
        raise ImportError("torchvision is required to build EfficientNet-B0.") from exc

    weights = _resolve_pretrained_weights(EfficientNet_B0_Weights, model_config.pretrained)

    try:
        model = efficientnet_b0(weights=weights)
    except Exception as exc:
        logger.exception("Failed to initialize EfficientNet-B0.")
        raise RuntimeError("EfficientNet-B0 initialization failed.") from exc

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=model_config.dropout, inplace=True),
        nn.Linear(in_features, model_config.num_classes),
    )

    if model_config.freeze_backbone:
        _set_backbone_trainable(model, trainable=False)

    logger.info(
        "Built EfficientNet-B0 with num_classes=%d, pretrained=%s, freeze_backbone=%s.",
        model_config.num_classes,
        model_config.pretrained,
        model_config.freeze_backbone,
    )
    return model


@register_model(ModelName.CONVNEXT_TINY)
def build_convnext_tiny(model_config: ModelConfig) -> nn.Module:
    """Build a ConvNeXt-Tiny model with a multi-label classification head.

    Args:
        model_config: Model configuration object.

    Returns:
        Initialized ConvNeXt-Tiny model.

    Raises:
        ImportError: If torchvision is unavailable.
        RuntimeError: If model construction fails.
    """
    try:
        from torchvision.models import convnext_tiny
        from torchvision.models import ConvNeXt_Tiny_Weights
    except ImportError as exc:
        raise ImportError("torchvision is required to build ConvNeXt-Tiny.") from exc

    weights = _resolve_pretrained_weights(ConvNeXt_Tiny_Weights, model_config.pretrained)

    try:
        model = convnext_tiny(weights=weights)
    except Exception as exc:
        logger.exception("Failed to initialize ConvNeXt-Tiny.")
        raise RuntimeError("ConvNeXt-Tiny initialization failed.") from exc

    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Linear(in_features, model_config.num_classes),
    )

    if model_config.freeze_backbone:
        _set_backbone_trainable(model, trainable=False)

    logger.info(
        "Built ConvNeXt-Tiny with num_classes=%d, pretrained=%s, freeze_backbone=%s.",
        model_config.num_classes,
        model_config.pretrained,
        model_config.freeze_backbone,
    )
    return model


def build_model(
    config: Optional[Config] = None,
    model_config: Optional[ModelConfig] = None,
) -> nn.Module:
    """Build a model from configuration using the registry.

    Example:
        Change only one line in ``ModelConfig`` to switch architectures::

            model = build_model(build_config(model=ModelConfig(model_name=ModelName.EFFICIENTNET_B0)))

    Args:
        config: Optional full project configuration.
        model_config: Optional model configuration override.

    Returns:
        Initialized PyTorch model.

    Raises:
        ValueError: If the requested model is not registered.
        RuntimeError: If model construction fails.
    """
    config = config or get_default_config()
    model_config = model_config or config.model

    builder = MODEL_REGISTRY.get(model_config.model_name)
    if builder is None:
        supported = [name.value for name in MODEL_REGISTRY]
        raise ValueError(
            f"Unsupported model '{model_config.model_name.value}'. "
            f"Supported models: {supported}."
        )

    try:
        model = builder(model_config)
    except Exception as exc:
        logger.exception("Model build failed for '%s'.", model_config.model_name.value)
        raise RuntimeError(
            f"Failed to build model '{model_config.model_name.value}'."
        ) from exc

    return model


def get_supported_models() -> List[str]:
    """Return a list of supported model identifiers.

    Returns:
        Sorted list of registered model names.
    """
    return sorted(name.value for name in MODEL_REGISTRY)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: Count only parameters with ``requires_grad=True``.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())


def freeze_backbone(model: nn.Module) -> None:
    """Freeze feature extractor weights.

    Args:
        model: Model containing a ``features`` module.

    Raises:
        AttributeError: If the model has no ``features`` attribute.
    """
    _set_backbone_trainable(model, trainable=False)
    logger.info("Backbone parameters frozen.")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze feature extractor weights.

    Args:
        model: Model containing a ``features`` module.

    Raises:
        AttributeError: If the model has no ``features`` attribute.
    """
    _set_backbone_trainable(model, trainable=True)
    logger.info("Backbone parameters unfrozen.")


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel or DistributedDataParallel modules.

    Args:
        model: Possibly wrapped PyTorch model.

    Returns:
        Underlying base model.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def get_model_device(model: nn.Module) -> torch.device:
    """Return the device of the model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Device hosting model parameters.

    Raises:
        RuntimeError: If the model has no parameters.
    """
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def move_model_to_device(
    model: nn.Module,
    device: torch.device | str,
    enable_multi_gpu: bool = False,
) -> nn.Module:
    """Move a model to the target device with optional DataParallel wrapping.

    Args:
        model: PyTorch model to move.
        device: Target device string or ``torch.device``.
        enable_multi_gpu: Wrap with ``nn.DataParallel`` when multiple GPUs exist.

    Returns:
        Model placed on the requested device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    device_obj = torch.device(device)

    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    model = model.to(device_obj)

    if enable_multi_gpu and device_obj.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(
            "Wrapped model with DataParallel across %d GPUs.",
            torch.cuda.device_count(),
        )

    return model


def get_gradcam_target_layer(
    model: nn.Module,
    model_name: Optional[ModelName] = None,
    target_layer_name: Optional[str] = None,
) -> nn.Module:
    """Return the target convolutional layer for Grad-CAM visualization.

    Args:
        model: Trained or initialized model (may be DataParallel-wrapped).
        model_name: Optional model identifier for default layer selection.
        target_layer_name: Optional explicit dotted attribute path override.

    Returns:
        Target layer module for Grad-CAM hooks.

    Raises:
        ValueError: If the target layer cannot be resolved.
        AttributeError: If an explicit layer path is invalid.
    """
    base_model = _unwrap_model(model)

    if target_layer_name:
        layer: nn.Module = base_model
        for attribute in target_layer_name.split("."):
            if not hasattr(layer, attribute):
                raise AttributeError(
                    f"Layer path '{target_layer_name}' is invalid at attribute '{attribute}'."
                )
            layer = getattr(layer, attribute)
        if not isinstance(layer, nn.Module):
            raise ValueError(
                f"Resolved target '{target_layer_name}' is not an nn.Module."
            )
        return layer

    resolved_model_name = model_name or getattr(base_model, "model_name", None)
    if resolved_model_name is None:
        resolved_model_name = _infer_model_name(base_model)

    if resolved_model_name == ModelName.DENSENET121:
        denseblock = base_model.features.denseblock4
        return denseblock.denselayer16.conv2

    if resolved_model_name == ModelName.EFFICIENTNET_B0:
        return base_model.features[-1]

    if resolved_model_name == ModelName.CONVNEXT_TINY:
        return base_model.features[-1][-1].block[-1]

    raise ValueError(
        f"Unable to determine Grad-CAM target layer for model '{resolved_model_name}'."
    )


def _infer_model_name(model: nn.Module) -> ModelName:
    """Infer model enum from torchvision class name.

    Args:
        model: Torchvision model instance.

    Returns:
        Inferred ``ModelName`` enum value.

    Raises:
        ValueError: If the model type is not supported.
    """
    class_name = model.__class__.__name__.lower()

    if "densenet121" in class_name:
        return ModelName.DENSENET121
    if "efficientnet" in class_name:
        return ModelName.EFFICIENTNET_B0
    if "convnext" in class_name and "tiny" in class_name:
        return ModelName.CONVNEXT_TINY

    raise ValueError(
        f"Unable to infer model name from class '{model.__class__.__name__}'."
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Optional[Config] = None,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Load a model and metadata from a checkpoint file.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint file.
        config: Optional configuration used to rebuild the architecture.
        map_location: Device mapping for ``torch.load``.
        strict: Whether to strictly enforce state dict key matching.

    Returns:
        Tuple of ``(model, checkpoint_dict)``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If checkpoint loading fails.
    """
    checkpoint_file = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint_file, dict):
        raise RuntimeError("Checkpoint must be a dictionary.")

    config = config or get_default_config()
    model = build_model(config=config)

    state_dict = checkpoint_file.get("model_state_dict", checkpoint_file.get("state_dict"))
    if state_dict is None:
        raise RuntimeError(
            "Checkpoint does not contain 'model_state_dict' or 'state_dict'."
        )

    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as exc:
        logger.exception("Failed to load checkpoint state dict from %s.", checkpoint_path)
        raise RuntimeError(f"Unable to load checkpoint: {checkpoint_path}") from exc

    logger.info("Loaded model checkpoint from %s.", checkpoint_path)
    return model, checkpoint_file


def summarize_model(model: nn.Module) -> Dict[str, int | str]:
    """Return a concise parameter summary for logging.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary containing model class name and parameter counts.
    """
    base_model = _unwrap_model(model)
    summary = {
        "model_class": base_model.__class__.__name__,
        "total_parameters": count_parameters(base_model, trainable_only=False),
        "trainable_parameters": count_parameters(base_model, trainable_only=True),
    }
    logger.info(
        "Model summary: class=%s, total_params=%d, trainable_params=%d.",
        summary["model_class"],
        summary["total_parameters"],
        summary["trainable_parameters"],
    )
    return summary
