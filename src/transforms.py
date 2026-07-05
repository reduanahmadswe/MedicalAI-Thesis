"""Image transformation pipelines for Chest X-ray multi-label classification.

Provides split-aware augmentation for training and deterministic preprocessing
for validation, testing, and inference. Transforms are designed for grayscale
NIH ChestX-ray14 images converted to 3-channel tensors for ImageNet-pretrained
backbones.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Compose

from src.config import Config, DataConfig, SplitName, get_default_config


logger = logging.getLogger(__name__)

TransformPipeline = Union[Compose, Callable[..., torch.Tensor]]


class TransformMode(str, Enum):
    """Transform pipeline mode identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INFERENCE = "inference"


class GrayscaleToRGB:
    """Ensure PIL images are converted to RGB before torchvision transforms.

    Handles grayscale (``L``), RGBA, palette (``P``), and other PIL modes by
    converting to RGB. Pretrained backbones expect 3-channel input.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        """Convert an input PIL image to RGB.

        Args:
            image: Input PIL image in any supported mode.

        Returns:
            RGB PIL image.

        Raises:
            TypeError: If the input is not a PIL Image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"GrayscaleToRGB expects a PIL Image, got {type(image).__name__}."
            )

        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


class ResizeMaxSide:
    """Resize an image so that its longest side matches a target length.

    Preserves aspect ratio, which is important for chest radiographs where
    anatomical proportions should not be distorted before center cropping.
    """

    def __init__(self, max_size: int, interpolation: T.InterpolationMode) -> None:
        """Initialize the resize transform.

        Args:
            max_size: Target length for the longest spatial dimension.
            interpolation: PIL interpolation mode used during resizing.

        Raises:
            ValueError: If max_size is not positive.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}.")
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        """Resize the input image while preserving aspect ratio.

        Args:
            image: Input PIL image.

        Returns:
            Resized PIL image.

        Raises:
            TypeError: If the input is not a PIL Image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"ResizeMaxSide expects a PIL Image, got {type(image).__name__}."
            )

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: ({width}, {height}).")

        scale = self.max_size / float(max(width, height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return TF.resize(image, [new_height, new_width], interpolation=self.interpolation)


def _build_normalization(data_config: DataConfig) -> T.Normalize:
    """Create a normalization transform from data configuration.

    Args:
        data_config: Data preprocessing configuration.

    Returns:
        torchvision Normalize transform.
    """
    return T.Normalize(mean=list(data_config.normalize_mean), std=list(data_config.normalize_std))


def build_train_transforms(data_config: Optional[DataConfig] = None) -> Compose:
    """Build augmentation pipeline for training.

    Augmentations are conservative and appropriate for chest radiography:
    horizontal flip, mild rotation/translation, and photometric jitter applied
    after grayscale-to-RGB conversion.

    Args:
        data_config: Optional data configuration. Defaults to project defaults.

    Returns:
        Composed training transform pipeline.
    """
    data_config = data_config or get_default_config().data
    height, width = data_config.image_size
    resize_side = max(height, width)

    transforms = Compose(
        [
            GrayscaleToRGB(),
            ResizeMaxSide(
                max_size=resize_side,
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomCrop(size=data_config.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=7.0,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                shear=3.0,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0,
            ),
            T.ColorJitter(
                brightness=0.08,
                contrast=0.08,
            ),
            T.ToTensor(),
            _build_normalization(data_config),
        ]
    )
    logger.debug(
        "Built training transforms with image_size=%s, resize_side=%d.",
        data_config.image_size,
        resize_side,
    )
    return transforms


def build_eval_transforms(data_config: Optional[DataConfig] = None) -> Compose:
    """Build deterministic preprocessing for validation and testing.

    Args:
        data_config: Optional data configuration. Defaults to project defaults.

    Returns:
        Composed evaluation transform pipeline.
    """
    data_config = data_config or get_default_config().data
    height, width = data_config.image_size
    resize_side = max(height, width)

    transforms = Compose(
        [
            GrayscaleToRGB(),
            ResizeMaxSide(
                max_size=resize_side,
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.CenterCrop(size=data_config.image_size),
            T.ToTensor(),
            _build_normalization(data_config),
        ]
    )
    logger.debug(
        "Built evaluation transforms with image_size=%s, resize_side=%d.",
        data_config.image_size,
        resize_side,
    )
    return transforms


def build_inference_transforms(data_config: Optional[DataConfig] = None) -> Compose:
    """Build deterministic preprocessing for single-image or batch inference.

    This pipeline mirrors validation preprocessing to ensure train/serve parity.

    Args:
        data_config: Optional data configuration. Defaults to project defaults.

    Returns:
        Composed inference transform pipeline.
    """
    return build_eval_transforms(data_config=data_config)


def get_transforms(
    split: Union[SplitName, TransformMode, str],
    config: Optional[Config] = None,
) -> Compose:
    """Return the transform pipeline for a dataset split or runtime mode.

    Args:
        split: One of ``train``, ``val``, ``test``, or ``inference``.
        config: Optional full project configuration.

    Returns:
        Composed transform pipeline for the requested split.

    Raises:
        ValueError: If the split identifier is not recognized.
    """
    config = config or get_default_config()
    split_value = split.value if isinstance(split, Enum) else str(split).lower()

    if split_value == SplitName.TRAIN.value:
        return build_train_transforms(data_config=config.data)
    if split_value in {
        SplitName.VAL.value,
        SplitName.TEST.value,
        TransformMode.INFERENCE.value,
    }:
        return build_eval_transforms(data_config=config.data)

    raise ValueError(
        f"Unknown split '{split}'. Expected one of: "
        f"{SplitName.TRAIN.value}, {SplitName.VAL.value}, "
        f"{SplitName.TEST.value}, {TransformMode.INFERENCE.value}."
    )


def get_all_transforms(config: Optional[Config] = None) -> Dict[str, Compose]:
    """Build transform pipelines for all supported splits.

    Args:
        config: Optional full project configuration.

    Returns:
        Dictionary mapping split names to composed transform pipelines.
    """
    config = config or get_default_config()
    pipelines = {
        SplitName.TRAIN.value: get_transforms(SplitName.TRAIN, config=config),
        SplitName.VAL.value: get_transforms(SplitName.VAL, config=config),
        SplitName.TEST.value: get_transforms(SplitName.TEST, config=config),
        TransformMode.INFERENCE.value: get_transforms(
            TransformMode.INFERENCE, config=config
        ),
    }
    logger.info("Initialized transform pipelines for splits: %s", list(pipelines))
    return pipelines


def denormalize_tensor(
    tensor: torch.Tensor,
    data_config: Optional[DataConfig] = None,
) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization overlays.

    Args:
        tensor: Normalized image tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        data_config: Optional data configuration containing mean and std.

    Returns:
        Denormalized tensor clipped to ``[0, 1]``.

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the tensor shape is invalid.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"denormalize_tensor expects a torch.Tensor, got {type(tensor).__name__}."
        )

    data_config = data_config or get_default_config().data
    mean = torch.tensor(data_config.normalize_mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(data_config.normalize_std, dtype=tensor.dtype, device=tensor.device)

    if tensor.ndim == 3:
        if tensor.shape[0] != 3:
            raise ValueError(
                f"Expected 3-channel tensor with shape (C, H, W), got {tuple(tensor.shape)}."
            )
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    elif tensor.ndim == 4:
        if tensor.shape[1] != 3:
            raise ValueError(
                f"Expected 3-channel tensor with shape (B, C, H, W), got {tuple(tensor.shape)}."
            )
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError(
            f"Expected tensor with 3 or 4 dimensions, got shape {tuple(tensor.shape)}."
        )

    denormalized = tensor * std + mean
    return torch.clamp(denormalized, min=0.0, max=1.0)


def tensor_to_pil_image(
    tensor: torch.Tensor,
    data_config: Optional[DataConfig] = None,
) -> Image.Image:
    """Convert a normalized model input tensor back to a PIL RGB image.

    Args:
        tensor: Normalized tensor of shape ``(C, H, W)``.
        data_config: Optional data configuration for denormalization.

    Returns:
        PIL RGB image suitable for overlaying Grad-CAM heatmaps.

    Raises:
        TypeError: If the input is not a torch.Tensor.
        ValueError: If the tensor shape is invalid.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"tensor_to_pil_image expects a torch.Tensor, got {type(tensor).__name__}."
        )
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected tensor shape (3, H, W), got {tuple(tensor.shape)}."
        )

    denormalized = denormalize_tensor(tensor, data_config=data_config)
    return TF.to_pil_image(denormalized.detach().cpu())


def apply_transforms_to_pil(
    image: Image.Image,
    transforms: TransformPipeline,
) -> torch.Tensor:
    """Apply a transform pipeline to a PIL image with validation.

    Args:
        image: Input PIL image.
        transforms: Composed transform pipeline.

    Returns:
        Transformed image tensor.

    Raises:
        TypeError: If the image is not a PIL Image.
        RuntimeError: If transform application fails.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            f"apply_transforms_to_pil expects a PIL Image, got {type(image).__name__}."
        )

    try:
        output = transforms(image)
    except Exception as exc:
        logger.exception("Failed to apply transforms to image.")
        raise RuntimeError("Transform pipeline failed during image preprocessing.") from exc

    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            f"Transform pipeline must return torch.Tensor, got {type(output).__name__}."
        )
    return output
