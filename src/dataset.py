"""Dataset and DataLoader utilities for NIH ChestX-ray14 multi-label classification.

This module provides CSV-backed dataset loading, path resolution, class-weight
computation, and factory helpers for training, validation, testing, and
inference pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import Config, PathConfig, SplitName, get_default_config
from src.transforms import TransformPipeline, get_transforms


logger = logging.getLogger(__name__)

SampleDict = Dict[str, Union[torch.Tensor, str, int]]


def _split_to_csv_path(split: Union[SplitName, str], paths: PathConfig) -> Path:
    """Map a split identifier to its configured CSV path.

    Args:
        split: Dataset split name.
        paths: Path configuration object.

    Returns:
        Path to the CSV file for the requested split.

    Raises:
        ValueError: If the split name is not recognized.
    """
    split_value = split.value if isinstance(split, SplitName) else str(split).lower()
    split_map = {
        SplitName.TRAIN.value: paths.train_csv,
        SplitName.VAL.value: paths.val_csv,
        SplitName.TEST.value: paths.test_csv,
    }
    if split_value not in split_map:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of: {list(split_map.keys())}."
        )
    return split_map[split_value]


def resolve_image_path(
    image_path: Union[str, Path],
    paths: PathConfig,
) -> Path:
    """Resolve an image path from CSV into an absolute filesystem path.

    Resolution order:
    1. Use the path directly if it already exists.
    2. Resolve relative to ``paths.data_root``.
    3. Resolve relative to ``paths.project_root``.

    Args:
        image_path: Relative or absolute image path from the CSV.
        paths: Path configuration object.

    Returns:
        Absolute resolved path to the image file.

    Raises:
        ValueError: If the image path string is empty.
        FileNotFoundError: If the image cannot be located.
    """
    if not str(image_path).strip():
        raise ValueError("Image path must be a non-empty string.")

    candidate = Path(image_path)
    search_paths = [
        candidate,
        paths.data_root / candidate,
        paths.project_root / candidate,
    ]

    if candidate.is_absolute():
        search_paths = [candidate, paths.data_root / candidate.name]

    for path_option in search_paths:
        if path_option.exists():
            return path_option.resolve()

    raise FileNotFoundError(
        f"Image not found for path '{image_path}'. Checked: "
        f"{[str(path) for path in search_paths]}."
    )


def load_split_dataframe(
    split: Union[SplitName, str],
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Load and validate a dataset split CSV file.

    Args:
        split: Dataset split identifier.
        config: Optional project configuration.

    Returns:
        Validated pandas DataFrame for the requested split.

    Raises:
        FileNotFoundError: If the split CSV does not exist.
        ValueError: If required columns are missing or invalid.
    """
    config = config or get_default_config()
    csv_path = _split_to_csv_path(split, config.paths)

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as exc:
        logger.exception("Failed to read CSV file: %s", csv_path)
        raise RuntimeError(f"Unable to read split CSV: {csv_path}") from exc

    if dataframe.empty:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    required_columns = [config.paths.image_column, *config.paths.label_columns]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {missing_columns}."
        )

    for label_column in config.paths.label_columns:
        if not pd.api.types.is_numeric_dtype(dataframe[label_column]):
            raise ValueError(
                f"Label column '{label_column}' must be numeric (0/1 encoding)."
            )

    logger.info(
        "Loaded %s split with %d samples from %s.",
        split.value if isinstance(split, SplitName) else split,
        len(dataframe),
        csv_path,
    )
    return dataframe.reset_index(drop=True)


class ChestXrayDataset(Dataset[SampleDict]):
    """PyTorch Dataset for NIH ChestX-ray14 multi-label classification.

    Each sample returns a dictionary containing the transformed image tensor,
    multi-hot label vector, resolved image path, and sample index.

    Args:
        dataframe: Pre-loaded split dataframe.
        transform: Optional image transform pipeline.
        paths: Path configuration for image resolution.
        image_column: CSV column containing image paths.
        label_columns: CSV columns containing binary disease labels.
        validate_images: Verify image existence during dataset initialization.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[TransformPipeline] = None,
        paths: Optional[PathConfig] = None,
        image_column: str = "Image_Path",
        label_columns: Optional[Sequence[str]] = None,
        validate_images: bool = False,
    ) -> None:
        """Initialize the Chest X-ray dataset."""
        if dataframe.empty:
            raise ValueError("dataframe must contain at least one sample.")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.paths = paths or get_default_config().paths
        self.image_column = image_column
        self.label_columns = tuple(label_columns or self.paths.label_columns)

        self._validate_columns()
        self.labels_array = self._extract_labels()
        self.num_classes = self.labels_array.shape[1]

        if validate_images:
            self._validate_image_paths()

        logger.debug(
            "Initialized ChestXrayDataset with %d samples and %d classes.",
            len(self),
            self.num_classes,
        )

    def _validate_columns(self) -> None:
        """Ensure required dataframe columns are present."""
        required = [self.image_column, *self.label_columns]
        missing = [column for column in required if column not in self.dataframe.columns]
        if missing:
            raise ValueError(f"dataframe is missing required columns: {missing}")

    def _extract_labels(self) -> np.ndarray:
        """Extract multi-label targets as a float32 numpy array."""
        labels = self.dataframe[list(self.label_columns)].astype(np.float32).to_numpy()
        if labels.ndim != 2:
            raise ValueError("Label array must be two-dimensional.")
        if np.any((labels != 0.0) & (labels != 1.0)):
            logger.warning(
                "Non-binary values detected in label columns; values will be clipped to [0, 1]."
            )
            labels = np.clip(labels, 0.0, 1.0)
        return labels

    def _validate_image_paths(self) -> None:
        """Validate that all image paths in the dataset can be resolved."""
        missing_count = 0
        for image_path in self.dataframe[self.image_column]:
            try:
                resolve_image_path(image_path, self.paths)
            except FileNotFoundError:
                missing_count += 1

        if missing_count > 0:
            raise FileNotFoundError(
                f"{missing_count} image path(s) could not be resolved during validation."
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, index: int) -> SampleDict:
        """Load and transform a sample by index.

        Args:
            index: Sample index.

        Returns:
            Dictionary with keys ``image``, ``labels``, ``image_path``, and ``index``.

        Raises:
            IndexError: If index is out of range.
            FileNotFoundError: If the image file does not exist.
            RuntimeError: If image loading or transform application fails.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        row = self.dataframe.iloc[index]
        raw_image_path = row[self.image_column]
        resolved_path = resolve_image_path(raw_image_path, self.paths)

        try:
            with Image.open(resolved_path) as image:
                image = image.copy()
        except UnidentifiedImageError as exc:
            logger.error("Corrupt or unsupported image: %s", resolved_path)
            raise RuntimeError(f"Failed to decode image: {resolved_path}") from exc
        except OSError as exc:
            logger.error("OS error while opening image: %s", resolved_path)
            raise RuntimeError(f"Failed to open image: {resolved_path}") from exc

        if self.transform is not None:
            try:
                image_tensor = self.transform(image)
            except Exception as exc:
                logger.exception("Transform failed for image: %s", resolved_path)
                raise RuntimeError(
                    f"Transform pipeline failed for image: {resolved_path}"
                ) from exc
        else:
            raise RuntimeError("Transform pipeline must be provided for ChestXrayDataset.")

        if not isinstance(image_tensor, torch.Tensor):
            raise RuntimeError(
                f"Transform must return torch.Tensor, got {type(image_tensor).__name__}."
            )

        labels_tensor = torch.from_numpy(self.labels_array[index])

        return {
            "image": image_tensor,
            "labels": labels_tensor,
            "image_path": str(resolved_path),
            "index": index,
        }

    def get_label_matrix(self) -> np.ndarray:
        """Return the full multi-label matrix for the dataset.

        Returns:
            Float32 array of shape ``(num_samples, num_classes)``.
        """
        return self.labels_array.copy()

    def get_class_frequencies(self) -> Dict[str, float]:
        """Compute positive label frequency for each disease class.

        Returns:
            Mapping from class name to positive sample ratio in ``[0, 1]``.
        """
        positive_counts = self.labels_array.sum(axis=0)
        total = len(self)
        return {
            label: float(count / total)
            for label, count in zip(self.label_columns, positive_counts)
        }


def compute_pos_weight(
    dataset: ChestXrayDataset,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-class positive weights for ``BCEWithLogitsLoss``.

    Uses the standard imbalance weighting:
    ``pos_weight[c] = num_negative / num_positive`` for each class ``c``.

    Args:
        dataset: Training dataset containing multi-hot labels.
        eps: Small constant to avoid division by zero for absent classes.

    Returns:
        Tensor of shape ``(num_classes,)`` with positive class weights.

    Raises:
        ValueError: If the dataset contains no samples.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot compute pos_weight from an empty dataset.")

    labels = dataset.get_label_matrix()
    positive_counts = labels.sum(axis=0)
    negative_counts = len(labels) - positive_counts
    weights = negative_counts / np.maximum(positive_counts, eps)
    pos_weight = torch.tensor(weights, dtype=torch.float32)

    logger.info(
        "Computed pos_weight for %d classes. Min=%.4f, Max=%.4f.",
        pos_weight.numel(),
        float(pos_weight.min()),
        float(pos_weight.max()),
    )
    return pos_weight


def build_weighted_sampler(dataset: ChestXrayDataset) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` to mitigate class imbalance.

    Sample weights increase for images containing rare positive labels.

    Args:
        dataset: Training dataset with multi-hot encoded labels.

    Returns:
        WeightedRandomSampler configured for the dataset size.

    Raises:
        ValueError: If the dataset is empty.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot build weighted sampler for an empty dataset.")

    labels = dataset.get_label_matrix()
    class_counts = labels.sum(axis=0)
    total_samples = len(labels)
    class_weights = total_samples / np.maximum(class_counts, 1.0)

    sample_weights = np.ones(len(labels), dtype=np.float64)
    for index, sample_labels in enumerate(labels):
        positive_indices = np.where(sample_labels > 0.0)[0]
        if positive_indices.size > 0:
            sample_weights[index] = float(class_weights[positive_indices].sum())
        else:
            sample_weights[index] = 1.0

    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(dataset),
        replacement=True,
    )
    logger.info("Built WeightedRandomSampler for %d training samples.", len(dataset))
    return sampler


def collate_fn(batch: Sequence[SampleDict]) -> SampleDict:
    """Collate a list of sample dictionaries into a batch dictionary.

    Args:
        batch: Sequence of samples returned by ``ChestXrayDataset.__getitem__``.

    Returns:
        Batch dictionary with stacked tensors and list metadata fields.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    images = torch.stack([sample["image"] for sample in batch], dim=0)
    labels = torch.stack([sample["labels"] for sample in batch], dim=0)
    image_paths = [str(sample["image_path"]) for sample in batch]
    indices = [int(sample["index"]) for sample in batch]

    return {
        "image": images,
        "labels": labels,
        "image_path": image_paths,
        "index": indices,
    }


def create_dataset(
    split: Union[SplitName, str],
    config: Optional[Config] = None,
    transform: Optional[TransformPipeline] = None,
    validate_images: bool = False,
) -> ChestXrayDataset:
    """Create a dataset instance for a specific split.

    Args:
        split: Dataset split identifier.
        config: Optional project configuration.
        transform: Optional transform pipeline override.
        validate_images: Verify image paths during initialization.

    Returns:
        Initialized ``ChestXrayDataset`` instance.
    """
    config = config or get_default_config()
    dataframe = load_split_dataframe(split=split, config=config)
    pipeline = transform or get_transforms(split=split, config=config)

    return ChestXrayDataset(
        dataframe=dataframe,
        transform=pipeline,
        paths=config.paths,
        image_column=config.paths.image_column,
        label_columns=config.paths.label_columns,
        validate_images=validate_images,
    )


def create_dataloader(
    split: Union[SplitName, str],
    config: Optional[Config] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    transform: Optional[TransformPipeline] = None,
    use_weighted_sampler: Optional[bool] = None,
) -> DataLoader:
    """Create a DataLoader for a specific dataset split.

    Args:
        split: Dataset split identifier.
        config: Optional project configuration.
        batch_size: Optional batch size override.
        shuffle: Whether to shuffle samples. Defaults to ``True`` for training only.
        transform: Optional transform pipeline override.
        use_weighted_sampler: Enable weighted sampling for training split.

    Returns:
        Configured PyTorch DataLoader.

    Raises:
        ValueError: If weighted sampling is requested for non-training splits.
    """
    config = config or get_default_config()
    split_value = split.value if isinstance(split, SplitName) else str(split).lower()
    is_train = split_value == SplitName.TRAIN.value

    if batch_size is None:
        batch_size = (
            config.training.batch_size
            if is_train
            else config.evaluation.batch_size
        )

    if shuffle is None:
        shuffle = is_train

    dataset = create_dataset(split=split, config=config, transform=transform)

    sampler: Optional[WeightedRandomSampler] = None
    enable_weighted_sampler = (
        config.data.use_weighted_sampler
        if use_weighted_sampler is None
        else use_weighted_sampler
    )

    if enable_weighted_sampler:
        if not is_train:
            raise ValueError("Weighted sampler is supported only for the training split.")
        sampler = build_weighted_sampler(dataset)
        shuffle = False

    num_workers = config.data.num_workers if is_train else config.evaluation.num_workers
    dataloader_kwargs: Dict[str, Union[int, bool, Callable[..., SampleDict]]] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": config.data.pin_memory,
        "drop_last": config.data.drop_last if is_train else False,
        "collate_fn": collate_fn,
    }

    if sampler is not None:
        dataloader_kwargs["shuffle"] = False
        dataloader_kwargs["sampler"] = sampler

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.data.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.data.prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    logger.info(
        "Created DataLoader for split='%s' with batch_size=%d, shuffle=%s, samples=%d.",
        split_value,
        batch_size,
        shuffle,
        len(dataset),
    )
    return dataloader


def get_dataloaders(config: Optional[Config] = None) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, validation, and test splits.

    Args:
        config: Optional project configuration.

    Returns:
        Dictionary mapping split names to DataLoader instances.
    """
    config = config or get_default_config()
    loaders = {
        SplitName.TRAIN.value: create_dataloader(SplitName.TRAIN, config=config),
        SplitName.VAL.value: create_dataloader(SplitName.VAL, config=config, shuffle=False),
        SplitName.TEST.value: create_dataloader(SplitName.TEST, config=config, shuffle=False),
    }
    logger.info("Initialized dataloaders for splits: %s", list(loaders.keys()))
    return loaders


class InferenceDataset(Dataset[torch.Tensor]):
    """Dataset for batch inference from a list of image file paths.

    Args:
        image_paths: Sequence of absolute or resolvable image paths.
        transform: Inference transform pipeline.
        paths: Path configuration for relative path resolution.
    """

    def __init__(
        self,
        image_paths: Sequence[Union[str, Path]],
        transform: Optional[TransformPipeline] = None,
        paths: Optional[PathConfig] = None,
    ) -> None:
        """Initialize the inference dataset."""
        if not image_paths:
            raise ValueError("image_paths must contain at least one path.")

        self.paths = paths or get_default_config().paths
        self.transform = transform or get_transforms(
            split=SplitName.TEST,
            config=get_default_config(),
        )
        self.image_paths: List[Path] = [
            resolve_image_path(path, self.paths) for path in image_paths
        ]

    def __len__(self) -> int:
        """Return the number of inference images."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load and transform a single inference image.

        Args:
            index: Image index.

        Returns:
            Transformed image tensor.

        Raises:
            IndexError: If index is out of range.
            RuntimeError: If image loading or preprocessing fails.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        resolved_path = self.image_paths[index]
        try:
            with Image.open(resolved_path) as image:
                image = image.copy()
            tensor = self.transform(image)
        except Exception as exc:
            logger.exception("Inference preprocessing failed for: %s", resolved_path)
            raise RuntimeError(
                f"Inference preprocessing failed for image: {resolved_path}"
            ) from exc

        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"Transform must return torch.Tensor, got {type(tensor).__name__}."
            )
        return tensor


def load_single_image(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
    transform: Optional[TransformPipeline] = None,
) -> torch.Tensor:
    """Load and preprocess a single image for inference.

    Args:
        image_path: Absolute or relative path to the image file.
        config: Optional project configuration.
        transform: Optional transform pipeline override.

    Returns:
        Preprocessed image tensor of shape ``(C, H, W)``.

    Raises:
        FileNotFoundError: If the image path cannot be resolved.
        RuntimeError: If image loading or preprocessing fails.
    """
    config = config or get_default_config()
    pipeline = transform or get_transforms(split=SplitName.TEST, config=config)
    dataset = InferenceDataset(
        image_paths=[image_path],
        transform=pipeline,
        paths=config.paths,
    )
    tensor = dataset[0]
    return tensor


def create_inference_dataloader(
    image_paths: Sequence[Union[str, Path]],
    config: Optional[Config] = None,
    batch_size: Optional[int] = None,
    transform: Optional[TransformPipeline] = None,
) -> DataLoader:
    """Create a DataLoader for batch inference over explicit image paths.

    Args:
        image_paths: Sequence of image paths to run inference on.
        config: Optional project configuration.
        batch_size: Optional batch size override.
        transform: Optional transform pipeline override.

    Returns:
        DataLoader yielding batches of preprocessed image tensors.
    """
    config = config or get_default_config()
    batch_size = batch_size or config.evaluation.batch_size
    dataset = InferenceDataset(
        image_paths=image_paths,
        transform=transform,
        paths=config.paths,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.evaluation.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )


def get_dataset_statistics(config: Optional[Config] = None) -> pd.DataFrame:
    """Compute per-split and per-class label statistics.

    Args:
        config: Optional project configuration.

    Returns:
        DataFrame summarizing sample counts and positive label ratios.
    """
    config = config or get_default_config()
    records: List[Dict[str, Union[str, float, int]]] = []

    for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
        try:
            dataset = create_dataset(split=split, config=config)
        except FileNotFoundError:
            logger.warning("Skipping statistics for missing split: %s", split.value)
            continue

        frequencies = dataset.get_class_frequencies()
        for class_name, frequency in frequencies.items():
            records.append(
                {
                    "split": split.value,
                    "class": class_name,
                    "positive_ratio": frequency,
                    "positive_count": int(frequency * len(dataset)),
                    "total_samples": len(dataset),
                }
            )

    stats_df = pd.DataFrame.from_records(records)
    logger.info("Computed dataset statistics for %d records.", len(stats_df))
    return stats_df
