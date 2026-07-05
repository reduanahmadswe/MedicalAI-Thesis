"""Central configuration module for the Chest X-ray multi-label classification pipeline.

This module defines all hyperparameters, paths, and runtime settings used across
training, evaluation, and explainability components. Configuration is organized
into focused dataclass groups to keep concerns separated and extensible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Google Drive root (Colab persistent storage)
# ---------------------------------------------------------------------------

GOOGLE_DRIVE_ROOT: Path = Path("/content/drive/MyDrive/MedicalAI-Thesis")

# Code location (may differ from Drive root when running in Colab)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class ModelName(str, Enum):
    """Supported backbone architectures for disease classification."""

    DENSENET121 = "densenet121"
    EFFICIENTNET_B0 = "efficientnet_b0"
    CONVNEXT_TINY = "convnext_tiny"


class LossName(str, Enum):
    """Supported loss functions for multi-label classification."""

    WEIGHTED_BCE = "weighted_bce"
    FOCAL = "focal"


class SplitName(str, Enum):
    """Dataset split identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# NIH ChestX-ray14 disease labels (14 pathologies + No Finding).
NIH_DISEASE_LABELS: Tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
)

NUM_CLASSES: int = len(NIH_DISEASE_LABELS)


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths for datasets, checkpoints, logs, and experiment outputs.

    All persistent artifacts are stored under Google Drive so training outputs
    survive Colab session restarts.

    Attributes:
        google_drive_root: Root Google Drive project folder.
        project_root: Local code/project directory.
        data_root: Directory containing train/val/test CSV splits and images.
        train_csv: Path to the training split CSV file.
        val_csv: Path to the validation split CSV file.
        test_csv: Path to the test split CSV file.
        image_column: Column name containing relative or absolute image paths.
        label_columns: Column names used as multi-label targets.
        checkpoint_dir: Directory for model checkpoints (``Models/``).
        results_dir: Directory for metrics, plots, and CSV exports (``Results/``).
        tensorboard_dir: Directory for TensorBoard event files.
        logs_dir: Directory for training log files (``Logs/``).
        roc_dir: Directory for ROC curve plots.
        confusion_matrix_dir: Directory for confusion matrix plots.
        gradcam_dir: Directory for Grad-CAM explainability outputs.
        best_model_filename: Filename for the best-performing checkpoint.
        last_model_filename: Filename for the most recent (last) checkpoint.
        epoch_checkpoint_prefix: Prefix for per-epoch checkpoint files.
    """

    google_drive_root: Path = GOOGLE_DRIVE_ROOT
    project_root: Path = PROJECT_ROOT

    data_root: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Dataset"
    )
    train_csv: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Dataset" / "train.csv"
    )
    val_csv: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Dataset" / "val.csv"
    )
    test_csv: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Dataset" / "test.csv"
    )
    image_column: str = "Image_Path"
    label_columns: Tuple[str, ...] = NIH_DISEASE_LABELS

    checkpoint_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Models"
    )
    results_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Results"
    )
    tensorboard_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "TensorBoard"
    )
    logs_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Logs"
    )
    roc_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Results" / "ROC"
    )
    confusion_matrix_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Results" / "ConfusionMatrix"
    )
    gradcam_dir: Path = field(
        default_factory=lambda: GOOGLE_DRIVE_ROOT / "Results" / "GradCAM"
    )

    best_model_filename: str = "best_model.pth"
    last_model_filename: str = "last_model.pth"
    epoch_checkpoint_prefix: str = "checkpoint_epoch_"

    def __post_init__(self) -> None:
        """Validate path-related configuration after initialization."""
        if not self.image_column.strip():
            raise ValueError("image_column must be a non-empty string.")
        if not self.label_columns:
            raise ValueError("label_columns must contain at least one label.")
        if len(set(self.label_columns)) != len(self.label_columns):
            raise ValueError("label_columns must not contain duplicate entries.")

    @property
    def best_model_path(self) -> Path:
        """Return the full path to the best model checkpoint."""
        return self.checkpoint_dir / self.best_model_filename

    @property
    def last_model_path(self) -> Path:
        """Return the full path to the last saved checkpoint."""
        return self.checkpoint_dir / self.last_model_filename

    @property
    def latest_checkpoint_path(self) -> Path:
        """Backward-compatible alias for ``last_model_path``."""
        return self.last_model_path

    def epoch_checkpoint_path(self, epoch: int) -> Path:
        """Return the checkpoint path for a specific 1-based epoch number.

        Args:
            epoch: One-based epoch index (e.g., 1, 2, 3).

        Returns:
            Path such as ``Models/checkpoint_epoch_001.pth``.
        """
        if epoch <= 0:
            raise ValueError("epoch must be a positive integer.")
        return self.checkpoint_dir / f"{self.epoch_checkpoint_prefix}{epoch:03d}.pth"

    @property
    def training_log_path(self) -> Path:
        """Return path to ``Results/training_log.csv``."""
        return self.results_dir / "training_log.csv"

    @property
    def training_summary_path(self) -> Path:
        """Return path to ``Results/training_summary.csv``."""
        return self.results_dir / "training_summary.csv"

    @property
    def metrics_csv_path(self) -> Path:
        """Return path to ``Results/metrics.csv``."""
        return self.results_dir / "metrics.csv"

    @property
    def predictions_csv_path(self) -> Path:
        """Return path to ``Results/predictions.csv``."""
        return self.results_dir / "predictions.csv"

    @property
    def thresholds_csv_path(self) -> Path:
        """Return path to ``Results/thresholds.csv``."""
        return self.results_dir / "thresholds.csv"

    @property
    def training_log_file_path(self) -> Path:
        """Return path to ``Logs/training.log``."""
        return self.logs_dir / "training.log"

    @property
    def config_json_path(self) -> Path:
        """Return path to ``Results/config.json``."""
        return self.results_dir / "config.json"

    def ensure_directories(self) -> None:
        """Create all Google Drive output directories if they do not exist."""
        for directory in (
            self.checkpoint_dir,
            self.results_dir,
            self.tensorboard_dir,
            self.logs_dir,
            self.roc_dir,
            self.confusion_matrix_dir,
            self.gradcam_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", directory)


@dataclass(frozen=True)
class DataConfig:
    """Data loading and preprocessing configuration.

    Attributes:
        image_size: Target spatial resolution (height, width) after resizing.
        num_workers: Number of subprocesses for DataLoader prefetching.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Keep worker processes alive between epochs.
        prefetch_factor: Number of batches loaded in advance by each worker.
        drop_last: Drop the last incomplete batch during training.
        use_weighted_sampler: Enable class-balanced sampling for training.
        normalize_mean: ImageNet normalization mean (RGB).
        normalize_std: ImageNet normalization standard deviation (RGB).
    """

    image_size: Tuple[int, int] = (224, 224)
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True
    use_weighted_sampler: bool = False
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        """Validate data configuration values."""
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError(
                f"image_size dimensions must be positive, got {self.image_size}."
            )
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1.")


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and initialization configuration.

    Attributes:
        model_name: Backbone architecture identifier.
        num_classes: Number of output logits (disease labels).
        pretrained: Load ImageNet-pretrained weights when available.
        dropout: Dropout probability applied before the classifier head.
        freeze_backbone: Freeze feature extractor weights during training.
    """

    model_name: ModelName = ModelName.DENSENET121
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        """Validate model configuration values."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")


@dataclass(frozen=True)
class LossConfig:
    """Loss function configuration for multi-label classification.

    Attributes:
        loss_name: Selected loss function identifier.
        pos_weight: Per-class positive weights for BCEWithLogitsLoss.
        focal_alpha: Balancing factor for focal loss (scalar or per-class).
        focal_gamma: Focusing parameter for focal loss.
        label_smoothing: Optional label smoothing factor in [0, 1).
        compute_class_weights: Automatically compute pos_weight from training data.
    """

    loss_name: LossName = LossName.WEIGHTED_BCE
    pos_weight: Optional[Tuple[float, ...]] = None
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    compute_class_weights: bool = True

    def __post_init__(self) -> None:
        """Validate loss configuration values."""
        if self.pos_weight is not None and len(self.pos_weight) != NUM_CLASSES:
            raise ValueError(
                f"pos_weight length ({len(self.pos_weight)}) must match "
                f"NUM_CLASSES ({NUM_CLASSES})."
            )
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0.0, 1.0).")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be >= 0.")
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError("focal_alpha must be in the range [0.0, 1.0].")


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters.

    Attributes:
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularization coefficient.
        betas: AdamW beta coefficients (beta1, beta2).
        eps: AdamW numerical stability epsilon.
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Validate optimizer hyperparameters."""
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0.")


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning rate scheduler configuration.

    Attributes:
        use_scheduler: Enable CosineAnnealingLR during training.
        t_max: Maximum number of iterations (typically num_epochs).
        eta_min: Minimum learning rate reached at the end of cosine annealing.
    """

    use_scheduler: bool = True
    t_max: int = 50
    eta_min: float = 1e-6

    def __post_init__(self) -> None:
        """Validate scheduler hyperparameters."""
        if self.t_max <= 0:
            raise ValueError("t_max must be a positive integer.")
        if self.eta_min < 0.0:
            raise ValueError("eta_min must be >= 0.")


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop and checkpointing configuration.

    Attributes:
        num_epochs: Total number of training epochs.
        batch_size: Mini-batch size per optimization step.
        gradient_clip_norm: Maximum gradient norm for clipping (0 disables).
        use_amp: Enable automatic mixed precision on CUDA devices.
        early_stopping: Enable early stopping based on validation metric.
        early_stopping_patience: Epochs to wait before stopping without improvement.
        early_stopping_min_delta: Minimum improvement to reset patience counter.
        early_stopping_monitor: Metric name monitored for early stopping.
        early_stopping_mode: Optimization direction ('min' or 'max').
        save_best_only: Save only the best checkpoint (always saves last model).
        save_epoch_checkpoints: Save ``checkpoint_epoch_XXX.pth`` every epoch.
        resume_training: Deprecated; use ``Trainer.fit(resume=True)`` instead.
        log_interval: Log training metrics every N batches.
        eval_interval: Run validation every N epochs (1 = every epoch).
        threshold: Default decision threshold for multi-label predictions.
        optimize_thresholds_on_val: Search per-class thresholds on validation each epoch.
        save_val_roc_each_epoch: Save validation ROC curves after every epoch.
        save_val_confusion_matrix_each_epoch: Save validation confusion matrices each epoch.
        save_val_classification_report_each_epoch: Save classification report each epoch.
        random_seed: Global random seed for reproducibility.
        deterministic: Enable deterministic algorithms where supported.
    """

    num_epochs: int = 50
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    use_amp: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "auroc"
    early_stopping_mode: str = "max"
    save_best_only: bool = False
    save_epoch_checkpoints: bool = True
    resume_training: bool = False
    log_interval: int = 50
    eval_interval: int = 1
    threshold: float = 0.5
    optimize_thresholds_on_val: bool = True
    save_val_roc_each_epoch: bool = False
    save_val_confusion_matrix_each_epoch: bool = True
    save_val_classification_report_each_epoch: bool = True
    random_seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        """Validate training configuration values."""
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.gradient_clip_norm < 0.0:
            raise ValueError("gradient_clip_norm must be >= 0.")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be > 0.")
        if self.early_stopping_mode not in {"min", "max"}:
            raise ValueError("early_stopping_mode must be 'min' or 'max'.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in the range [0.0, 1.0].")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be > 0.")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0.")


@dataclass(frozen=True)
class DeviceConfig:
    """Compute device and distributed training configuration.

    Attributes:
        device: Target device string ('cuda', 'cpu', or 'cuda:N').
        cuda_visible_devices: Optional CUDA device visibility override.
        multi_gpu: Enable DataParallel when multiple GPUs are available.
    """

    device: str = "cuda"
    cuda_visible_devices: Optional[str] = None
    multi_gpu: bool = True

    def __post_init__(self) -> None:
        """Validate device configuration."""
        valid_prefixes = ("cuda", "cpu")
        if not any(self.device.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"device must start with one of {valid_prefixes}, got '{self.device}'."
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and inference configuration.

    Attributes:
        batch_size: Batch size for validation, testing, and batch inference.
        num_workers: DataLoader workers for evaluation pipelines.
        threshold: Default classification threshold for metric computation.
        optimize_threshold: Search per-class thresholds on validation set.
        threshold_search_min: Lower bound for threshold search grid.
        threshold_search_max: Upper bound for threshold search grid.
        threshold_search_steps: Number of threshold grid points.
        save_predictions: Export raw predictions to CSV.
        save_roc_curves: Export ROC curve plots.
        save_pr_curves: Export precision-recall curve plots.
        save_confusion_matrices: Export confusion matrix plots.
        save_classification_report: Export sklearn-style classification report.
    """

    batch_size: int = 32
    num_workers: int = 4
    threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_search_min: float = 0.05
    threshold_search_max: float = 0.95
    threshold_search_steps: int = 19
    save_predictions: bool = True
    save_roc_curves: bool = True
    save_pr_curves: bool = True
    save_confusion_matrices: bool = True
    save_classification_report: bool = True

    def __post_init__(self) -> None:
        """Validate evaluation configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in the range [0.0, 1.0].")
        if self.threshold_search_min >= self.threshold_search_max:
            raise ValueError(
                "threshold_search_min must be less than threshold_search_max."
            )
        if self.threshold_search_steps <= 0:
            raise ValueError("threshold_search_steps must be > 0.")


@dataclass(frozen=True)
class GradCAMConfig:
    """Grad-CAM and explainability configuration.

    Attributes:
        method: Explainability method ('gradcam' or 'gradcam++').
        target_layer_name: Optional explicit target layer name for CAM hooks.
        use_predicted_class: Use argmax predicted class as CAM target.
        target_class_index: Fixed class index for CAM (overrides prediction).
        colormap: Matplotlib colormap name for heatmap visualization.
        alpha: Blending weight for heatmap overlay on original image.
        output_subdir: Optional subdirectory under ``gradcam_dir`` for CAM artifacts.
    """

    method: str = "gradcam"
    target_layer_name: Optional[str] = None
    use_predicted_class: bool = True
    target_class_index: Optional[int] = None
    colormap: str = "jet"
    alpha: float = 0.45
    output_subdir: str = "explainability"

    def __post_init__(self) -> None:
        """Validate Grad-CAM configuration values."""
        valid_methods = {"gradcam", "gradcam++"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{self.method}'."
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in the range [0.0, 1.0].")
        if self.target_class_index is not None and not (
            0 <= self.target_class_index < NUM_CLASSES
        ):
            raise ValueError(
                f"target_class_index must be in [0, {NUM_CLASSES - 1}] or None."
            )


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and experiment tracking configuration.

    Attributes:
        experiment_name: Human-readable experiment identifier.
        log_level: Python logging level name.
        log_to_file: Persist logs to disk under ``Logs/`` on Google Drive.
        log_filename: Log file name when log_to_file is enabled.
        tensorboard_enabled: Enable TensorBoard scalar and image logging.
        log_gpu_memory: Log peak GPU memory usage during training.
        log_learning_rate: Log learning rate at each epoch.
        log_training_time: Log per-epoch and total training duration.
    """

    experiment_name: str = "chestxray_multilabel"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "training.log"
    tensorboard_enabled: bool = True
    log_gpu_memory: bool = True
    log_learning_rate: bool = True
    log_training_time: bool = True

    def __post_init__(self) -> None:
        """Validate logging configuration values."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got '{self.log_level}'."
            )


@dataclass
class Config:
    """Aggregate configuration container for the full research pipeline.

    This class composes all sub-configurations and exposes convenience helpers
    for path creation, serialization, and runtime validation.

    Attributes:
        paths: Filesystem and dataset path settings.
        data: DataLoader and preprocessing settings.
        model: Model architecture settings.
        loss: Loss function settings.
        optimizer: Optimizer hyperparameters.
        scheduler: Learning rate scheduler settings.
        training: Training loop and checkpointing settings.
        device: Compute device settings.
        evaluation: Evaluation and inference settings.
        gradcam: Explainability settings.
        logging: Logging and TensorBoard settings.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    gradcam: GradCAMConfig = field(default_factory=GradCAMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def validate(self) -> None:
        """Validate cross-field consistency across configuration groups.

        Raises:
            FileNotFoundError: If required split CSV files are missing.
            ValueError: If dependent fields are inconsistent.
        """
        missing_csvs: List[Path] = []
        for csv_path, split_name in (
            (self.paths.train_csv, SplitName.TRAIN.value),
            (self.paths.val_csv, SplitName.VAL.value),
            (self.paths.test_csv, SplitName.TEST.value),
        ):
            if not csv_path.exists():
                missing_csvs.append(csv_path)
                logger.warning("Missing %s split CSV: %s", split_name, csv_path)

        if missing_csvs:
            logger.warning(
                "%d split CSV file(s) not found. Training will fail until data is "
                "placed under the configured paths.",
                len(missing_csvs),
            )

        if self.model.num_classes != len(self.paths.label_columns):
            raise ValueError(
                "model.num_classes must match len(paths.label_columns): "
                f"{self.model.num_classes} != {len(self.paths.label_columns)}."
            )

        if self.scheduler.t_max != self.training.num_epochs:
            logger.info(
                "scheduler.t_max (%d) differs from training.num_epochs (%d). "
                "Cosine annealing will use t_max as configured.",
                self.scheduler.t_max,
                self.training.num_epochs,
            )

    def setup(self) -> None:
        """Prepare runtime environment directories and validate configuration."""
        self.paths.ensure_directories()
        self.validate()
        logger.info(
            "Configuration initialized for experiment '%s'.",
            self.logging.experiment_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a nested dictionary.

        Returns:
            Dictionary representation suitable for logging or JSON export.
        """
        return {
            "paths": _dataclass_to_dict(self.paths),
            "data": _dataclass_to_dict(self.data),
            "model": _dataclass_to_dict(self.model),
            "loss": _dataclass_to_dict(self.loss),
            "optimizer": _dataclass_to_dict(self.optimizer),
            "scheduler": _dataclass_to_dict(self.scheduler),
            "training": _dataclass_to_dict(self.training),
            "device": _dataclass_to_dict(self.device),
            "evaluation": _dataclass_to_dict(self.evaluation),
            "gradcam": _dataclass_to_dict(self.gradcam),
            "logging": _dataclass_to_dict(self.logging),
        }


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a JSON-serializable dictionary.

    Args:
        obj: Dataclass instance to convert.

    Returns:
        Dictionary with Path and Enum values converted to plain Python types.

    Raises:
        TypeError: If the input object is not a dataclass instance.
    """
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj).__name__}.")

    raw: Dict[str, Any] = asdict(obj)
    return _normalize_config_values(raw)


def _normalize_config_values(value: Any) -> Any:
    """Recursively normalize config values for serialization.

    Args:
        value: Arbitrary configuration value or nested structure.

    Returns:
        Normalized value with Path and Enum instances converted to strings.
    """
    if isinstance(value, dict):
        return {key: _normalize_config_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config_values(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def get_default_config() -> Config:
    """Create and return the default project configuration.

    Returns:
        Fully initialized Config instance with default hyperparameters.
    """
    config = Config()
    return config


def build_config(**overrides: Any) -> Config:
    """Build a Config instance with optional top-level group overrides.

    Example:
        config = build_config(
            model=ModelConfig(model_name=ModelName.EFFICIENTNET_B0),
            training=TrainingConfig(num_epochs=30, batch_size=16),
        )

    Args:
        **overrides: Keyword arguments matching Config field names.

    Returns:
        Config instance with selected groups replaced.

    Raises:
        TypeError: If an override value has an invalid type for its field.
        ValueError: If an unknown override keyword is provided.
    """
    from dataclasses import fields

    config = get_default_config()
    valid_fields = {field_info.name for field_info in fields(Config)}

    unknown_keys = set(overrides) - valid_fields
    if unknown_keys:
        raise ValueError(
            f"Unknown configuration overrides: {sorted(unknown_keys)}. "
            f"Valid keys: {sorted(valid_fields)}."
        )

    for key, value in overrides.items():
        expected_type = type(getattr(config, key))
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Override for '{key}' must be of type {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        setattr(config, key, value)

    return config


# Default singleton used by training and evaluation entry points.
DEFAULT_CONFIG: Config = get_default_config()
