"""Evaluation metrics, threshold search, and visualization for multi-label CXR models.

Computes classification metrics (loss, accuracy, precision, recall, F1, AUROC),
searches per-class decision thresholds, and generates publication-ready plots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor

from src.config import (
    Config,
    EvaluationConfig,
    NIH_DISEASE_LABELS,
    NUM_CLASSES,
    get_default_config,
)


logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Tensor]


@dataclass
class MetricsResult:
    """Container for computed multi-label evaluation metrics.

    Attributes:
        loss: Mean loss value when provided by the caller.
        accuracy: Subset (exact-match) accuracy.
        hamming_accuracy: Per-label accuracy (1 - Hamming loss).
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        precision_micro: Micro-averaged precision.
        recall_micro: Micro-averaged recall.
        f1_micro: Micro-averaged F1 score.
        auroc_macro: Macro-averaged AUROC.
        auroc_micro: Micro-averaged AUROC.
        precision_per_class: Precision for each disease class.
        recall_per_class: Recall for each disease class.
        f1_per_class: F1 score for each disease class.
        auroc_per_class: AUROC for each disease class.
        support_per_class: Number of positive samples per class.
        confusion_matrices: Per-class 2x2 confusion matrices (TN, FP, FN, TP).
        thresholds: Decision thresholds used for binarization.
        best_thresholds: Optional optimized thresholds per class.
        num_samples: Number of evaluated samples.
    """

    loss: Optional[float] = None
    accuracy: float = 0.0
    hamming_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    precision_micro: float = 0.0
    recall_micro: float = 0.0
    f1_micro: float = 0.0
    auroc_macro: float = 0.0
    auroc_micro: float = 0.0
    precision_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES))
    recall_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES))
    f1_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES))
    auroc_per_class: np.ndarray = field(default_factory=lambda: np.full(NUM_CLASSES, np.nan))
    support_per_class: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CLASSES, dtype=int))
    confusion_matrices: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_CLASSES, 2, 2), dtype=int)
    )
    thresholds: Union[float, np.ndarray] = 0.5
    best_thresholds: Optional[np.ndarray] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert scalar metrics to a flat dictionary for CSV export.

        Returns:
            Dictionary of aggregate metric names and values.
        """
        metrics = {
            "loss": self.loss if self.loss is not None else float("nan"),
            "accuracy": self.accuracy,
            "hamming_accuracy": self.hamming_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_micro": self.precision_micro,
            "recall_micro": self.recall_micro,
            "f1_micro": self.f1_micro,
            "auroc_macro": self.auroc_macro,
            "auroc_micro": self.auroc_micro,
            "num_samples": float(self.num_samples),
        }
        return metrics

    def to_dataframe(self, class_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Convert per-class metrics to a pandas DataFrame.

        Args:
            class_names: Optional class label names.

        Returns:
            DataFrame with one row per disease class.
        """
        names = list(class_names or NIH_DISEASE_LABELS)
        return pd.DataFrame(
            {
                "class": names,
                "precision": self.precision_per_class,
                "recall": self.recall_per_class,
                "f1": self.f1_per_class,
                "auroc": self.auroc_per_class,
                "support": self.support_per_class,
                "threshold": (
                    self.best_thresholds
                    if self.best_thresholds is not None
                    else np.full(NUM_CLASSES, self.thresholds)
                    if isinstance(self.thresholds, float)
                    else self.thresholds
                ),
            }
        )


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert torch tensors or sequences to a float numpy array.

    Args:
        array: Input array-like object.

    Returns:
        NumPy array.

    Raises:
        TypeError: If the input type is unsupported.
        ValueError: If the resulting array is empty.
    """
    if isinstance(array, Tensor):
        array = array.detach().cpu().numpy()
    elif not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if array.size == 0:
        raise ValueError("Input array must not be empty.")

    return array.astype(np.float32, copy=False)


def _validate_shapes(y_true: ArrayLike, y_probs: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and align ground-truth and prediction arrays.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.

    Returns:
        Tuple of validated numpy arrays.

    Raises:
        ValueError: If shapes are invalid or mismatched.
    """
    labels = _to_numpy(y_true)
    probabilities = _to_numpy(y_probs)

    if labels.ndim != 2 or probabilities.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays of shape (N, C), got labels={labels.shape}, "
            f"probs={probabilities.shape}."
        )
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"Label and probability shapes must match, got {labels.shape} vs "
            f"{probabilities.shape}."
        )
    if labels.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, got {labels.shape[1]}."
        )

    labels = np.clip(labels, 0.0, 1.0)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return labels, probabilities


def _normalize_thresholds(threshold: Union[float, Sequence[float], np.ndarray]) -> np.ndarray:
    """Normalize threshold input to a per-class numpy vector.

    Args:
        threshold: Scalar or per-class threshold values.

    Returns:
        Threshold array of shape ``(C,)``.

    Raises:
        ValueError: If threshold length or values are invalid.
    """
    if isinstance(threshold, (float, int)):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Threshold must be in [0, 1].")
        return np.full(NUM_CLASSES, float(threshold), dtype=np.float32)

    threshold_array = np.asarray(threshold, dtype=np.float32).reshape(-1)
    if threshold_array.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Per-class threshold length must be {NUM_CLASSES}, got {threshold_array.shape[0]}."
        )
    if np.any((threshold_array < 0.0) | (threshold_array > 1.0)):
        raise ValueError("All thresholds must be in [0, 1].")
    return threshold_array


def binarize_predictions(
    y_probs: ArrayLike,
    threshold: Union[float, Sequence[float], np.ndarray] = 0.5,
) -> np.ndarray:
    """Convert predicted probabilities to binary multi-label predictions.

    Args:
        y_probs: Predicted probabilities of shape ``(N, C)``.
        threshold: Scalar or per-class threshold values.

    Returns:
        Binary prediction array of shape ``(N, C)``.
    """
    _, probabilities = _validate_shapes(np.zeros_like(_to_numpy(y_probs)), y_probs)
    thresholds = _normalize_thresholds(threshold)
    return (probabilities >= thresholds.reshape(1, -1)).astype(np.int32)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC with graceful handling of single-class edge cases.

    Args:
        y_true: Binary ground-truth vector.
        y_score: Predicted scores/probabilities.

    Returns:
        ROC AUC value or ``nan`` when undefined.
    """
    unique_values = np.unique(y_true)
    if unique_values.size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_confusion_matrices(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> np.ndarray:
    """Compute per-class binary confusion matrices for multi-label outputs.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_pred: Binary predictions of shape ``(N, C)``.

    Returns:
        Array of shape ``(C, 2, 2)`` with [[TN, FP], [FN, TP]] per class.
    """
    labels, _ = _validate_shapes(y_true, y_true)
    predictions = _to_numpy(y_pred)
    if predictions.shape != labels.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    matrices = np.zeros((labels.shape[1], 2, 2), dtype=int)
    for class_index in range(labels.shape[1]):
        matrices[class_index] = confusion_matrix(
            labels[:, class_index],
            predictions[:, class_index],
            labels=[0, 1],
        )
    return matrices


def search_best_thresholds(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    config: Optional[EvaluationConfig] = None,
    metric: str = "f1",
) -> np.ndarray:
    """Search per-class thresholds that maximize a selected metric on validation data.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        config: Optional evaluation configuration for threshold grid bounds.
        metric: Metric to optimize ('f1', 'precision', or 'recall').

    Returns:
        Best threshold array of shape ``(C,)``.

    Raises:
        ValueError: If the metric name is unsupported.
    """
    if metric not in {"f1", "precision", "recall"}:
        raise ValueError("metric must be one of {'f1', 'precision', 'recall'}.")

    config = config or get_default_config().evaluation
    labels, probabilities = _validate_shapes(y_true, y_probs)

    threshold_grid = np.linspace(
        config.threshold_search_min,
        config.threshold_search_max,
        config.threshold_search_steps,
        dtype=np.float32,
    )

    best_thresholds = np.full(labels.shape[1], config.threshold, dtype=np.float32)
    for class_index in range(labels.shape[1]):
        class_labels = labels[:, class_index]
        class_probs = probabilities[:, class_index]

        if np.unique(class_labels).size < 2:
            logger.warning(
                "Skipping threshold search for class index %d due to single-class labels.",
                class_index,
            )
            continue

        best_score = -1.0
        best_threshold = config.threshold
        for threshold in threshold_grid:
            class_predictions = (class_probs >= threshold).astype(int)
            if metric == "f1":
                score = f1_score(class_labels, class_predictions, zero_division=0)
            elif metric == "precision":
                score = precision_score(class_labels, class_predictions, zero_division=0)
            else:
                score = recall_score(class_labels, class_predictions, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        best_thresholds[class_index] = best_threshold

    logger.info(
        "Optimized per-class thresholds on validation set using metric='%s'.",
        metric,
    )
    return best_thresholds


def compute_metrics(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    threshold: Union[float, Sequence[float], np.ndarray] = 0.5,
    loss: Optional[float] = None,
    class_names: Optional[Sequence[str]] = None,
) -> MetricsResult:
    """Compute comprehensive multi-label metrics.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        threshold: Scalar or per-class decision thresholds.
        loss: Optional mean loss to include in the result.
        class_names: Optional class names (reserved for future metadata export).

    Returns:
        ``MetricsResult`` containing aggregate and per-class metrics.
    """
    labels, probabilities = _validate_shapes(y_true, y_probs)
    thresholds = _normalize_thresholds(threshold)

    subset_accuracy = float(accuracy_score(labels, predictions))
    hamming_accuracy = float(np.mean(labels == predictions))

    precision_macro = float(precision_score(labels, predictions, average="macro", zero_division=0))
    recall_macro = float(recall_score(labels, predictions, average="macro", zero_division=0))
    f1_macro = float(f1_score(labels, predictions, average="macro", zero_division=0))
    precision_micro = float(precision_score(labels, predictions, average="micro", zero_division=0))
    recall_micro = float(recall_score(labels, predictions, average="micro", zero_division=0))
    f1_micro = float(f1_score(labels, predictions, average="micro", zero_division=0))

    auroc_per_class = np.array(
        [_safe_roc_auc(labels[:, index], probabilities[:, index]) for index in range(labels.shape[1])],
        dtype=np.float32,
    )
    valid_auroc = auroc_per_class[~np.isnan(auroc_per_class)]
    auroc_macro = float(np.mean(valid_auroc)) if valid_auroc.size > 0 else float("nan")

    try:
        auroc_micro = float(roc_auc_score(labels, probabilities, average="micro"))
    except ValueError:
        auroc_micro = float("nan")

    precision_per_class = precision_score(
        labels, predictions, average=None, zero_division=0
    ).astype(np.float32)
    recall_per_class = recall_score(
        labels, predictions, average=None, zero_division=0
    ).astype(np.float32)
    f1_per_class = f1_score(
        labels, predictions, average=None, zero_division=0
    ).astype(np.float32)
    support_per_class = labels.sum(axis=0).astype(int)
    confusion_matrices = compute_confusion_matrices(labels, predictions)

    result = MetricsResult(
        loss=loss,
        accuracy=subset_accuracy,
        hamming_accuracy=hamming_accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        f1_micro=f1_micro,
        auroc_macro=auroc_macro,
        auroc_micro=auroc_micro,
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        auroc_per_class=auroc_per_class,
        support_per_class=support_per_class,
        confusion_matrices=confusion_matrices,
        thresholds=thresholds,
        num_samples=num_samples,
    )

    logger.info(
        "Computed metrics on %d samples. macro_auroc=%.4f, micro_auroc=%.4f, f1_macro=%.4f.",
        num_samples,
        auroc_macro,
        auroc_micro,
        f1_macro,
    )
    return result


class MetricsAccumulator:
    """Online accumulator for logits, labels, and losses across evaluation batches.

    Args:
        num_classes: Number of disease classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        """Initialize an empty metrics accumulator."""
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated predictions and losses."""
        self._labels: List[np.ndarray] = []
        self._probabilities: List[np.ndarray] = []
        self._losses: List[float] = []

    def update(
        self,
        logits: ArrayLike,
        labels: ArrayLike,
        loss: Optional[float] = None,
    ) -> None:
        """Append a batch of predictions and labels.

        Args:
            logits: Model logits of shape ``(N, C)``.
            labels: Ground-truth labels of shape ``(N, C)``.
            loss: Optional scalar batch loss.

        Raises:
            ValueError: If batch shapes are invalid.
        """
        if isinstance(logits, Tensor):
            batch_probabilities = torch.sigmoid(logits.detach()).cpu().numpy()
        else:
            logits_array = _to_numpy(logits)
            batch_probabilities = 1.0 / (1.0 + np.exp(-logits_array))

        batch_labels = _to_numpy(labels)
        batch_labels, batch_probabilities = _validate_shapes(batch_labels, batch_probabilities)

        if batch_probabilities.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, got {batch_probabilities.shape[1]}."
            )

        self._labels.append(batch_labels)
        self._probabilities.append(batch_probabilities)
        if loss is not None:
            self._losses.append(float(loss))

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """Return concatenated labels, probabilities, and mean loss.

        Returns:
            Tuple of ``(y_true, y_probs, mean_loss)``.

        Raises:
            RuntimeError: If no batches have been accumulated.
        """
        if not self._labels:
            raise RuntimeError("MetricsAccumulator is empty. Call update() before compute().")

        labels = np.concatenate(self._labels, axis=0)
        probabilities = np.concatenate(self._probabilities, axis=0)
        mean_loss = float(np.mean(self._losses)) if self._losses else None
        return labels, probabilities, mean_loss

    def compute(
        self,
        threshold: Union[float, Sequence[float], np.ndarray] = 0.5,
    ) -> MetricsResult:
        """Compute metrics from accumulated batches.

        Args:
            threshold: Scalar or per-class decision thresholds.

        Returns:
            Computed ``MetricsResult``.
        """
        labels, probabilities, mean_loss = self.get_arrays()
        return compute_metrics(
            y_true=labels,
            y_probs=probabilities,
            threshold=threshold,
            loss=mean_loss,
        )


def save_metrics_csv(
    metrics: MetricsResult,
    output_path: Union[str, Path],
    split_name: str = "validation",
) -> None:
    """Save aggregate and per-class metrics to CSV files.

    Writes:
    - ``{split_name}_metrics.csv`` for aggregate metrics
    - ``{split_name}_per_class_metrics.csv`` for per-class metrics

    Args:
        metrics: Computed metrics result.
        output_path: Directory or file prefix for CSV export.
        split_name: Split identifier used in filenames.

    Raises:
        OSError: If writing files fails.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        output_dir = output_path.parent
        prefix = output_path.stem
    else:
        output_dir = output_path
        prefix = split_name

    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_file = output_dir / f"{prefix}_metrics.csv"
    per_class_file = output_dir / f"{prefix}_per_class_metrics.csv"

    try:
        pd.DataFrame([metrics.to_dict()]).to_csv(aggregate_file, index=False)
        metrics.to_dataframe().to_csv(per_class_file, index=False)
    except Exception as exc:
        logger.exception("Failed to save metrics CSV files to %s.", output_dir)
        raise OSError(f"Unable to save metrics CSV files to {output_dir}") from exc

    logger.info("Saved metrics CSV files to %s.", output_dir)


def save_predictions_csv(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    image_paths: Sequence[str],
    output_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    threshold: Union[float, Sequence[float], np.ndarray] = 0.5,
) -> None:
    """Export ground-truth labels, probabilities, and binary predictions to CSV.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        image_paths: Sequence of image path strings of length ``N``.
        output_path: Destination CSV file path.
        class_names: Optional class names used as column prefixes.
        threshold: Decision threshold(s) for binary predictions.

    Raises:
        ValueError: If ``image_paths`` length does not match sample count.
        OSError: If writing the CSV fails.
    """
    labels, probabilities = _validate_shapes(y_true, y_probs)
    if len(image_paths) != labels.shape[0]:
        raise ValueError(
            f"image_paths length ({len(image_paths)}) must match number of samples "
            f"({labels.shape[0]})."
        )

    names = list(class_names or NIH_DISEASE_LABELS)
    predictions = binarize_predictions(probabilities, threshold=threshold)

    records: Dict[str, List[Union[str, float, int]]] = {"image_path": list(image_paths)}
    for index, class_name in enumerate(names):
        records[f"{class_name}_true"] = labels[:, index].astype(int).tolist()
        records[f"{class_name}_prob"] = probabilities[:, index].astype(float).tolist()
        records[f"{class_name}_pred"] = predictions[:, index].astype(int).tolist()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pd.DataFrame.from_dict(records).to_csv(output_path, index=False)
    except Exception as exc:
        logger.exception("Failed to save predictions CSV to %s.", output_path)
        raise OSError(f"Unable to save predictions CSV to {output_path}") from exc

    logger.info("Saved predictions CSV to %s.", output_path)


def save_best_thresholds_csv(
    thresholds: ArrayLike,
    output_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """Save optimized per-class thresholds to CSV.

    Args:
        thresholds: Threshold array of shape ``(C,)``.
        output_path: Destination CSV file path.
        class_names: Optional class names.

    Raises:
        OSError: If writing the CSV fails.
    """
    threshold_array = _normalize_thresholds(thresholds)
    names = list(class_names or NIH_DISEASE_LABELS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        pd.DataFrame({"class": names, "threshold": threshold_array}).to_csv(
            output_path, index=False
        )
    except Exception as exc:
        logger.exception("Failed to save thresholds CSV to %s.", output_path)
        raise OSError(f"Unable to save thresholds CSV to {output_path}") from exc

    logger.info("Saved best thresholds CSV to %s.", output_path)


def plot_roc_curves(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    output_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    title: str = "ROC Curves",
) -> None:
    """Plot and save per-class ROC curves.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        output_path: Destination image file path.
        class_names: Optional class names for legend labels.
        title: Plot title.

    Raises:
        OSError: If saving the figure fails.
    """
    labels, probabilities = _validate_shapes(y_true, y_probs)
    names = list(class_names or NIH_DISEASE_LABELS)

    fig, axis = plt.subplots(figsize=(10, 8))
    try:
        for index, class_name in enumerate(names):
            if np.unique(labels[:, index]).size < 2:
                continue
            false_positive_rate, true_positive_rate, _ = roc_curve(
                labels[:, index], probabilities[:, index]
            )
            roc_auc = auc(false_positive_rate, true_positive_rate)
            axis.plot(
                false_positive_rate,
                true_positive_rate,
                label=f"{class_name} (AUC={roc_auc:.3f})",
                linewidth=1.5,
            )

        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title(title)
        axis.legend(loc="lower right", fontsize=8)
        axis.grid(alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        logger.exception("Failed to plot ROC curves.")
        raise OSError("Unable to save ROC curve plot.") from exc
    finally:
        plt.close(fig)

    logger.info("Saved ROC curves to %s.", output_path)


def plot_precision_recall_curves(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    output_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    title: str = "Precision-Recall Curves",
) -> None:
    """Plot and save per-class precision-recall curves.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        output_path: Destination image file path.
        class_names: Optional class names for legend labels.
        title: Plot title.

    Raises:
        OSError: If saving the figure fails.
    """
    labels, probabilities = _validate_shapes(y_true, y_probs)
    names = list(class_names or NIH_DISEASE_LABELS)

    fig, axis = plt.subplots(figsize=(10, 8))
    try:
        for index, class_name in enumerate(names):
            if labels[:, index].sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(
                labels[:, index], probabilities[:, index]
            )
            axis.plot(recall, precision, label=class_name, linewidth=1.5)

        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(title)
        axis.legend(loc="lower left", fontsize=8)
        axis.grid(alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        logger.exception("Failed to plot precision-recall curves.")
        raise OSError("Unable to save precision-recall curve plot.") from exc
    finally:
        plt.close(fig)

    logger.info("Saved precision-recall curves to %s.", output_path)


def plot_confusion_matrices(
    confusion_matrices: ArrayLike,
    output_path: Union[str, Path],
    class_names: Optional[Sequence[str]] = None,
    title: str = "Per-Class Confusion Matrices",
) -> None:
    """Plot and save a grid of per-class confusion matrices.

    Args:
        confusion_matrices: Array of shape ``(C, 2, 2)``.
        output_path: Destination image file path.
        class_names: Optional class names for subplot titles.
        title: Figure title.

    Raises:
        ValueError: If confusion matrix shape is invalid.
        OSError: If saving the figure fails.
    """
    matrices = np.asarray(confusion_matrices)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
        raise ValueError(
            f"confusion_matrices must have shape (C, 2, 2), got {matrices.shape}."
        )

    names = list(class_names or NIH_DISEASE_LABELS)
    num_classes = matrices.shape[0]
    columns = 5
    rows = int(np.ceil(num_classes / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    fig.suptitle(title, fontsize=14)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)

    try:
        for index in range(rows * columns):
            row, column = divmod(index, columns)
            axis = axes_array[row, column]
            if index >= num_classes:
                axis.axis("off")
                continue

            matrix = matrices[index]
            im = axis.imshow(matrix, cmap="Blues")
            axis.set_title(names[index], fontsize=9)
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            axis.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
            axis.set_yticklabels(["True 0", "True 1"], fontsize=7)

            for row_index in range(2):
                for col_index in range(2):
                    axis.text(
                        col_index,
                        row_index,
                        int(matrix[row_index, col_index]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        logger.exception("Failed to plot confusion matrices.")
        raise OSError("Unable to save confusion matrix plot.") from exc
    finally:
        plt.close(fig)

    logger.info("Saved confusion matrices to %s.", output_path)


def plot_loss_curve(
    history: pd.DataFrame,
    output_path: Union[str, Path],
    train_column: str = "train_loss",
    val_column: str = "val_loss",
    title: str = "Loss Curves",
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: DataFrame containing epoch-wise loss values.
        output_path: Destination image file path.
        train_column: Column name for training loss.
        val_column: Column name for validation loss.
        title: Plot title.

    Raises:
        ValueError: If required columns are missing.
        OSError: If saving the figure fails.
    """
    for column in (train_column, val_column):
        if column not in history.columns:
            raise ValueError(f"history DataFrame missing required column '{column}'.")

    fig, axis = plt.subplots(figsize=(8, 5))
    try:
        axis.plot(history[train_column], label="Train Loss", linewidth=2.0)
        axis.plot(history[val_column], label="Validation Loss", linewidth=2.0)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        logger.exception("Failed to plot loss curves.")
        raise OSError("Unable to save loss curve plot.") from exc
    finally:
        plt.close(fig)

    logger.info("Saved loss curves to %s.", output_path)


def plot_learning_curves(
    history: pd.DataFrame,
    output_path: Union[str, Path],
    metric_columns: Optional[Sequence[str]] = None,
    title: str = "Learning Curves",
) -> None:
    """Plot validation metric learning curves from training history.

    Args:
        history: DataFrame containing epoch-wise metric values.
        output_path: Destination image file path.
        metric_columns: Metric columns to plot. Defaults to AUROC and F1 curves.
        title: Plot title.

    Raises:
        ValueError: If no valid metric columns are found.
        OSError: If saving the figure fails.
    """
    metric_columns = list(
        metric_columns
        or [
            "val_macro_auroc",
            "val_micro_auroc",
            "val_f1_macro",
        ]
    )
    available_columns = [column for column in metric_columns if column in history.columns]
    if not available_columns:
        raise ValueError("No requested metric columns exist in history DataFrame.")

    fig, axis = plt.subplots(figsize=(8, 5))
    try:
        for column in available_columns:
            axis.plot(history[column], label=column, linewidth=2.0)

        axis.set_xlabel("Epoch")
        axis.set_ylabel("Metric Value")
        axis.set_title(title)
        axis.legend()
        axis.grid(alpha=0.3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as exc:
        logger.exception("Failed to plot learning curves.")
        raise OSError("Unable to save learning curve plot.") from exc
    finally:
        plt.close(fig)

    logger.info("Saved learning curves to %s.", output_path)


def export_evaluation_artifacts(
    y_true: ArrayLike,
    y_probs: ArrayLike,
    output_dir: Union[str, Path],
    config: Optional[Config] = None,
    split_name: str = "validation",
    image_paths: Optional[Sequence[str]] = None,
    loss: Optional[float] = None,
    thresholds: Optional[ArrayLike] = None,
) -> MetricsResult:
    """Compute metrics and export all configured evaluation artifacts.

    Args:
        y_true: Ground-truth labels of shape ``(N, C)``.
        y_probs: Predicted probabilities of shape ``(N, C)``.
        output_dir: Directory for metrics, plots, and CSV exports.
        config: Optional project configuration.
        split_name: Split identifier used in filenames.
        image_paths: Optional image paths for predictions CSV export.
        loss: Optional mean loss value.
        thresholds: Optional per-class thresholds for metric computation.

    Returns:
        Computed ``MetricsResult``.
    """
    config = config or get_default_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_config = config.evaluation
    threshold_values = (
        _normalize_thresholds(thresholds)
        if thresholds is not None
        else eval_config.threshold
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_probs=y_probs,
        threshold=threshold_values,
        loss=loss,
    )

    save_metrics_csv(metrics, output_dir / f"{split_name}_metrics.csv", split_name=split_name)

    if eval_config.save_predictions and image_paths is not None:
        save_predictions_csv(
            y_true=y_true,
            y_probs=y_probs,
            image_paths=image_paths,
            output_path=output_dir / "predictions.csv",
            threshold=threshold_values,
        )

    if metrics.best_thresholds is not None or thresholds is not None:
        save_best_thresholds_csv(
            thresholds if thresholds is not None else metrics.best_thresholds,
            output_path=output_dir / "best_thresholds.csv",
        )

    if eval_config.save_roc_curves:
        plot_roc_curves(
            y_true=y_true,
            y_probs=y_probs,
            output_path=output_dir / f"{split_name}_roc_curves.png",
            title=f"{split_name.capitalize()} ROC Curves",
        )

    if eval_config.save_pr_curves:
        plot_precision_recall_curves(
            y_true=y_true,
            y_probs=y_probs,
            output_path=output_dir / f"{split_name}_pr_curves.png",
            title=f"{split_name.capitalize()} Precision-Recall Curves",
        )

    if eval_config.save_confusion_matrices:
        plot_confusion_matrices(
            confusion_matrices=metrics.confusion_matrices,
            output_path=output_dir / f"{split_name}_confusion_matrices.png",
            title=f"{split_name.capitalize()} Confusion Matrices",
        )

    return metrics
