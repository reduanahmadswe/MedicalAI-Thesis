"""Evaluation and inference pipeline for multi-label Chest X-ray classification.

Supports validation, testing, single-image inference, and batch inference with
automatic metric computation, threshold optimization, and artifact export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import Config, NIH_DISEASE_LABELS, SplitName, get_default_config
from src.utils import normalize_split_name, resolve_device
from src.dataset import (
    InferenceDataset,
    create_dataloader,
    create_inference_dataloader,
    load_single_image,
)
from src.losses import build_loss, compute_batch_loss
from src.metrics import (
    MetricsAccumulator,
    MetricsResult,
    export_evaluation_artifacts,
    save_best_thresholds_csv,
    search_best_thresholds,
)
from src.models import load_model_from_checkpoint, move_model_to_device, summarize_model


logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Prediction output for one or more images.

    Attributes:
        image_paths: Image paths included in the inference run.
        probabilities: Predicted probabilities of shape ``(N, C)``.
        predictions: Binary predictions of shape ``(N, C)``.
        labels: Optional ground-truth labels of shape ``(N, C)``.
        thresholds: Threshold values used for binarization.
        class_names: Disease class names corresponding to columns.
    """

    image_paths: List[str]
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: Optional[np.ndarray] = None
    thresholds: Union[float, np.ndarray] = 0.5
    class_names: Tuple[str, ...] = NIH_DISEASE_LABELS

    def to_dataframe(self) -> pd.DataFrame:
        """Convert inference outputs to a pandas DataFrame.

        Returns:
            DataFrame with one row per image and per-class probability columns.
        """
        records: Dict[str, Any] = {"image_path": self.image_paths}
        for index, class_name in enumerate(self.class_names):
            records[f"{class_name}_prob"] = self.probabilities[:, index].tolist()
            records[f"{class_name}_pred"] = self.predictions[:, index].astype(int).tolist()
            if self.labels is not None:
                records[f"{class_name}_true"] = self.labels[:, index].astype(int).tolist()
        return pd.DataFrame.from_dict(records)

    def get_single_prediction_summary(self, index: int = 0) -> Dict[str, Any]:
        """Build a human-readable summary for a single sample.

        Args:
            index: Sample index in the batch.

        Returns:
            Dictionary containing probabilities and predicted disease names.

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(f"Index {index} out of range for {len(self.image_paths)} samples.")

        probabilities = self.probabilities[index]
        predictions = self.predictions[index]

        predicted_diseases = [
            class_name
            for class_name, predicted in zip(self.class_names, predictions)
            if predicted == 1
        ]

        return {
            "image_path": self.image_paths[index],
            "probabilities": {
                class_name: float(probabilities[class_index])
                for class_index, class_name in enumerate(self.class_names)
            },
            "predictions": {
                class_name: int(predictions[class_index])
                for class_index, class_name in enumerate(self.class_names)
            },
            "predicted_diseases": predicted_diseases,
        }


@dataclass
class EvaluationReport:
    """Container for split-wise evaluation outputs.

    Attributes:
        split_name: Evaluated split identifier.
        metrics: Computed metrics result.
        thresholds: Thresholds used for evaluation.
        output_dir: Directory containing exported artifacts.
    """

    split_name: str
    metrics: MetricsResult
    thresholds: np.ndarray
    output_dir: Path


class Evaluator:
    """Evaluation and inference engine for trained Chest X-ray models.

    Args:
        config: Full project configuration.
        model: Optional pre-loaded model instance.
        checkpoint_path: Optional checkpoint path used to load model weights.
        device: Optional device override.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initialize the evaluator."""
        self.config = config or get_default_config()
        self.config.setup()

        self.device = self._resolve_device(device)
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self.config.paths.best_model_path
        )

        self.model = model
        self.checkpoint: Dict[str, Any] = {}
        if self.model is None:
            self._load_model_from_checkpoint()

        self.model = move_model_to_device(
            self.model,
            device=self.device,
            enable_multi_gpu=False,
        )
        self.model.eval()
        summarize_model(self.model)

        self.loss_fn = build_loss(config=self.config, device=self.device)
        self.optimized_thresholds: Optional[np.ndarray] = None

        self.test_loader = create_dataloader(
            split="test",
            config=self.config,
            shuffle=False,
        )

    def load_best_model(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """Load the best model checkpoint for evaluation.

        Args:
            checkpoint_path: Optional override path. Defaults to ``best_model.pth``.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self.config.paths.best_model_path
        )
        self._load_model_from_checkpoint()
        self.model = move_model_to_device(
            self.model,
            device=self.device,
            enable_multi_gpu=False,
        )
        self.model.eval()
        logger.info("Loaded best model from %s.", self.checkpoint_path)

    def _resolve_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Resolve compute device from override or configuration."""
        return resolve_device(device=device, config=self.config, fallback_to_cpu=True)

    def _load_model_from_checkpoint(self) -> None:
        """Load model weights from the configured checkpoint path."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model, self.checkpoint = load_model_from_checkpoint(
            checkpoint_path=str(self.checkpoint_path),
            config=self.config,
            map_location=self.device,
        )

    def _use_amp(self) -> bool:
        """Return whether mixed precision is enabled during inference."""
        return self.config.training.use_amp and self.device.type == "cuda"

    def _amp_device_type(self) -> str:
        """Return the device type string used by ``torch.amp`` APIs."""
        return "cuda" if self.device.type == "cuda" else "cpu"

    @torch.no_grad()
    def run_split_inference(
        self,
        split: Union[SplitName, str],
        dataloader: Optional[DataLoader] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[float]]:
        """Run inference on a labeled dataset split.

        Args:
            split: Dataset split identifier.
            dataloader: Optional pre-constructed DataLoader.

        Returns:
            Tuple of ``(y_true, y_probs, image_paths, mean_loss)``.
        """
        split_value = normalize_split_name(split)
        loader = dataloader or create_dataloader(
            split=split,
            config=self.config,
            shuffle=False,
        )

        accumulator = MetricsAccumulator()
        image_paths: List[str] = []

        progress = tqdm(loader, desc=f"Inference [{split_value}]", leave=False)
        for batch in progress:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            batch_paths = batch["image_path"]

            with autocast(self._amp_device_type(), enabled=self._use_amp()):
                logits = self.model(images)
                loss = compute_batch_loss(self.loss_fn, logits, labels)

            accumulator.update(logits=logits, labels=labels, loss=float(loss.item()))
            image_paths.extend([str(path) for path in batch_paths])

        labels_array, probabilities, mean_loss = accumulator.get_arrays()
        return labels_array, probabilities, image_paths, mean_loss

    def evaluate_split(
        self,
        split: Union[SplitName, str],
        thresholds: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        output_subdir: Optional[str] = None,
    ) -> EvaluationReport:
        """Evaluate a dataset split and export configured artifacts.

        Args:
            split: Dataset split identifier (``val`` or ``test``).
            thresholds: Optional scalar or per-class thresholds.
            output_subdir: Optional subdirectory under ``results_dir``.

        Returns:
            ``EvaluationReport`` for the evaluated split.
        """
        split_value = normalize_split_name(split)
        output_dir = self.config.paths.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        labels, probabilities, image_paths, mean_loss = self.run_split_inference(split=split)

        threshold_values = thresholds
        if threshold_values is None:
            threshold_values = (
                self.optimized_thresholds
                if self.optimized_thresholds is not None
                else self.config.evaluation.threshold
            )

        metrics = export_evaluation_artifacts(
            y_true=labels,
            y_probs=probabilities,
            output_dir=output_dir,
            config=self.config,
            split_name=split_value,
            image_paths=image_paths,
            loss=mean_loss,
            thresholds=threshold_values,
        )

        if isinstance(threshold_values, (float, int)):
            threshold_array = np.full(len(NIH_DISEASE_LABELS), float(threshold_values))
        else:
            threshold_array = np.asarray(threshold_values, dtype=np.float32)

        logger.info(
            "Evaluation complete for split='%s'. macro_auroc=%.4f, f1_macro=%.4f.",
            split_value,
            metrics.auroc_macro,
            metrics.f1_macro,
        )

        return EvaluationReport(
            split_name=split_value,
            metrics=metrics,
            thresholds=threshold_array,
            output_dir=output_dir,
        )

    def validate(self, optimize_thresholds: Optional[bool] = None) -> EvaluationReport:
        """Run validation evaluation and optionally optimize thresholds.

        Args:
            optimize_thresholds: Override ``EvaluationConfig.optimize_threshold``.

        Returns:
            Validation ``EvaluationReport``.
        """
        should_optimize = (
            self.config.evaluation.optimize_threshold
            if optimize_thresholds is None
            else optimize_thresholds
        )

        if should_optimize:
            labels, probabilities, _, _ = self.run_split_inference(split=SplitName.VAL)
            self.optimized_thresholds = search_best_thresholds(
                y_true=labels,
                y_probs=probabilities,
                config=self.config.evaluation,
            )
            thresholds_path = self.config.paths.thresholds_csv_path
            save_best_thresholds_csv(
                thresholds=self.optimized_thresholds,
                output_path=thresholds_path,
            )
            logger.info("Saved optimized thresholds to %s.", thresholds_path)

        report = self.evaluate_split(
            split=SplitName.VAL,
            thresholds=self.optimized_thresholds,
            output_subdir="validation",
        )
        return report

    def test(
        self,
        thresholds: Optional[Union[float, Sequence[float], np.ndarray]] = None,
        dataloader: Optional[DataLoader] = None,
    ) -> EvaluationReport:
        """Run held-out test evaluation using the test DataLoader.

        Args:
            thresholds: Optional thresholds. Uses optimized validation thresholds when absent.
            dataloader: Optional test DataLoader override. Defaults to ``self.test_loader``.

        Returns:
            Test ``EvaluationReport``.
        """
        threshold_values = thresholds
        if threshold_values is None:
            threshold_values = self._load_thresholds_if_available()

        loader = dataloader or self.test_loader
        labels, probabilities, image_paths, mean_loss = self.run_split_inference(
            split=SplitName.TEST,
            dataloader=loader,
        )

        output_dir = self.config.paths.results_dir
        metrics = export_evaluation_artifacts(
            y_true=labels,
            y_probs=probabilities,
            output_dir=output_dir,
            config=self.config,
            split_name="test",
            image_paths=image_paths,
            loss=mean_loss,
            thresholds=threshold_values,
        )

        if isinstance(threshold_values, (float, int)):
            threshold_array = np.full(len(NIH_DISEASE_LABELS), float(threshold_values))
        else:
            threshold_array = np.asarray(threshold_values, dtype=np.float32)

        logger.info(
            "Test evaluation complete. macro_auroc=%.4f, f1_macro=%.4f.",
            metrics.auroc_macro,
            metrics.f1_macro,
        )

        return EvaluationReport(
            split_name="test",
            metrics=metrics,
            thresholds=threshold_array,
            output_dir=output_dir,
        )

    def _load_thresholds_if_available(self) -> Union[float, np.ndarray]:
        """Load optimized thresholds from disk when available."""
        thresholds_path = self.config.paths.thresholds_csv_path
        if thresholds_path.exists():
            try:
                thresholds_df = pd.read_csv(thresholds_path)
                if "threshold" in thresholds_df.columns:
                    loaded = thresholds_df["threshold"].to_numpy(dtype=np.float32)
                    if loaded.shape[0] == len(NIH_DISEASE_LABELS):
                        logger.info("Loaded thresholds from %s.", thresholds_path)
                        return loaded
            except Exception as exc:
                logger.warning("Failed to load thresholds from %s: %s", thresholds_path, exc)

        if self.optimized_thresholds is not None:
            return self.optimized_thresholds

        logger.info(
            "Using default evaluation threshold=%.2f.",
            self.config.evaluation.threshold,
        )
        return self.config.evaluation.threshold

    @torch.no_grad()
    def predict_single(
        self,
        image_path: Union[str, Path],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run inference on a single Chest X-ray image.

        Args:
            image_path: Path to the input image.
            threshold: Optional decision threshold. Uses optimized or default threshold.

        Returns:
            Dictionary with probabilities, binary predictions, and disease names.
        """
        threshold_value = threshold if threshold is not None else self._resolve_single_threshold()
        image_tensor = load_single_image(image_path=image_path, config=self.config)
        batch_tensor = image_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        with autocast(self._amp_device_type(), enabled=self._use_amp()):
            logits = self.model(batch_tensor)

        probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]
        predictions = (probabilities >= threshold_value).astype(np.int32)

        inference_result = InferenceResult(
            image_paths=[str(image_path)],
            probabilities=probabilities.reshape(1, -1),
            predictions=predictions.reshape(1, -1),
            thresholds=threshold_value,
        )
        return inference_result.get_single_prediction_summary(index=0)

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: Sequence[Union[str, Path]],
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
        save_csv: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> InferenceResult:
        """Run batch inference on a list of image paths.

        Args:
            image_paths: Sequence of image file paths.
            threshold: Optional decision threshold.
            batch_size: Optional batch size override.
            save_csv: Whether to export predictions to CSV.
            output_path: Optional CSV output path when ``save_csv=True``.

        Returns:
            ``InferenceResult`` containing batch predictions.
        """
        if not image_paths:
            raise ValueError("image_paths must contain at least one path.")

        threshold_value = threshold if threshold is not None else self._resolve_single_threshold()
        inference_dataset = InferenceDataset(
            image_paths=image_paths,
            paths=self.config.paths,
        )
        dataloader = create_inference_dataloader(
            image_paths=image_paths,
            config=self.config,
            batch_size=batch_size,
        )

        probabilities_list: List[np.ndarray] = []
        resolved_paths = [str(path) for path in inference_dataset.image_paths]

        progress = tqdm(dataloader, desc="Batch Inference", leave=False)
        for batch_tensor in progress:
            batch_tensor = batch_tensor.to(self.device, non_blocking=True)
            with autocast(self._amp_device_type(), enabled=self._use_amp()):
                logits = self.model(batch_tensor)

            batch_probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            probabilities_list.append(batch_probabilities)

        probabilities = np.concatenate(probabilities_list, axis=0)
        if isinstance(threshold_value, float):
            predictions = (probabilities >= threshold_value).astype(np.int32)
        else:
            threshold_array = np.asarray(threshold_value, dtype=np.float32).reshape(1, -1)
            predictions = (probabilities >= threshold_array).astype(np.int32)

        result = InferenceResult(
            image_paths=resolved_paths,
            probabilities=probabilities,
            predictions=predictions,
            thresholds=threshold_value,
        )

        if save_csv:
            csv_path = output_path or self.config.paths.predictions_csv_path
            result_df = result.to_dataframe()
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(csv_path, index=False)
            logger.info("Saved batch predictions to %s.", csv_path)

        return result

    def _resolve_single_threshold(self) -> float:
        """Resolve a scalar threshold for single/batch inference."""
        loaded = self._load_thresholds_if_available()
        if isinstance(loaded, (float, int)):
            return float(loaded)
        return float(np.mean(loaded))

    def run_full_evaluation(self) -> Dict[str, EvaluationReport]:
        """Run validation (with threshold tuning) followed by test evaluation.

        Returns:
            Dictionary with ``validation`` and ``test`` evaluation reports.
        """
        validation_report = self.validate(optimize_thresholds=True)
        test_report = self.test(thresholds=validation_report.thresholds)

        summary_path = self.config.paths.results_dir / "evaluation_summary.csv"
        summary_records = []

        for report_name, report in {
            "validation": validation_report,
            "test": test_report,
        }.items():
            record = report.metrics.to_dict()
            record["split"] = report_name
            summary_records.append(record)

        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved evaluation summary to %s.", summary_path)

        return {
            "validation": validation_report,
            "test": test_report,
        }


def evaluate_validation(
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> EvaluationReport:
    """Evaluate the validation split.

    Args:
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.

    Returns:
        Validation evaluation report.
    """
    evaluator = Evaluator(config=config, checkpoint_path=checkpoint_path)
    return evaluator.validate()


def evaluate_test(
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    thresholds: Optional[Union[float, Sequence[float], np.ndarray]] = None,
) -> EvaluationReport:
    """Evaluate the test split.

    Args:
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.
        thresholds: Optional decision thresholds.

    Returns:
        Test evaluation report.
    """
    evaluator = Evaluator(config=config, checkpoint_path=checkpoint_path)
    return evaluator.test(thresholds=thresholds)


def run_inference_on_image(
    image_path: Union[str, Path],
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run single-image inference with a trained checkpoint.

    Args:
        image_path: Path to the Chest X-ray image.
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.
        threshold: Optional decision threshold.

    Returns:
        Prediction summary dictionary.
    """
    evaluator = Evaluator(config=config, checkpoint_path=checkpoint_path)
    return evaluator.predict_single(image_path=image_path, threshold=threshold)


def run_batch_inference(
    image_paths: Sequence[Union[str, Path]],
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    threshold: Optional[float] = None,
    save_csv: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> InferenceResult:
    """Run batch inference over multiple image paths.

    Args:
        image_paths: Sequence of image file paths.
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.
        threshold: Optional decision threshold.
        save_csv: Whether to save predictions CSV.
        output_path: Optional CSV destination path.

    Returns:
        Batch ``InferenceResult``.
    """
    evaluator = Evaluator(config=config, checkpoint_path=checkpoint_path)
    return evaluator.predict_batch(
        image_paths=image_paths,
        threshold=threshold,
        save_csv=save_csv,
        output_path=output_path,
    )


def run_full_evaluation_pipeline(
    config: Optional[Config] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict[str, EvaluationReport]:
    """Run the complete validation and test evaluation pipeline.

    Args:
        config: Optional project configuration.
        checkpoint_path: Optional model checkpoint path.

    Returns:
        Dictionary containing validation and test reports.
    """
    evaluator = Evaluator(config=config, checkpoint_path=checkpoint_path)
    return evaluator.run_full_evaluation()
