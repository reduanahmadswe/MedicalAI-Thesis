"""Professional training pipeline for multi-label Chest X-ray classification.

Implements mixed-precision training, checkpointing, early stopping, TensorBoard
logging, CSV logging, and reproducible experiment execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.config import Config, SplitName, get_default_config
from src.dataset import compute_pos_weight, create_dataloader
from src.losses import build_loss, compute_batch_loss
from src.metrics import (
    MetricsAccumulator,
    MetricsResult,
    build_epoch_log_record,
    compute_metrics,
    save_metrics_csv,
    save_best_thresholds_csv,
    save_training_log_csv,
    save_training_plots,
    save_training_summary_csv,
    save_validation_epoch_artifacts,
    search_best_thresholds,
)
from src.models import _align_state_dict_keys, build_model, move_model_to_device, summarize_model
from src.utils import load_checkpoint_dict, set_seed


logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Runtime training state persisted across checkpoints.

    Attributes:
        epoch: Next epoch index to run (0-based).
        global_step: Total number of optimization steps completed.
        best_metric: Best monitored validation metric observed so far.
        epochs_without_improvement: Early stopping counter.
        history: Epoch-wise metric records for CSV and plotting.
    """

    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    epochs_without_improvement: int = 0
    history: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


class EarlyStopping:
    """Early stopping utility based on a monitored validation metric.

    Args:
        monitor: Metric name to monitor (default: ``auroc``).
        mode: ``'max'`` when higher is better, ``'min'`` when lower is better.
        patience: Number of epochs to wait without improvement.
        min_delta: Minimum change required to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "auroc",
        mode: str = "max",
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        """Initialize early stopping."""
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("patience must be > 0.")

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metrics: Dict[str, float]) -> bool:
        """Update early stopping state with the latest epoch metrics.

        Args:
            metrics: Dictionary of epoch metrics including the monitored value.

        Returns:
            ``True`` if training should stop, otherwise ``False``.

        Raises:
            KeyError: If the monitored metric is missing from ``metrics``.
        """
        if self.monitor not in metrics:
            raise KeyError(f"Monitored metric '{self.monitor}' not found in metrics.")

        current_score = float(metrics[self.monitor])
        if self.best_score is None:
            self.best_score = current_score
            self.counter = 0
            return False

        improved = (
            current_score > self.best_score + self.min_delta
            if self.mode == "max"
            else current_score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement on '%s'.",
                    self.patience,
                    self.monitor,
                )
                return True

        return False


def _unwrap_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return an unwrapped model state dictionary suitable for checkpointing."""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load a state dictionary into a possibly wrapped model."""
    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    except Exception as exc:
        logger.exception("Failed to load model state dict.")
        raise RuntimeError("Unable to load model state dict.") from exc


def _build_checkpoint_payload(
    trainer: "Trainer",
    epoch: int,
) -> Dict[str, Any]:
    """Build a complete checkpoint dictionary for saving.

    Args:
        trainer: Active trainer instance.
        epoch: Zero-based epoch index completed.

    Returns:
        Checkpoint dictionary with model, optimizer, scheduler, and scaler states.
    """
    return {
        "epoch": epoch,
        "global_step": trainer.state.global_step,
        "model_state_dict": _unwrap_state_dict(trainer.model),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": (
            trainer.scheduler.state_dict() if trainer.scheduler is not None else None
        ),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "best_metric": trainer.state.best_metric,
        "epochs_without_improvement": trainer.state.epochs_without_improvement,
        "optimized_thresholds": (
            trainer.optimized_thresholds.tolist()
            if trainer.optimized_thresholds is not None
            else None
        ),
        "config": trainer.config.to_dict(),
        "history": trainer.state.history,
    }


class Trainer:
    """End-to-end trainer for multi-label Chest X-ray disease classification.

    Example:
        >>> from src.trainer import Trainer
        >>> trainer = Trainer()
        >>> history = trainer.fit(resume=False)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the trainer and runtime dependencies."""
        self.config = config or get_default_config()
        self.config.setup()

        set_seed(
            seed=self.config.training.random_seed,
            deterministic=self.config.training.deterministic,
        )

        self.device = self._resolve_device()
        self.state = TrainerState()

        self.train_loader = train_loader or create_dataloader(
            SplitName.TRAIN, config=self.config
        )
        self.val_loader = val_loader or create_dataloader(
            SplitName.VAL, config=self.config, shuffle=False
        )
        self.test_loader = test_loader or create_dataloader(
            split="test", config=self.config, shuffle=False
        )

        self.model = model or build_model(config=self.config)
        self.model = move_model_to_device(
            self.model,
            device=self.device,
            enable_multi_gpu=self.config.device.multi_gpu,
        )
        summarize_model(self.model)

        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            pos_weight = None
            if self.config.loss.compute_class_weights:
                pos_weight = compute_pos_weight(self.train_loader.dataset)
            self.loss_fn = build_loss(
                config=self.config,
                pos_weight=pos_weight,
                device=self.device,
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(self._amp_device_type(), enabled=self._use_amp())
        self.early_stopping = self._build_early_stopping()
        self.writer = self._build_tensorboard_writer()
        self._configure_file_logging()
        self.optimized_thresholds: Optional[np.ndarray] = None

    def _resolve_device(self) -> torch.device:
        """Resolve the compute device from configuration."""
        requested = self.config.device.device
        if requested.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(requested if requested != "cuda" else "cuda:0")
        else:
            if requested.startswith("cuda"):
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            device = torch.device("cpu")
        logger.info("Using device: %s", device)
        return device

    def _use_amp(self) -> bool:
        """Return whether automatic mixed precision is enabled."""
        return self.config.training.use_amp and self.device.type == "cuda"

    def _amp_device_type(self) -> str:
        """Return the device type string used by ``torch.amp`` APIs."""
        return "cuda" if self.device.type == "cuda" else "cpu"

    def _build_optimizer(self) -> AdamW:
        """Create the AdamW optimizer."""
        optimizer_config = self.config.optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps,
        )
        logger.info(
            "Initialized AdamW optimizer (lr=%.2e, weight_decay=%.2e).",
            optimizer_config.learning_rate,
            optimizer_config.weight_decay,
        )
        return optimizer

    def _build_scheduler(self) -> Optional[LRScheduler]:
        """Create the cosine annealing learning rate scheduler."""
        if not self.config.scheduler.use_scheduler:
            return None

        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.scheduler.t_max,
            eta_min=self.config.scheduler.eta_min,
        )
        logger.info(
            "Initialized CosineAnnealingLR (T_max=%d, eta_min=%.2e).",
            self.config.scheduler.t_max,
            self.config.scheduler.eta_min,
        )
        return scheduler

    def _build_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping helper when enabled."""
        if not self.config.training.early_stopping:
            return None

        return EarlyStopping(
            monitor=self.config.training.early_stopping_monitor,
            mode=self.config.training.early_stopping_mode,
            patience=self.config.training.early_stopping_patience,
            min_delta=self.config.training.early_stopping_min_delta,
        )

    def _build_tensorboard_writer(self) -> Optional[SummaryWriter]:
        """Create a TensorBoard summary writer when enabled."""
        if not self.config.logging.tensorboard_enabled:
            return None

        run_dir = (
            self.config.paths.tensorboard_dir
            / self.config.logging.experiment_name
            / time.strftime("%Y%m%d-%H%M%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_dir))
        logger.info("TensorBoard logging enabled at %s.", run_dir)
        return writer

    def _configure_file_logging(self) -> None:
        """Attach a file handler for training logs when configured."""
        if not self.config.logging.log_to_file:
            return

        log_path = self.config.paths.training_log_file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, self.config.logging.log_level.upper()))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )

        root_logger = logging.getLogger()
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", "") == str(log_path)
            for handler in root_logger.handlers
        ):
            root_logger.addHandler(file_handler)

        logger.info("File logging enabled at %s.", log_path)

    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Return peak GPU memory usage in megabytes when CUDA is available."""
        if not self.config.logging.log_gpu_memory or self.device.type != "cuda":
            return None
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        return None

    def _reset_gpu_memory_stats(self) -> None:
        """Reset CUDA peak memory statistics before an epoch."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def fit(self, resume: bool = False) -> pd.DataFrame:
        """Run the full training loop.

        Args:
            resume: When ``True``, resume from ``Models/last_model.pth``.

        Returns:
            DataFrame containing epoch-wise training history.
        """
        if resume or self.config.training.resume_training:
            self.resume_training()

        start_time = time.time()
        num_epochs = self.config.training.num_epochs

        logger.info("Starting training for %d epochs.", num_epochs)
        for epoch in range(self.state.epoch, num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            train_loss = self.train_one_epoch(epoch=epoch)
            val_metrics = self.validate(epoch=epoch)
            learning_rate = self.optimizer.param_groups[0]["lr"]

            epoch_record = build_epoch_log_record(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_metrics=val_metrics,
                learning_rate=learning_rate,
            )

            monitored_value = float(epoch_record[self.config.training.early_stopping_monitor])
            if self.config.training.early_stopping_mode == "min":
                if self.state.best_metric == float("-inf"):
                    self.state.best_metric = float("inf")
                is_best = monitored_value < self.state.best_metric
            else:
                is_best = monitored_value > self.state.best_metric

            if is_best:
                self.state.best_metric = monitored_value

            self.save_checkpoint(is_best=is_best, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self.state.history.append(epoch_record)
            self._log_epoch_to_tensorboard(epoch_record, epoch)
            save_training_log_csv(
                self.state.history,
                self.config.paths.training_log_path,
            )

            epoch_time = time.time() - epoch_start
            if self.config.logging.log_training_time:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | auroc=%.4f | "
                    "f1=%.4f | time=%.2fs",
                    epoch + 1,
                    num_epochs,
                    epoch_record["train_loss"],
                    epoch_record["val_loss"],
                    epoch_record["auroc"],
                    epoch_record["f1"],
                    epoch_time,
                )

            if self.early_stopping is not None:
                if self.early_stopping.step(epoch_record):
                    self.state.epochs_without_improvement = self.early_stopping.counter
                    break
                self.state.epochs_without_improvement = self.early_stopping.counter

        total_time = time.time() - start_time
        logger.info("Training finished in %.2f seconds.", total_time)

        history_df = pd.DataFrame(self.state.history)
        best_epoch = self._find_best_epoch(history_df)
        save_training_summary_csv(
            history=self.state.history,
            output_path=self.config.paths.training_summary_path,
            best_metric=self.state.best_metric,
            best_epoch=best_epoch,
        )
        self._save_training_artifacts(history_df)
        self.close()
        return history_df

    def _find_best_epoch(self, history_df: pd.DataFrame) -> Optional[int]:
        """Return the epoch number with the best monitored validation metric."""
        monitor = self.config.training.early_stopping_monitor
        if monitor not in history_df.columns or history_df.empty:
            return None

        series = history_df[monitor]
        if self.config.training.early_stopping_mode == "min":
            best_index = int(series.idxmin())
        else:
            best_index = int(series.idxmax())
        return int(history_df.iloc[best_index]["epoch"])

    def train(self, resume: bool = False) -> pd.DataFrame:
        """Backward-compatible alias for :meth:`fit`.

        Args:
            resume: When ``True``, resume from the last checkpoint.

        Returns:
            Training history DataFrame.
        """
        return self.fit(resume=resume)

    def train_one_epoch(self, epoch: int) -> float:
        """Train the model for a single epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        self._reset_gpu_memory_stats()

        running_loss = 0.0
        num_batches = 0
        progress = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch + 1}/{self.config.training.num_epochs}",
            leave=True,
        )

        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(self._amp_device_type(), enabled=self._use_amp()):
                    logits = self.model(images)
                    loss = compute_batch_loss(self.loss_fn, logits, labels)
            except Exception as exc:
                logger.exception("Forward pass failed at epoch %d, batch %d.", epoch, batch_index)
                raise RuntimeError("Training forward pass failed.") from exc

            self.scaler.scale(loss).backward()

            if self.config.training.gradient_clip_norm > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = float(loss.item())
            running_loss += batch_loss
            num_batches += 1
            self.state.global_step += 1

            progress.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

            if (batch_index + 1) % self.config.training.log_interval == 0 and self.writer is not None:
                self.writer.add_scalar(
                    "train/batch_loss",
                    batch_loss,
                    global_step=self.state.global_step,
                )

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: Optional[int] = None) -> MetricsResult:
        """Run validation, optimize thresholds, and export epoch artifacts.

        Args:
            epoch: Optional epoch index for logging and artifact naming.

        Returns:
            Computed validation ``MetricsResult`` using optimized thresholds when enabled.
        """
        self.model.eval()
        accumulator = MetricsAccumulator()
        image_paths: List[str] = []

        progress = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {(epoch + 1) if epoch is not None else ''}".strip(),
            leave=False,
        )

        for batch in progress:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(self._amp_device_type(), enabled=self._use_amp()):
                logits = self.model(images)
                loss = compute_batch_loss(self.loss_fn, logits, labels)

            accumulator.update(
                logits=logits,
                labels=labels,
                loss=float(loss.item()),
            )
            image_paths.extend([str(path) for path in batch["image_path"]])

        labels_array, probabilities, mean_loss = accumulator.get_arrays()
        threshold_values: Union[float, np.ndarray] = self.config.training.threshold

        if self.config.training.optimize_thresholds_on_val:
            self.optimized_thresholds = search_best_thresholds(
                y_true=labels_array,
                y_probs=probabilities,
                config=self.config.evaluation,
            )
            threshold_values = self.optimized_thresholds
            save_best_thresholds_csv(
                thresholds=self.optimized_thresholds,
                output_path=self.config.paths.thresholds_csv_path,
            )

        metrics = compute_metrics(
            y_true=labels_array,
            y_probs=probabilities,
            threshold=threshold_values,
            loss=mean_loss,
        )
        metrics.best_thresholds = self.optimized_thresholds

        validation_dir = self.config.paths.results_dir
        save_validation_epoch_artifacts(
            y_true=labels_array,
            y_probs=probabilities,
            metrics=metrics,
            output_dir=validation_dir,
            config=self.config,
            epoch=(epoch + 1) if epoch is not None else None,
            image_paths=image_paths,
            thresholds=threshold_values if isinstance(threshold_values, np.ndarray) else None,
            split_name="validation",
        )

        save_metrics_csv(
            metrics,
            self.config.paths.metrics_csv_path,
            split_name="validation",
        )

        if epoch is not None:
            logger.info(
                "Validation epoch %d | loss=%.4f | auroc=%.4f | precision=%.4f | "
                "recall=%.4f | f1=%.4f | accuracy=%.4f",
                epoch + 1,
                metrics.loss if metrics.loss is not None else float("nan"),
                metrics.auroc_macro,
                metrics.precision_macro,
                metrics.recall_macro,
                metrics.f1_macro,
                metrics.accuracy,
            )

        return metrics

    def save_checkpoint(self, is_best: bool, epoch: int) -> None:
        """Save last, per-epoch, and optional best model checkpoints.

        Saves:
        - ``Models/last_model.pth`` every epoch (overwritten)
        - ``Models/Checkpoints/checkpoint_epoch_XXX.pth`` when enabled (never overwritten)
        - ``Models/best_model.pth`` when validation metric improves

        Checkpoint save failures are logged as warnings and do not stop training.

        Args:
            is_best: Whether the current epoch produced the best monitored metric.
            epoch: Zero-based epoch index.
        """
        checkpoint = _build_checkpoint_payload(self, epoch=epoch)
        paths = self.config.paths
        paths.ensure_directories()

        last_path = paths.last_model_path
        try:
            torch.save(checkpoint, last_path)
            logger.debug("Saved last checkpoint to %s.", last_path)
        except Exception:
            logger.warning(
                "Failed to save last checkpoint to %s. Training will continue.",
                last_path,
                exc_info=True,
            )

        save_epoch_checkpoints = (
            self.config.training.save_epoch_checkpoints
            and not self.config.training.save_best_only
        )
        if save_epoch_checkpoints:
            epoch_path = paths.epoch_checkpoint_path(epoch + 1)
            try:
                torch.save(checkpoint, epoch_path)
                logger.debug("Saved epoch checkpoint to %s.", epoch_path)
            except Exception:
                logger.warning(
                    "Failed to save epoch checkpoint to %s. Training will continue.",
                    epoch_path,
                    exc_info=True,
                )

        if is_best:
            best_path = paths.best_model_path
            try:
                torch.save(checkpoint, best_path)
                logger.info(
                    "Saved best checkpoint to %s (metric=%.4f).",
                    best_path,
                    self.state.best_metric,
                )
            except Exception:
                logger.warning(
                    "Failed to save best checkpoint to %s. Training will continue.",
                    best_path,
                    exc_info=True,
                )

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training state from a checkpoint file.

        Args:
            checkpoint_path: Path to a saved ``.pth`` checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = load_checkpoint_dict(checkpoint_path, map_location=self.device)
        except Exception as exc:
            logger.exception("Failed to read checkpoint: %s", checkpoint_path)
            raise RuntimeError(f"Unable to read checkpoint: {checkpoint_path}") from exc

        model_state = checkpoint.get("model_state_dict")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if model_state is None or optimizer_state is None:
            raise RuntimeError(
                "Checkpoint must contain 'model_state_dict' and 'optimizer_state_dict'."
            )

        model_state = _align_state_dict_keys(model_state, self.model)
        _load_state_dict(self.model, model_state)
        self.optimizer.load_state_dict(optimizer_state)

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.state.epoch = int(checkpoint.get("epoch", 0)) + 1
        self.state.global_step = int(checkpoint.get("global_step", 0))
        self.state.best_metric = float(checkpoint.get("best_metric", float("-inf")))
        self.state.epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        self.state.history = list(checkpoint.get("history", []))

        optimized = checkpoint.get("optimized_thresholds")
        if optimized is not None:
            self.optimized_thresholds = np.asarray(optimized, dtype=np.float32)

        if self.early_stopping is not None:
            self.early_stopping.best_score = float(
                checkpoint.get("best_metric", self.early_stopping.best_score or 0.0)
            )
            self.early_stopping.counter = self.state.epochs_without_improvement

        logger.info(
            "Loaded checkpoint from %s. Resuming at epoch %d (best_metric=%.4f).",
            checkpoint_path,
            self.state.epoch + 1,
            self.state.best_metric,
        )

    def resume_training(self) -> None:
        """Resume training from ``Models/last_model.pth`` if it exists."""
        checkpoint_path = self.config.paths.last_model_path
        if checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
        else:
            logger.warning(
                "Resume requested but no checkpoint found at %s. Starting fresh.",
                checkpoint_path,
            )

    def _log_epoch_to_tensorboard(self, epoch_record: Dict[str, float], epoch: int) -> None:
        """Write epoch metrics to TensorBoard."""
        if self.writer is None:
            return

        self.writer.add_scalar("train/loss", epoch_record["train_loss"], epoch)
        self.writer.add_scalar("val/loss", epoch_record["val_loss"], epoch)
        self.writer.add_scalar("val/auroc", epoch_record["auroc"], epoch)
        self.writer.add_scalar("val/precision", epoch_record["precision"], epoch)
        self.writer.add_scalar("val/recall", epoch_record["recall"], epoch)
        self.writer.add_scalar("val/f1", epoch_record["f1"], epoch)
        self.writer.add_scalar("val/accuracy", epoch_record["accuracy"], epoch)

        if self.config.logging.log_learning_rate:
            self.writer.add_scalar("train/learning_rate", epoch_record["learning_rate"], epoch)

        gpu_memory_mb = self._get_gpu_memory_mb()
        if gpu_memory_mb is not None and self.config.logging.log_gpu_memory:
            self.writer.add_scalar("system/gpu_memory_mb", gpu_memory_mb, epoch)

    def _save_training_artifacts(self, history_df: pd.DataFrame) -> None:
        """Save training curves at the end of training."""
        results_dir = self.config.paths.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        save_training_log_csv(
            self.state.history,
            self.config.paths.training_log_path,
        )
        save_training_plots(history_df, self.config.paths.results_dir)
        logger.info("Saved training artifacts to %s.", results_dir)

    def close(self) -> None:
        """Close TensorBoard writer and release resources."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            logger.info("TensorBoard writer closed.")


def train_model(config: Optional[Config] = None, resume: bool = False) -> pd.DataFrame:
    """Convenience entry point to train a model using project defaults.

    Args:
        config: Optional project configuration.
        resume: Whether to resume from the last checkpoint.

    Returns:
        Training history DataFrame.
    """
    config = config or get_default_config()
    trainer = Trainer(config=config)
    return trainer.fit(resume=resume)


def create_trainer_from_config(config: Optional[Config] = None) -> Trainer:
    """Create a configured Trainer instance.

    Args:
        config: Optional project configuration.

    Returns:
        Initialized Trainer.
    """
    return Trainer(config=config or get_default_config())
