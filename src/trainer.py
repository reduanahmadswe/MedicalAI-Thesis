"""Professional training pipeline for multi-label Chest X-ray classification.

Implements mixed-precision training, checkpointing, early stopping, TensorBoard
logging, and reproducible experiment execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
    plot_learning_curves,
    plot_loss_curve,
    save_metrics_csv,
)
from src.models import build_model, move_model_to_device, summarize_model
from src.utils import set_seed


logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Runtime training state persisted across checkpoints.

    Attributes:
        epoch: Current epoch index (0-based during training).
        global_step: Total number of optimization steps completed.
        best_metric: Best monitored validation metric observed so far.
        epochs_without_improvement: Early stopping counter.
        history: Epoch-wise metric records.
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
        monitor: Metric name to monitor (e.g., ``val_macro_auroc``).
        mode: ``'max'`` when higher is better, ``'min'`` when lower is better.
        patience: Number of epochs to wait without improvement.
        min_delta: Minimum change required to qualify as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_macro_auroc",
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


def _extract_metric_from_result(result: MetricsResult, metric_name: str) -> float:
    """Map a monitored metric name to a scalar from ``MetricsResult``.

    Args:
        result: Computed validation metrics.
        metric_name: Metric identifier such as ``val_macro_auroc`` or ``val_loss``.

    Returns:
        Scalar metric value.

    Raises:
        KeyError: If the metric name is not supported.
    """
    mapping = {
        "val_loss": result.loss if result.loss is not None else float("nan"),
        "val_accuracy": result.accuracy,
        "val_hamming_accuracy": result.hamming_accuracy,
        "val_precision_macro": result.precision_macro,
        "val_recall_macro": result.recall_macro,
        "val_f1_macro": result.f1_macro,
        "val_precision_micro": result.precision_micro,
        "val_recall_micro": result.recall_micro,
        "val_f1_micro": result.f1_micro,
        "val_macro_auroc": result.auroc_macro,
        "val_micro_auroc": result.auroc_micro,
    }
    if metric_name not in mapping:
        raise KeyError(f"Unsupported monitored metric '{metric_name}'.")
    return float(mapping[metric_name])


def _unwrap_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return an unwrapped model state dictionary suitable for checkpointing.

    Args:
        model: Model that may be wrapped with ``DataParallel``.

    Returns:
        Model state dictionary.
    """
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load a state dictionary into a possibly wrapped model.

    Args:
        model: Target model.
        state_dict: Checkpoint state dictionary.

    Raises:
        RuntimeError: If loading fails.
    """
    try:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    except Exception as exc:
        logger.exception("Failed to load model state dict.")
        raise RuntimeError("Unable to load model state dict.") from exc


class Trainer:
    """End-to-end trainer for multi-label Chest X-ray disease classification.

    Args:
        config: Full project configuration.
        model: Optional pre-initialized model.
        train_loader: Optional training DataLoader.
        val_loader: Optional validation DataLoader.
        loss_fn: Optional loss function module.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
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
                train_dataset = self.train_loader.dataset
                pos_weight = compute_pos_weight(train_dataset)
            self.loss_fn = build_loss(
                config=self.config,
                pos_weight=pos_weight,
                device=self.device,
            )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=self._use_amp())
        self.early_stopping = self._build_early_stopping()
        self.writer = self._build_tensorboard_writer()
        self._configure_file_logging()

        if self.config.training.resume_training:
            self._try_resume_from_checkpoint()

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

        log_path = self.config.paths.results_dir / self.config.logging.log_filename
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

    def train(self) -> pd.DataFrame:
        """Run the full training loop.

        Returns:
            DataFrame containing epoch-wise training history.
        """
        start_time = time.time()
        num_epochs = self.config.training.num_epochs

        logger.info("Starting training for %d epochs.", num_epochs)
        for epoch in range(self.state.epoch, num_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()

            train_loss = self._train_one_epoch(epoch=epoch)
            epoch_time = time.time() - epoch_start
            gpu_memory_mb = self._get_gpu_memory_mb()

            epoch_record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
            }

            if gpu_memory_mb is not None:
                epoch_record["gpu_memory_mb"] = gpu_memory_mb

            if self.config.logging.log_training_time:
                logger.info(
                    "Epoch %d/%d completed in %.2f sec (train_loss=%.4f).",
                    epoch + 1,
                    num_epochs,
                    epoch_time,
                    train_loss,
                )

            run_validation = (epoch + 1) % self.config.training.eval_interval == 0
            if run_validation:
                val_metrics = self.validate(epoch=epoch)
                epoch_record.update(
                    {
                        "val_loss": val_metrics.loss,
                        "val_accuracy": val_metrics.accuracy,
                        "val_hamming_accuracy": val_metrics.hamming_accuracy,
                        "val_precision_macro": val_metrics.precision_macro,
                        "val_recall_macro": val_metrics.recall_macro,
                        "val_f1_macro": val_metrics.f1_macro,
                        "val_precision_micro": val_metrics.precision_micro,
                        "val_recall_micro": val_metrics.recall_micro,
                        "val_f1_micro": val_metrics.f1_micro,
                        "val_macro_auroc": val_metrics.auroc_macro,
                        "val_micro_auroc": val_metrics.auroc_micro,
                    }
                )

                monitored_value = _extract_metric_from_result(
                    val_metrics,
                    self.config.training.early_stopping_monitor,
                )
                if self.config.training.early_stopping_mode == "min":
                    if self.state.best_metric == float("-inf"):
                        self.state.best_metric = float("inf")
                    is_best = monitored_value < self.state.best_metric
                else:
                    is_best = monitored_value > self.state.best_metric

                if is_best:
                    self.state.best_metric = monitored_value

                self.save_checkpoint(is_best=is_best)

                if self.early_stopping is not None:
                    if self.early_stopping.step(epoch_record):
                        self.state.epochs_without_improvement = self.early_stopping.counter
                        self.state.history.append(epoch_record)
                        self._log_epoch_to_tensorboard(epoch_record, epoch)
                        self._save_training_history()
                        break
                    self.state.epochs_without_improvement = self.early_stopping.counter
            else:
                self.save_checkpoint(is_best=False)

            if self.scheduler is not None:
                self.scheduler.step()

            self.state.history.append(epoch_record)
            self._log_epoch_to_tensorboard(epoch_record, epoch)
            self._save_training_history()

        total_time = time.time() - start_time
        logger.info("Training finished in %.2f seconds.", total_time)

        history_df = pd.DataFrame(self.state.history)
        self._save_training_artifacts(history_df)
        self.close()
        return history_df

    def _train_one_epoch(self, epoch: int) -> float:
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
            leave=False,
        )

        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            try:
                with autocast(device_type=self.device.type, enabled=self._use_amp()):
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

            progress.set_postfix({"loss": f"{batch_loss:.4f}"})

            if (batch_index + 1) % self.config.training.log_interval == 0:
                logger.debug(
                    "Epoch %d Batch %d/%d - loss=%.4f, lr=%.2e",
                    epoch + 1,
                    batch_index + 1,
                    len(self.train_loader),
                    batch_loss,
                    self.optimizer.param_groups[0]["lr"],
                )
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/batch_loss",
                        batch_loss,
                        global_step=self.state.global_step,
                    )

        return running_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: Optional[int] = None) -> MetricsResult:
        """Run validation and compute metrics.

        Args:
            epoch: Optional epoch index for logging.

        Returns:
            Computed validation ``MetricsResult``.
        """
        self.model.eval()
        accumulator = MetricsAccumulator()

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for batch in progress:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self._use_amp()):
                logits = self.model(images)
                loss = compute_batch_loss(self.loss_fn, logits, labels)

            accumulator.update(
                logits=logits,
                labels=labels,
                loss=float(loss.item()),
            )

        metrics = accumulator.compute(threshold=self.config.training.threshold)

        if epoch is not None:
            logger.info(
                "Validation epoch %d - loss=%.4f, macro_auroc=%.4f, f1_macro=%.4f.",
                epoch + 1,
                metrics.loss if metrics.loss is not None else float("nan"),
                metrics.auroc_macro,
                metrics.f1_macro,
            )

        val_metrics_path = self.config.paths.results_dir / "validation_metrics.csv"
        save_metrics_csv(metrics, val_metrics_path, split_name="validation")
        return metrics

    def save_checkpoint(self, is_best: bool) -> None:
        """Save latest and optional best model checkpoints.

        Args:
            is_best: Whether the current epoch produced the best monitored metric.
        """
        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": _unwrap_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.state.best_metric,
            "epochs_without_improvement": self.state.epochs_without_improvement,
            "config": self.config.to_dict(),
            "history": self.state.history,
        }

        latest_path = self.config.paths.latest_checkpoint_path
        latest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(checkpoint, latest_path)
            logger.debug("Saved latest checkpoint to %s.", latest_path)
        except Exception as exc:
            logger.exception("Failed to save latest checkpoint.")
            raise RuntimeError(f"Unable to save latest checkpoint to {latest_path}.") from exc

        if is_best:
            best_path = self.config.paths.best_model_path
            try:
                torch.save(checkpoint, best_path)
                logger.info("Saved best checkpoint to %s.", best_path)
            except Exception as exc:
                logger.exception("Failed to save best checkpoint.")
                raise RuntimeError(f"Unable to save best checkpoint to {best_path}.") from exc

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training state from a checkpoint file.

        Args:
            checkpoint_path: Path to a saved checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as exc:
            logger.exception("Failed to read checkpoint: %s", checkpoint_path)
            raise RuntimeError(f"Unable to read checkpoint: {checkpoint_path}") from exc

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint file must contain a dictionary.")

        _load_state_dict(self.model, checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

        if self.early_stopping is not None:
            self.early_stopping.best_score = float(
                checkpoint.get("best_metric", self.early_stopping.best_score or 0.0)
            )
            self.early_stopping.counter = self.state.epochs_without_improvement

        logger.info(
            "Resumed training from %s at epoch %d (best_metric=%.4f).",
            checkpoint_path,
            self.state.epoch + 1,
            self.state.best_metric,
        )

    def _try_resume_from_checkpoint(self) -> None:
        """Attempt to resume from the latest checkpoint if it exists."""
        checkpoint_path = self.config.paths.latest_checkpoint_path
        if checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)
        else:
            logger.info(
                "Resume requested but no latest checkpoint found at %s.",
                checkpoint_path,
            )

    def _log_epoch_to_tensorboard(self, epoch_record: Dict[str, Any], epoch: int) -> None:
        """Write epoch metrics to TensorBoard."""
        if self.writer is None:
            return

        self.writer.add_scalar("train/loss", epoch_record["train_loss"], epoch)
        if self.config.logging.log_learning_rate:
            self.writer.add_scalar(
                "train/learning_rate",
                epoch_record["learning_rate"],
                epoch,
            )

        if "val_loss" in epoch_record:
            self.writer.add_scalar("val/loss", epoch_record["val_loss"], epoch)
            self.writer.add_scalar("val/macro_auroc", epoch_record["val_macro_auroc"], epoch)
            self.writer.add_scalar("val/micro_auroc", epoch_record["val_micro_auroc"], epoch)
            self.writer.add_scalar("val/f1_macro", epoch_record["val_f1_macro"], epoch)

        if "gpu_memory_mb" in epoch_record and self.config.logging.log_gpu_memory:
            self.writer.add_scalar("system/gpu_memory_mb", epoch_record["gpu_memory_mb"], epoch)

    def _save_training_history(self) -> None:
        """Persist epoch history to CSV after each epoch."""
        if not self.state.history:
            return

        history_df = pd.DataFrame(self.state.history)
        output_path = self.config.paths.results_dir / "training_log.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            history_df.to_csv(output_path, index=False)
        except Exception as exc:
            logger.exception("Failed to save training history CSV.")
            raise OSError(f"Unable to save training history to {output_path}") from exc

    def _save_training_artifacts(self, history_df: pd.DataFrame) -> None:
        """Save training curves and aggregate metrics at the end of training."""
        results_dir = self.config.paths.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        history_path = results_dir / "training_log.csv"
        history_df.to_csv(history_path, index=False)

        if {"train_loss", "val_loss"}.issubset(history_df.columns):
            plot_loss_curve(
                history=history_df,
                output_path=results_dir / "loss_curves.png",
            )

        metric_columns = [
            column
            for column in [
                "val_macro_auroc",
                "val_micro_auroc",
                "val_f1_macro",
            ]
            if column in history_df.columns
        ]
        if metric_columns:
            plot_learning_curves(
                history=history_df,
                output_path=results_dir / "learning_curves.png",
                metric_columns=metric_columns,
            )

        metrics_summary_path = results_dir / "metrics.csv"
        summary_columns = [
            column
            for column in history_df.columns
            if column.startswith("val_") or column in {"train_loss", "epoch"}
        ]
        if summary_columns:
            history_df[summary_columns].to_csv(metrics_summary_path, index=False)

        logger.info("Saved training artifacts to %s.", results_dir)

    def close(self) -> None:
        """Close TensorBoard writer and release resources."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            logger.info("TensorBoard writer closed.")


def train_model(config: Optional[Config] = None) -> pd.DataFrame:
    """Convenience entry point to train a model using project defaults.

    Args:
        config: Optional project configuration.

    Returns:
        Training history DataFrame.
    """
    config = config or get_default_config()
    trainer = Trainer(config=config)
    return trainer.train()


def create_trainer_from_config(config: Optional[Config] = None) -> Trainer:
    """Create a configured Trainer instance.

    Args:
        config: Optional project configuration.

    Returns:
        Initialized Trainer.
    """
    return Trainer(config=config or get_default_config())
