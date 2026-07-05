"""Shared utility functions for the Chest X-ray multi-label research pipeline.

Provides logging setup, reproducibility helpers, device management, experiment
directory creation, configuration export, and system diagnostics.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from src.config import Config, PROJECT_ROOT, get_default_config


logger = logging.getLogger(__name__)


def setup_logging(
    config: Optional[Config] = None,
    log_level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_filename: Optional[str] = None,
) -> None:
    """Configure root logging for console and optional file output.

    Args:
        config: Optional project configuration.
        log_level: Optional logging level override.
        log_to_file: Optional flag to enable file logging.
        log_filename: Optional log file name override.
    """
    config = config or get_default_config()
    level_name = (log_level or config.logging.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    enable_file_logging = (
        config.logging.log_to_file if log_to_file is None else log_to_file
    )
    if enable_file_logging:
        file_name = log_filename or config.logging.log_filename
        log_path = config.paths.results_dir / file_name
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", "") == str(log_path)
            for handler in root_logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info("File logging configured at %s.", log_path)

    logger.debug("Logging configured with level=%s.", level_name)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible deep learning experiments.

    Args:
        seed: Global random seed value.
        deterministic: Enable deterministic cuDNN behavior on CUDA devices.

    Raises:
        ValueError: If seed is negative.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Random seed set to %d (deterministic=%s).", seed, deterministic)


def resolve_device(
    device: Optional[Union[str, torch.device]] = None,
    config: Optional[Config] = None,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """Resolve the compute device from override or configuration.

    Args:
        device: Optional explicit device string or ``torch.device``.
        config: Optional project configuration.
        fallback_to_cpu: Fall back to CPU when CUDA is unavailable.

    Returns:
        Resolved torch device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable and fallback is disabled.
    """
    if device is not None:
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            if fallback_to_cpu:
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
                return torch.device("cpu")
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        return device_obj

    config = config or get_default_config()
    requested = config.device.device

    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested if requested != "cuda" else "cuda:0")

    if requested.startswith("cuda"):
        if fallback_to_cpu:
            logger.warning("CUDA requested in config but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError("CUDA requested in config but CUDA is not available.")

    return torch.device("cpu")


def get_gpu_memory_mb(device: Optional[torch.device] = None) -> Optional[float]:
    """Return peak allocated GPU memory in megabytes.

    Args:
        device: Optional CUDA device.

    Returns:
        Peak allocated GPU memory in MB, or ``None`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return None

    device_obj = device or torch.device("cuda")
    if device_obj.type != "cuda":
        return None

    return float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2))


def reset_gpu_memory_stats(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak memory statistics.

    Args:
        device: Optional CUDA device.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device or torch.device("cuda"))


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    if seconds < 0:
        raise ValueError("seconds must be >= 0.")

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:.2f}s"


def get_timestamp(format_string: str = "%Y%m%d-%H%M%S") -> str:
    """Return the current timestamp as a formatted string.

    Args:
        format_string: ``datetime.strftime`` format string.

    Returns:
        Timestamp string.
    """
    return datetime.now().strftime(format_string)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        Resolved directory path.

    Raises:
        OSError: If directory creation fails.
    """
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create directory: %s", directory)
        raise OSError(f"Unable to create directory: {directory}") from exc
    return directory.resolve()


def create_experiment_directory(
    config: Optional[Config] = None,
    base_dir: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
) -> Path:
    """Create a timestamped experiment directory under the results path.

    Args:
        config: Optional project configuration.
        base_dir: Optional base directory override.
        experiment_name: Optional experiment name override.

    Returns:
        Path to the created experiment directory.
    """
    config = config or get_default_config()
    name = experiment_name or config.logging.experiment_name
    root = Path(base_dir) if base_dir is not None else config.paths.results_dir
    experiment_dir = ensure_directory(root / name / get_timestamp())
    logger.info("Created experiment directory at %s.", experiment_dir)
    return experiment_dir


def save_json(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    indent: int = 2,
) -> None:
    """Save a dictionary to a JSON file.

    Args:
        data: JSON-serializable dictionary.
        output_path: Destination JSON file path.
        indent: JSON indentation level.

    Raises:
        OSError: If writing the file fails.
        TypeError: If data is not JSON serializable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=indent)
    except TypeError as exc:
        logger.exception("JSON serialization failed for %s.", output_path)
        raise TypeError(f"Data is not JSON serializable: {output_path}") from exc
    except Exception as exc:
        logger.exception("Failed to write JSON file: %s", output_path)
        raise OSError(f"Unable to write JSON file: {output_path}") from exc

    logger.info("Saved JSON file to %s.", output_path)


def save_config_json(
    config: Optional[Config] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Export the active project configuration to JSON.

    Args:
        config: Optional project configuration.
        output_path: Optional output JSON path.

    Returns:
        Path to the saved configuration file.
    """
    config = config or get_default_config()
    destination = Path(output_path) if output_path is not None else (
        config.paths.results_dir / "config.json"
    )
    save_json(config.to_dict(), destination)
    return destination


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file into a dictionary.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        RuntimeError: If parsing fails.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON file: %s", json_path)
        raise RuntimeError(f"Unable to parse JSON file: {json_path}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {json_path}.")
    return data


def get_system_info() -> Dict[str, Any]:
    """Collect system, Python, and PyTorch runtime information.

    Returns:
        Dictionary describing the current runtime environment.
    """
    info: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name(current_device)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(current_device)

    return info


def log_system_info(config: Optional[Config] = None) -> Dict[str, Any]:
    """Log runtime system and hardware information.

    Args:
        config: Optional project configuration for context logging.

    Returns:
        System information dictionary.
    """
    info = get_system_info()
    logger.info("Project root: %s", info["project_root"])
    logger.info("Platform: %s", info["platform"])
    logger.info("Python: %s", info["python_version"])
    logger.info("PyTorch: %s", info["torch_version"])
    logger.info("CUDA available: %s", info["cuda_available"])

    if info["cuda_available"]:
        logger.info("CUDA devices: %d", info["cuda_device_count"])
        logger.info("CUDA device name: %s", info.get("cuda_device_name"))

    if config is not None:
        logger.info("Experiment: %s", config.logging.experiment_name)
        logger.info("Model: %s", config.model.model_name.value)

    return info


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a PyTorch model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def initialize_experiment(
    config: Optional[Config] = None,
    seed: Optional[int] = None,
    create_dirs: bool = True,
) -> Config:
    """Initialize logging, reproducibility, and experiment directories.

    Args:
        config: Optional project configuration.
        seed: Optional seed override.
        create_dirs: Whether to create output directories.

    Returns:
        Initialized configuration object.
    """
    config = config or get_default_config()

    if create_dirs:
        config.setup()

    setup_logging(config=config)
    set_seed(
        seed if seed is not None else config.training.random_seed,
        deterministic=config.training.deterministic,
    )
    save_config_json(config=config)
    log_system_info(config=config)

    logger.info("Experiment initialization complete.")
    return config


class Timer:
    """Simple context manager for measuring elapsed time.

    Args:
        name: Optional timer label used in logs.
        log_result: Whether to log elapsed time on exit.
    """

    def __init__(self, name: str = "Operation", log_result: bool = True) -> None:
        """Initialize the timer."""
        self.name = name
        self.log_result = log_result
        self.start_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the timer and optionally log elapsed time."""
        self.elapsed_seconds = time.perf_counter() - self.start_time
        if self.log_result:
            logger.info("%s completed in %s.", self.name, format_duration(self.elapsed_seconds))


def seed_worker(worker_id: int) -> None:
    """Seed numpy and random in DataLoader worker processes.

    Args:
        worker_id: Worker process ID provided by PyTorch DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_generator(seed: int) -> torch.Generator:
    """Create a torch Generator for reproducible DataLoader shuffling.

    Args:
        seed: Random seed value.

    Returns:
        Seeded torch Generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def print_config_summary(config: Optional[Config] = None) -> None:
    """Print a concise summary of the active configuration.

    Args:
        config: Optional project configuration.
    """
    config = config or get_default_config()
    summary_lines = [
        "=== Configuration Summary ===",
        f"Experiment: {config.logging.experiment_name}",
        f"Model: {config.model.model_name.value}",
        f"Loss: {config.loss.loss_name.value}",
        f"Image Size: {config.data.image_size}",
        f"Batch Size: {config.training.batch_size}",
        f"Epochs: {config.training.num_epochs}",
        f"Learning Rate: {config.optimizer.learning_rate}",
        f"Device: {config.device.device}",
        f"Train CSV: {config.paths.train_csv}",
        f"Val CSV: {config.paths.val_csv}",
        f"Test CSV: {config.paths.test_csv}",
        f"Results Dir: {config.paths.results_dir}",
        "=============================",
    ]
    for line in summary_lines:
        logger.info(line)


def checkpoint_exists(config: Optional[Config] = None, best: bool = True) -> bool:
    """Check whether a model checkpoint file exists.

    Args:
        config: Optional project configuration.
        best: Check best model checkpoint when ``True``, else latest checkpoint.

    Returns:
        ``True`` if the checkpoint exists.
    """
    config = config or get_default_config()
    checkpoint_path = (
        config.paths.best_model_path if best else config.paths.latest_checkpoint_path
    )
    return checkpoint_path.exists()


def get_checkpoint_metadata(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load lightweight metadata from a checkpoint without restoring weights.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Dictionary containing epoch, best metric, and config snapshot when available.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint cannot be read.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        logger.exception("Failed to read checkpoint metadata from %s.", checkpoint_path)
        raise RuntimeError(f"Unable to read checkpoint: {checkpoint_path}") from exc

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary.")

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "best_metric": checkpoint.get("best_metric"),
        "epochs_without_improvement": checkpoint.get("epochs_without_improvement"),
        "has_optimizer_state": "optimizer_state_dict" in checkpoint,
        "has_scheduler_state": checkpoint.get("scheduler_state_dict") is not None,
        "config": checkpoint.get("config"),
    }
    return metadata
