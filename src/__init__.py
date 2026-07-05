"""MedicalAI-Thesis: Explainable multi-label Chest X-ray disease classification."""

from src.config import (
    Config,
    ModelName,
    LossName,
    SplitName,
    NIH_DISEASE_LABELS,
    NUM_CLASSES,
    build_config,
    get_default_config,
    DEFAULT_CONFIG,
)
from src.evaluate import (
    Evaluator,
    run_full_evaluation_pipeline,
    run_inference_on_image,
    evaluate_test,
    evaluate_validation,
)
from src.dataset import unpack_batch, unpack_sample
from src.gradcam import explain_prediction, GradCAMExplainer
from src.trainer import Trainer, train_model, create_trainer_from_config
from src.utils import initialize_experiment, setup_logging, set_seed

__all__ = [
    "Config",
    "ModelName",
    "LossName",
    "SplitName",
    "NIH_DISEASE_LABELS",
    "NUM_CLASSES",
    "build_config",
    "get_default_config",
    "DEFAULT_CONFIG",
    "Evaluator",
    "run_full_evaluation_pipeline",
    "run_inference_on_image",
    "evaluate_test",
    "evaluate_validation",
    "unpack_batch",
    "unpack_sample",
    "explain_prediction",
    "GradCAMExplainer",
    "Trainer",
    "train_model",
    "create_trainer_from_config",
    "initialize_experiment",
    "setup_logging",
    "set_seed",
]

__version__ = "1.0.0"
