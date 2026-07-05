# Explainable Deep Learning Framework for Multi-Label Chest X-ray Disease Classification

Research-grade PyTorch pipeline for **NIH ChestX-ray14** multi-label disease classification with training, evaluation, and Grad-CAM explainability.

## Project Overview

| Item | Detail |
|---|---|
| **Dataset** | NIH ChestX-ray14 |
| **Task** | Multi-label classification (15 labels) |
| **Models** | DenseNet121, EfficientNet-B0, ConvNeXt-Tiny |
| **Loss** | Weighted BCEWithLogitsLoss, optional Focal Loss |
| **Explainability** | Grad-CAM, Grad-CAM++ |

## Project Structure

```
MedicalAI-Thesis/
├── src/
│   ├── __init__.py
│   ├── config.py          # All hyperparameters and paths
│   ├── transforms.py      # Train/val/inference preprocessing
│   ├── dataset.py         # Dataset and DataLoader factories
│   ├── models.py          # Model registry and factory
│   ├── losses.py          # Loss registry and factory
│   ├── metrics.py         # Metrics, plots, CSV export
│   ├── trainer.py         # Training loop with AMP, early stopping
│   ├── evaluate.py        # Validation, test, and inference
│   ├── gradcam.py         # Grad-CAM explainability
│   └── utils.py           # Logging, seeding, experiment setup
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── images/            # Chest X-ray PNG files
├── checkpoints/           # Saved model weights
├── results/               # Metrics, plots, predictions
├── runs/                  # TensorBoard logs
├── requirements.txt
└── README.md
```

## Installation

```bash
cd MedicalAI-Thesis
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

## Dataset Setup

Place patient-wise split CSV files under `data/`:

| File | Description |
|---|---|
| `train.csv` | Training split |
| `val.csv` | Validation split |
| `test.csv` | Test split |

Each CSV must include:

- `Image_Path` — relative or absolute path to the image
- 15 binary label columns (0/1 multi-label encoding)

Expected label columns:

`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Hernia`, `No Finding`

## DataLoader API (Dictionary-Based)

This project uses a **dictionary-based pipeline**, not tuple unpacking.

Each batch/sample contains:

| Key | Type | Description |
|---|---|---|
| `image` | `Tensor (B, 3, H, W)` | Preprocessed image tensor |
| `labels` | `Tensor (B, 15)` | Multi-hot disease labels |
| `image_path` | `list[str]` | Resolved image paths |
| `index` | `list[int]` | Sample indices |

### Load a batch

```python
from src.dataset import create_dataloader, unpack_batch
from src.config import SplitName

train_loader = create_dataloader(SplitName.TRAIN)

batch = next(iter(train_loader))
images = batch["image"]
labels = batch["labels"]
paths = batch["image_path"]
index = batch["index"]

# Or use the helper:
images, labels, paths, indices = unpack_batch(batch)
```

### Load a single sample

```python
from src.dataset import create_dataset, unpack_sample

dataset = create_dataset("train")
sample = dataset[0]
image, label, path, idx = unpack_sample(sample)
```

### Visualize images (denormalized)

```python
import matplotlib.pyplot as plt
from src.transforms import tensor_to_pil_image

batch = next(iter(train_loader))
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    pil_img = tensor_to_pil_image(batch["image"][i])
    ax.imshow(pil_img)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## Quick Start

### 1. Train (notebook — two lines only)

```python
from src.trainer import Trainer

trainer = Trainer()
history = trainer.fit()
```

Resume training:

```python
trainer = Trainer()
history = trainer.fit(resume=True)
```

### 2. Initialize experiment (optional)

```python
from src.utils import initialize_experiment, print_config_summary
from src.trainer import Trainer

config = initialize_experiment()
print_config_summary(config)
trainer = Trainer(config=config)
history = trainer.fit()
```

### 3. Switch model (one line)

```python
from src.config import build_config, ModelConfig, ModelName
from src.trainer import Trainer

config = build_config(model=ModelConfig(model_name=ModelName.EFFICIENTNET_B0))
trainer = Trainer(config=config)
history = trainer.train()
```

### 4. Full evaluation pipeline

```python
from src.evaluate import run_full_evaluation_pipeline

reports = run_full_evaluation_pipeline()
print(reports["test"].metrics.auroc_macro)
```

### 5. Single-image inference

```python
from src.evaluate import run_inference_on_image

result = run_inference_on_image("data/images/sample.png")
print(result["predicted_diseases"])
```

### 6. Grad-CAM explanation

```python
from src.gradcam import explain_prediction

explanation = explain_prediction("data/images/sample.png")
print(explanation.target_class_name, explanation.probability)
```

## Configuration

All hyperparameters live in `src/config.py`. Key settings:

| Group | Examples |
|---|---|
| `TrainingConfig` | epochs, batch size, AMP, early stopping |
| `OptimizerConfig` | AdamW learning rate, weight decay |
| `ModelConfig` | architecture, pretrained, dropout |
| `LossConfig` | weighted BCE, focal loss, class weights |
| `EvaluationConfig` | threshold search, plot export |
| `GradCAMConfig` | gradcam / gradcam++, colormap, alpha |

Override at runtime:

```python
from src.config import build_config, TrainingConfig, ModelConfig, ModelName

config = build_config(
    model=ModelConfig(model_name=ModelName.CONVNEXT_TINY),
    training=TrainingConfig(num_epochs=30, batch_size=16),
)
```

## Training Features

- Mixed precision (AMP)
- Gradient clipping
- CosineAnnealingLR scheduler
- Early stopping
- Automatic checkpointing (best + latest)
- Resume training
- TensorBoard logging
- GPU memory logging
- Class-weighted loss from training label frequencies

## Outputs

After training and evaluation, `results/` contains:

| Artifact | Description |
|---|---|
| `training_log.csv` | Per-epoch metrics |
| `metrics.csv` | Validation metric summary |
| `predictions.csv` | Per-sample probabilities |
| `best_thresholds.csv` | Optimized per-class thresholds |
| `loss_curves.png` | Train vs validation loss |
| `learning_curves.png` | AUROC / F1 curves |
| `*_roc_curves.png` | ROC curves |
| `*_pr_curves.png` | Precision-recall curves |
| `*_confusion_matrices.png` | Per-class confusion matrices |
| `explainability/` | Grad-CAM heatmaps and overlays |

## Metrics

- Loss, accuracy, precision, recall, F1 (macro / micro / per-class)
- AUROC (macro / micro / per-class)
- Confusion matrices
- ROC and precision-recall curves

## Command-Line Usage

Run from the project root so `src` imports resolve:

```bash
# Train
python -c "from src.utils import initialize_experiment; from src.trainer import train_model; train_model(initialize_experiment())"

# Evaluate
python -c "from src.evaluate import run_full_evaluation_pipeline; run_full_evaluation_pipeline()"

# Explain one image
python -c "from src.gradcam import explain_prediction; explain_prediction('data/images/sample.png')"
```

## Resume Training

Set `resume=True` when calling `fit()`:

```python
trainer = Trainer()
history = trainer.fit(resume=True)
```

The trainer loads `checkpoints/last_model.pth` and restores model, optimizer, scheduler, and AMP scaler.

## Training Outputs

```
checkpoints/
├── best_model.pth
├── last_model.pth
├── checkpoint_epoch_001.pth
├── checkpoint_epoch_002.pth
└── ...

results/
├── training_log.csv
├── loss_curve.png
├── auroc_curve.png
├── lr_curve.png
├── precision_curve.png
├── recall_curve.png
├── f1_curve.png
└── tensorboard/
```

## License

Academic / thesis research use. NIH ChestX-ray14 dataset usage must follow the [NIH dataset terms](https://nihcc.app.box.com/v/ChestXray-NPNGCC).

## Citation

If you use this framework in academic work, cite the NIH ChestX-ray14 dataset and relevant model architectures (DenseNet, EfficientNet, ConvNeXt) as appropriate for your thesis.
