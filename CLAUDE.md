# TP3 — 8INF804 Traitement numérique des images

## Project Goal
Compare two CNN approaches for binary image classification (AI-generated art vs Real art):
1. **Transfer learning** on a pretrained ResNet-50
2. **Custom CNN** architecture built from scratch

Each approach is trained 5 independent runs to account for stochastic behavior. Performance is compared using accuracy, Cohen's kappa, classification report, and learning curves.

## Dataset
- Location: `./data/`
- Classes: `AiArtData` (539 images), `RealArt` (436 images)
- Split: 70% train / 15% validation / 15% test — fixed with `SEED=42`, same split for both models
- Labels are auto-assigned integers by `ImageFolder`: `AiArtData=0`, `RealArt=1`

## Project Structure
```
tp3/
├── data/
│   ├── AiArtData/       # AI-generated art images
│   └── RealArt/         # Real art images
├── src/
│   ├── main.py          # Entry point
│   ├── config.py        # All constants and hyperparameters
│   ├── models/
│   │   └── finetuning.py    # ResNet-50 transfer learning setup
│   └── training/
│       ├── dataset.py   # get_sets() — loads and splits the dataset
│       ├── loaders.py   # setup_loader() — wraps sets in DataLoader
│       └── train.py     # train_model() — training loop (WIP)
├── environment.yml
└── CLAUDE.md
```

## Key Technical Decisions
- **Loss function**: `CrossEntropyLoss` with integer labels (no one-hot encoding needed)
- **Transfer learning normalization**: ImageNet stats `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- **Custom CNN normalization**: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`
- **Input size**: `(224, 224)` — required by ResNet-50
- **Resize strategy**: `Resize((224, 224))` direct (preserves all content, avoids edge cropping for art images)
- **Pretrained model**: `ResNet50_Weights.IMAGENET1K_V2`
- **Device**: auto-detected (`cuda:0` if available, else `cpu`)

## Config (`src/config.py`)
| Constant | Value | Notes |
|---|---|---|
| `DATASET_PATH` | `./data` | |
| `DATASET_DISTRIBUTION` | `[0.7, 0.15, 0.15]` | train/val/test |
| `SEED` | `42` | fixed split + reproducibility |
| `LEARNING_RATE` | `0.005` | for custom CNN |
| `BATCH_SIZE` | `8` | |
| `EPOCH_NUMBER` | `5` | |
| `RESULT_PATH` | `./results` | |

## Still To Implement
- Freeze/unfreeze layers in `FineTuning` model
- Training loop with loss/accuracy tracking in `train.py`
- Custom CNN architecture in `models/`
- 5-run loop with different seeds
- Metrics: accuracy, Cohen's kappa, `classification_report()`
- Learning curves (train vs val loss/accuracy per run)

## Running the Project
```bash
cd src
python main.py
```

## Environment
```bash
conda activate vision
```
