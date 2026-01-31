# ML Models

## Responsibility

Defines and trains the prediction models for NBA game outcomes. Contains the Transformer encoder for game flow sequences, GATv2 graph neural network for player interactions, and Two-Tower fusion architecture for combining static and dynamic features.

## Status

✅ **Phase 4 - Complete**

## Structure

| File | Purpose | Key Exports |
|------|---------|-------------|
| `__init__.py` | Package public API | All model classes |
| `transformer.py` | Sequence model | `GameFlowTransformer`, `EventTokenizer`, `TokenizedSequence` |
| `gnn.py` | Player graph model | `PlayerInteractionGNN`, `LineupGraphBuilder`, `PlayerFeatures` |
| `fusion.py` | Two-tower fusion | `TwoTowerFusion`, `ContextFeatureBuilder`, `MultiTaskLoss` |
| `trainer.py` | Training pipeline | `FusionTrainer`, `TrainingConfig`, `TrainingHistory` |
| `dataset.py` | Data loading | `NBADataset`, `GameSample`, `nba_collate_fn` |
| `registry.py` | Model versioning | `ModelRegistry`, `VersionInfo` |

## Architecture Overview

```
Context Features ─────────────────────────────────┐
(32-dim: team stats, fatigue, RAPM, spacing)      │
                                                   ├──→ TwoTowerFusion ──→ [P(home), margin, total]
Play-by-Play ──→ EventTokenizer ──→ Transformer ──┤
(seq_len=50)           (vocab=15)    (d=128)      │
                                                   │
Lineup Graph ──→ LineupGraphBuilder ──→ GNN ──────┘
(10 nodes)           (16 features)    (d=128)
```

## Key Classes

| Class | Params | Output Shape | Purpose |
|-------|--------|--------------|---------|
| `GameFlowTransformer` | ~200K | (batch, 128) | Encode game flow |
| `PlayerInteractionGNN` | ~100K | (batch, 128) | Model player interactions |
| `TwoTowerFusion` | ~150K | 3 heads | Multi-task prediction |
| `FusionTrainer` | - | TrainingHistory | Training orchestration |
| `NBADataset` | - | GameSample | Data pipeline |
| `ModelRegistry` | - | - | Version management |

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `d_model` | 128 | transformer.py |
| `nhead` | 4 | transformer.py, gnn.py |
| `num_layers` | 2 | transformer.py, gnn.py |
| `seq_len` | 50 | transformer.py, dataset.py |
| `node_features` | 16 | gnn.py |
| `hidden_dim` | 64 (GNN), 256 (Fusion) | gnn.py, fusion.py |
| `dropout` | 0.1 (models), 0.2 (fusion) | All model files |
| `learning_rate` | 1e-4 | config.py |
| `weight_decay` | 1e-5 | trainer.py |
| `gradient_clip` | 1.0 | trainer.py |
| `patience` | 10 | trainer.py |

## Loss Functions

- **Win Probability:** Binary cross-entropy with sigmoid
- **Margin:** Huber loss (delta=1.0) for outlier robustness
- **Total Points:** Huber loss (delta=1.0)
- **Multi-task:** Weighted sum (1.0 each, optionally learnable)

## Validation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Win Accuracy | > 55% | Prediction correctness |
| Brier Score | < 0.25 | Probability calibration |
| Log Loss | < 0.68 | Cross-entropy |
| Margin MAE | < 8 pts | Mean absolute error |
| Total MAE | < 12 pts | Mean absolute error |

## CLI Commands

```bash
# Train all models
python -m nba_model.cli train all --epochs 50

# Train individual components
python -m nba_model.cli train transformer --epochs 50
python -m nba_model.cli train gnn --epochs 50
python -m nba_model.cli train fusion --epochs 50

# List model versions
python -m nba_model.cli train list

# Compare versions
python -m nba_model.cli train compare v1.0.0 v1.1.0
```

## Integration Points

**Upstream (inputs from):**
- `data/` - Plays, stints, player stats (via NBADataset)
- `features/` - RAPM coefficients, spacing metrics (via ContextFeatureBuilder)

**Downstream (outputs to):**
- `predict/` - Loads trained models for inference
- `backtest/` - Uses models for historical evaluation
- `monitor/` - Compares model versions for drift

## Usage Examples

### Training Pipeline

```python
from nba_model.models import (
    GameFlowTransformer,
    PlayerInteractionGNN,
    TwoTowerFusion,
    FusionTrainer,
    NBADataset,
    nba_collate_fn,
)
from torch.utils.data import DataLoader

# Initialize models
transformer = GameFlowTransformer(vocab_size=15)
gnn = PlayerInteractionGNN(node_features=16)
fusion = TwoTowerFusion(context_dim=32)

# Create dataset
dataset = NBADataset.from_season("2023-24", db_session)
train_loader = DataLoader(dataset, batch_size=32, collate_fn=nba_collate_fn)

# Train
trainer = FusionTrainer(transformer, gnn, fusion)
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Model Registry

```python
from nba_model.models import ModelRegistry

registry = ModelRegistry()

# Save model
registry.save_model(
    version="1.0.0",
    models={"transformer": transformer, "gnn": gnn, "fusion": fusion},
    metrics={"accuracy": 0.58, "brier_score": 0.23},
    config={"d_model": 128, "learning_rate": 1e-4},
)

# Load model
weights = registry.load_model("latest")
transformer.load_state_dict(weights["transformer"])
```

### Tokenization

```python
from nba_model.models.transformer import EventTokenizer

tokenizer = EventTokenizer(max_seq_len=50)
tokens = tokenizer.tokenize_game(plays_df)
# tokens.events, tokens.times, tokens.scores, tokens.lineups, tokens.mask
```

### Graph Building

```python
from nba_model.models.gnn import LineupGraphBuilder

builder = LineupGraphBuilder(player_features_df)
graph = builder.build_graph(
    home_lineup=[203507, 2544, 201566, 203076, 1628389],
    away_lineup=[201142, 201935, 203954, 1628369, 203507],
)
# graph.x: (10, 16), graph.edge_index: (2, num_edges)
```

## Model Storage

```
data/models/
├── v1.0.0/
│   ├── transformer.pt      # State dict (~200K params)
│   ├── gnn.pt              # State dict (~100K params)
│   ├── fusion.pt           # State dict (~150K params)
│   └── metadata.json       # Training metadata
├── v1.1.0/
│   └── ...
└── latest -> v1.1.0        # Symlink to current
```

## Anti-Patterns

- ❌ Never train without walk-forward validation (temporal split)
- ❌ Never save models without version metadata
- ❌ Never use MSE for margin (use Huber for outlier robustness)
- ❌ Never mix train/test temporal data (causes leakage)
- ❌ Never skip gradient clipping (models can diverge)
- ❌ Never hardcode paths (use config.get_settings())
- ❌ Never ignore padding mask in Transformer
- ❌ Never batch graphs without PyG Batch.from_data_list()
