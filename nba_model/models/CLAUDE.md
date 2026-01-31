# ML Models

## Responsibility

Defines and trains the prediction models: Transformer for game flow, GNN for player interactions, and Two-Tower fusion for final predictions.

## Status

🔲 **Phase 4 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Architecture |
|------|---------|--------------|
| `__init__.py` | Model exports | - |
| `transformer.py` | Sequence model | GameFlowTransformer (d=128, heads=4) |
| `gnn.py` | Player graph | PlayerInteractionGNN (GATv2) |
| `fusion.py` | Two-tower combo | TwoTowerFusion (context + dynamic) |
| `trainer.py` | Training loop | Multi-task loss, validation |
| `dataset.py` | Data loading | PyTorch Dataset classes |
| `registry.py` | Model versioning | Save/load with metadata |

## Architecture Overview

```
Context Features ─────┐
                      ├──→ TwoTowerFusion ──→ [P(home), margin, total]
Dynamic Features ─────┤
                      │
Game Flow Sequence ───┼──→ Transformer ──┐
                                         ├──→ Fusion Input
Player Graph ─────────┼──→ GNN ──────────┘
```

## Key Hyperparameters

| Parameter | Value | Location |
|-----------|-------|----------|
| `d_model` | 128 | transformer.py |
| `nhead` | 4 | transformer.py, gnn.py |
| `learning_rate` | 1e-4 | config.py |
| `batch_size` | 32 | config.py |

## Loss Functions

- **Classification:** Binary cross-entropy (win/loss)
- **Regression:** Huber loss (margin, total) for outlier robustness
- **Multi-task:** Weighted sum with learned weights

## Anti-Patterns

- ❌ Never train without walk-forward validation
- ❌ Never save models without version metadata
- ❌ Never use MSE for margin (use Huber)
- ❌ Never mix train/test temporal data
