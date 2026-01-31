# Tests for ML Models

## Purpose

Unit tests for the `nba_model/models/` package covering all model components, training infrastructure, and versioning.

## Status

✅ **Complete** - All Phase 4 components tested.

## Structure

| File | Tests | Source Module |
|------|-------|---------------|
| `test_transformer.py` | Transformer encoder, tokenizer | `models/transformer.py` |
| `test_gnn.py` | GATv2 GNN, graph builder | `models/gnn.py` |
| `test_fusion.py` | Two-tower fusion, loss | `models/fusion.py` |
| `test_trainer.py` | Training loop, metrics | `models/trainer.py` |
| `test_dataset.py` | Dataset, collate, split | `models/dataset.py` |
| `test_registry.py` | Version management | `models/registry.py` |
| `conftest.py` | Shared fixtures | - |

## Test Coverage

- **GameFlowTransformer**: Forward pass, shapes, gradients, masks
- **EventTokenizer**: Tokenization, padding, collation
- **PlayerInteractionGNN**: Forward pass, batching, node embeddings
- **LineupGraphBuilder**: Graph construction, edge patterns
- **TwoTowerFusion**: Multi-task outputs, probability ranges
- **MultiTaskLoss**: Loss computation, gradient flow
- **FusionTrainer**: Training loop, checkpoints, metrics
- **NBADataset**: Data loading, labels, collation
- **ModelRegistry**: Save/load, versioning, comparison

## Fixtures (conftest.py)

| Fixture | Type | Description |
|---------|------|-------------|
| `sample_events` | Tensor | Event indices (2, 50) |
| `sample_times` | Tensor | Time features (2, 50, 1) |
| `sample_scores` | Tensor | Score features (2, 50, 1) |
| `sample_lineups` | Tensor | Lineup encoding (2, 50, 20) |
| `sample_graph` | Data | PyG graph (10 nodes) |
| `sample_context` | Tensor | Context features (2, 32) |
| `sample_plays_df` | DataFrame | Play-by-play events |
| `sample_games_df` | DataFrame | Game records |
| `transformer_model` | Model | Small transformer |
| `gnn_model` | Model | Small GNN |
| `fusion_model` | Model | Small fusion |

## Running Tests

```bash
# All model tests
pytest tests/unit/models/ -v

# Specific test file
pytest tests/unit/models/test_transformer.py -v

# With coverage
pytest tests/unit/models/ --cov=nba_model.models --cov-report=term-missing

# Skip slow tests
pytest tests/unit/models/ -m "not slow"
```

## Anti-Patterns

- ❌ Never use real database in unit tests
- ❌ Never test with large models (use small fixtures)
- ❌ Never share state between tests
- ❌ Never skip gradient flow tests
