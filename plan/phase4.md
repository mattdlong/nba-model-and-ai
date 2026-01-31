# Phase 4: Model Training

## Overview

Implement the machine learning architecture comprising a Transformer encoder for game flow sequences, GATv2 graph neural network for player interactions, and a Two-Tower fusion model that combines static context with dynamic representations. Establish multi-task training infrastructure and model versioning system.

## Dependencies

- Phase 2: Populated database tables (games, plays, stints, shots, player_game_stats)
- Phase 3: Computed RAPM coefficients in player_rapm table
- Phase 3: Computed lineup spacing metrics in lineup_spacing table
- Phase 3: Season normalization statistics in season_stats table

## Objectives

1. Implement Transformer encoder for sequence modeling of play-by-play events
2. Implement GATv2 graph neural network for player interaction modeling
3. Build Two-Tower fusion architecture combining context and dynamic features
4. Create multi-task training pipeline with proper loss weighting
5. Establish model registry with versioning and metadata tracking

---

## Task 4.1: Transformer Sequence Model

### Component: GameFlowTransformer

Implement a Transformer encoder that processes tokenized play-by-play sequences to capture game flow dynamics.

**Architecture Specifications:**
- Embedding dimension: 128
- Attention heads: 4
- Encoder layers: 2
- Maximum sequence length: 50 events
- Dropout rate: 0.1

**Input Representations:**
- Event type embedding: Categorical embedding for EVENTMSGTYPE values (made shot, missed shot, free throw, rebound, turnover, foul, violation, substitution, timeout, jump ball, period markers)
- Temporal embedding: Sinusoidal positional encoding for sequence position
- Time embedding: Linear projection of normalized game clock remaining
- Score embedding: Linear projection of normalized score differential
- Lineup embedding: Linear projection of 20-dimensional one-hot lineup encoding (10 players, home/away indicator)

**Output:**
- Fixed-dimension sequence representation (128-dim) via CLS token pooling or mean pooling over sequence positions

### Component: EventTokenizer

Implement tokenization logic to convert raw play-by-play data into model-ready tensors.

**Responsibilities:**
- Map EVENTMSGTYPE integers to vocabulary indices
- Normalize time remaining to [0, 1] range
- Normalize score differential by historical standard deviation
- Encode current lineup configuration per event
- Handle variable-length sequences with padding and attention masking

---

## Task 4.2: Graph Neural Network (GATv2)

### Component: PlayerInteractionGNN

Implement a GATv2-based graph neural network that models player interactions within lineup matchups.

**Architecture Specifications:**
- Node feature dimension: 16 input features
- Hidden dimension: 64
- Output dimension: 128
- Attention heads: 4
- GATv2 layers: 2
- Dropout rate: 0.1

**Graph Structure:**
- Nodes: 10 players (5 home, 5 away)
- Teammate edges: Fully connected within each team (20 edges total)
- Opponent edges: Positional matchups between teams (5 edges minimum, extensible to full cross-team connectivity)
- Edge attributes: Optional chemistry weights based on minutes shared

**Node Features:**
- RAPM components: ORAPM, DRAPM, total RAPM (3 features)
- Physical attributes: Normalized height, weight (2 features)
- Position embedding: One-hot or learned position representation (5 features)
- Shooting profile: FG%, 3P%, FT% season averages (3 features)
- Usage indicators: Minutes per game, usage rate (3 features)

**Output:**
- Graph-level representation (128-dim) via global mean/max pooling over nodes

### Component: LineupGraphBuilder

Implement graph construction logic to create PyTorch Geometric Data objects from lineup information.

**Responsibilities:**
- Look up player features from feature tables
- Construct edge_index tensor for teammate and opponent connectivity
- Optionally compute chemistry edge weights from historical lineup co-occurrence
- Handle missing player features gracefully with zero-padding or learned defaults

---

## Task 4.3: Two-Tower Fusion Architecture

### Component: TwoTowerFusion

Implement the fusion model that combines static context features with dynamic sequence and graph representations.

**Tower A (Context Tower):**
- Input: 32-dimensional static feature vector
- Architecture: 2-layer MLP (32 -> 256 -> 256) with ReLU activations and dropout

**Tower B (Dynamic Tower):**
- Input: Concatenated Transformer output (128-dim) and GNN output (128-dim) = 256-dim
- Architecture: 2-layer MLP (256 -> 256 -> 256) with ReLU activations and dropout

**Fusion Layer:**
- Concatenate Tower A and Tower B outputs (512-dim)
- 2-layer MLP (512 -> 256 -> 128) with ReLU and dropout

**Multi-Task Heads:**
- Win probability head: Linear projection to 1 output with sigmoid activation
- Margin prediction head: Linear projection to 1 output (regression)
- Total points head: Linear projection to 1 output (regression)

### Component: ContextFeatureBuilder

Implement feature extraction logic for Tower A static inputs.

**Feature Categories:**
- Team offensive/defensive ratings (z-scored by season): 4 features
- Team efficiency metrics (EFG%, TOV%, ORB%, FT rate): 8 features
- Rest and fatigue indicators: rest days, back-to-back flags, travel miles: 6 features
- Lineup RAPM aggregates: sum of ORAPM, DRAPM for each team: 6 features
- Spacing metrics: hull area for each team: 2 features
- Derived differentials: RAPM diff, rest diff: 6 features

---

## Task 4.4: Training Pipeline

### Component: FusionTrainer

Implement the training loop that jointly optimizes all model components.

**Loss Function:**
- Multi-task weighted loss combining:
  - Binary cross-entropy for win probability
  - Huber loss for margin prediction (robust to outliers)
  - Huber loss for total points prediction
- Configurable loss weights (default: 1.0 for each task)

**Optimization:**
- Optimizer: AdamW with weight decay (1e-5)
- Learning rate: 1e-4 with cosine annealing or reduce-on-plateau scheduling
- Gradient clipping: Max norm 1.0

**Training Protocol:**
- Early stopping with patience (default: 10 epochs)
- Checkpoint best model by validation loss
- Log training metrics per epoch

**Validation Metrics:**
- Win prediction accuracy
- Margin MAE (mean absolute error)
- Total points MAE
- Brier score for probability calibration
- Log loss (cross-entropy)

### Component: NBADataset

Implement PyTorch Dataset that assembles complete training samples.

**Sample Contents:**
- Tokenized play-by-play sequence tensors (for Transformer)
- PyG Data object with lineup graph (for GNN)
- Context feature vector (for Tower A)
- Labels: win outcome (binary), point margin (float), total points (float)

**Data Loading Considerations:**
- Efficient batching with collate function for variable-length sequences
- PyG batching for graph data
- Stratified sampling to balance home/away wins if needed

---

## Task 4.5: Model Registry

### Component: ModelRegistry

Implement versioned model storage and retrieval system.

**Storage Structure:**
```
data/models/
├── v1.0.0/
│   ├── transformer.pt
│   ├── gnn.pt
│   ├── fusion.pt
│   └── metadata.json
├── v1.1.0/
│   └── ...
└── latest -> v1.1.0 (symlink)
```

**Metadata Contents:**
- Version string (semantic versioning: major.minor.patch)
- Training timestamp
- Training data date range
- Hyperparameter configuration
- Validation metrics dictionary
- Git commit hash (if available)
- Parent version (for lineage tracking)

**Registry Operations:**
- save_model: Serialize model weights and metadata to versioned directory
- load_model: Deserialize model weights by version (or "latest")
- list_versions: Return all versions with summary metrics
- compare_versions: Side-by-side metric comparison between two versions

---

## Technical Specifications

| Component | Approximate Parameters | Training Time per Epoch | GPU Memory |
|-----------|------------------------|-------------------------|------------|
| Transformer | 200K | 10 minutes | 500MB |
| GNN | 100K | 5 minutes | 300MB |
| Fusion | 150K | 5 minutes | 200MB |
| **Total** | 450K | ~20 minutes | ~1GB |

**Training Configuration Defaults:**
- Epochs: 50 maximum
- Early stopping patience: 10 epochs
- Batch size: 32 games
- Validation split: 20% of training data (temporal split)

---

## Success Criteria

1. All three model components (Transformer, GNN, Fusion) instantiate and forward pass without errors
2. Training loop completes 50 epochs on historical data without memory overflow
3. Validation accuracy exceeds 55% on held-out games
4. Brier score below 0.25 indicating reasonable calibration
5. Model registry successfully saves and loads versioned checkpoints
6. Metadata.json contains all required fields with valid values

---

## Testing Requirements

### Unit Tests

**Transformer Tests:**
- Verify output shape matches (batch_size, 128) for valid inputs
- Verify attention mask correctly ignores padded positions
- Test gradient flow through all embedding layers

**GNN Tests:**
- Verify output shape matches (batch_size, 128) for valid graph batches
- Test with varying graph sizes (full 10 players vs. partial lineups)
- Verify edge construction produces expected connectivity patterns

**Fusion Tests:**
- Verify all three output heads produce correct shapes
- Test forward pass with mock inputs from each tower
- Verify multi-task loss computation is numerically stable

**Registry Tests:**
- Round-trip test: save model, load model, verify weights match exactly
- Verify metadata.json schema compliance
- Test list_versions returns correct ordering
- Test "latest" symlink resolution

### Integration Tests

**End-to-End Training Test:**
- Load small subset of real data (100 games)
- Run 5 training epochs
- Verify loss decreases monotonically (or near-monotonically)
- Verify checkpoint saved on best validation loss
- Verify training metrics logged correctly

**Dataset Tests:**
- Verify NBADataset returns correctly shaped tensors
- Test collate function produces valid batches
- Verify no NaN values in features for completed games

### Validation Tests

**Smoke Test on Historical Season:**
- Train on 2022-23 season data
- Evaluate on first 100 games of 2023-24 season
- Win accuracy should exceed 52% (better than random)
- Predictions should span reasonable probability range [0.3, 0.7]

**Calibration Test:**
- Bin predictions by decile
- Compute actual win rate per bin
- Verify monotonic relationship (higher predicted prob = higher actual win rate)

---

## File Locations

| Component | Path |
|-----------|------|
| Transformer model | `nba_model/models/transformer.py` |
| GNN model | `nba_model/models/gnn.py` |
| Fusion model | `nba_model/models/fusion.py` |
| Training pipeline | `nba_model/models/trainer.py` |
| Model registry | `nba_model/models/registry.py` |
| Unit tests | `tests/test_models/` |

---

## Completion Instructions

Upon completion, commit all changes with message "feat: implement phase 4 - model training" and push to origin.
