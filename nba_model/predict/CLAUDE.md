# Prediction Pipeline

## Responsibility

Runs inference for upcoming games, applies injury adjustments, and generates betting signals with confidence scores.

## Status

🔲 **Phase 7 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Key Functions |
|------|---------|---------------|
| `__init__.py` | Public API | - |
| `inference.py` | Model inference | `InferencePipeline.predict()` |
| `injuries.py` | Injury adjustments | `apply_injury_adjustment()` |
| `signals.py` | Betting signals | `generate_signals()`, `Signal` |

## Prediction Flow

```
Today's Games → Feature Extraction → Model Inference
                                          ↓
                               Apply Injury Adjustments
                                          ↓
                               Compare to Market Odds
                                          ↓
                               Generate Signals (if edge > min_edge)
```

## Signal Generation

| Field | Description |
|-------|-------------|
| `game_id` | NBA game identifier |
| `bet_type` | ML (moneyline), spread, total |
| `model_prob` | Model's probability estimate |
| `market_prob` | Implied probability from odds |
| `edge` | model_prob - market_prob |
| `kelly_size` | Recommended bet size |
| `confidence` | Model confidence (0-1) |

## Injury Adjustment

Bayesian adjustment using historical RAPM:
```
adjusted_prob = base_prob * (1 - Σ(rapm_i * minutes_i / 48))
```

## Integration Points

- **Upstream:** `models/` for inference, `features/` for extraction
- **Downstream:** `output/` for dashboard display

## Anti-Patterns

- ❌ Never generate signals without checking injury reports
- ❌ Never bet on games within 2 hours (lines move)
- ❌ Never use stale model (check version date)
- ❌ Never ignore confidence scores in sizing
