# Prediction Pipeline

## Responsibility

Runs inference for upcoming games, applies Bayesian injury adjustments, and generates betting signals with confidence scores and Kelly criterion sizing.

## Status

✅ **Phase 7 - Complete**

## Module Structure

| File | Purpose | Key Classes |
|------|---------|-------------|
| `__init__.py` | Public API | Exports all public components |
| `inference.py` | Model inference | `InferencePipeline`, `GamePrediction`, `PredictionBatch` |
| `injuries.py` | Injury adjustments | `InjuryAdjuster`, `InjuryReport`, `InjuryReportFetcher` |
| `signals.py` | Betting signals | `SignalGenerator`, `BettingSignal`, `MarketOdds` |

## Prediction Flow

```
Today's Games → InferencePipeline.predict_today()
                      │
                      ├─→ Load models from ModelRegistry
                      │
                      ├─→ Build context features (32-dim)
                      │
                      ├─→ Get Transformer output (zeros for pre-game)
                      │
                      ├─→ Get GNN output (lineup graph)
                      │
                      ├─→ Run TwoTowerFusion model
                      │
                      ├─→ Apply InjuryAdjuster (Bayesian)
                      │
                      ▼
              GamePrediction
                      │
                      ▼
          SignalGenerator.generate_signals()
                      │
                      ├─→ Devig market odds (Power Method)
                      │
                      ├─→ Calculate edge = model_prob - market_prob
                      │
                      ├─→ Filter by min_edge threshold
                      │
                      ├─→ Apply Kelly criterion (0.25x fractional)
                      │
                      ▼
              BettingSignal (if edge > min_edge)
```

## Key Classes

### InferencePipeline

```python
from nba_model.predict import InferencePipeline
from nba_model.models import ModelRegistry
from nba_model.data import session_scope

with session_scope() as session:
    pipeline = InferencePipeline(
        model_registry=ModelRegistry(),
        db_session=session,
        model_version="latest",
    )

    # Single game
    prediction = pipeline.predict_game("0022300123")

    # Today's games
    predictions = pipeline.predict_today()

    # Specific date
    predictions = pipeline.predict_date(date(2024, 1, 15))
```

### GamePrediction

```python
@dataclass
class GamePrediction:
    game_id: GameId
    game_date: date
    home_team: str
    away_team: str
    matchup: str

    # Raw model outputs
    home_win_prob: float        # 0.01-0.99
    predicted_margin: float     # -35 to +35
    predicted_total: float      # 175-270

    # Injury-adjusted outputs
    home_win_prob_adjusted: float
    predicted_margin_adjusted: float
    predicted_total_adjusted: float

    # Uncertainty quantification
    confidence: float           # 0-1
    injury_uncertainty: float   # 0-1

    # Explainability
    top_factors: list[tuple[str, float]]
```

### InjuryAdjuster

Bayesian updating for player availability:

```python
from nba_model.predict import InjuryAdjuster

adjuster = InjuryAdjuster(db_session)

# Get play probability for a player
prob = adjuster.get_play_probability(
    player_id=203507,
    injury_status="questionable",
    injury_type="ankle",
    team_context={"back_to_back": True},
)
# Returns: 0.47 (55% prior × 1.0 injury × 0.85 context)

# Prior probabilities
PRIOR_PLAY_PROBABILITIES = {
    "probable": 0.93,
    "questionable": 0.55,
    "doubtful": 0.03,
    "out": 0.00,
    "available": 1.00,
}
```

### SignalGenerator

```python
from nba_model.predict import SignalGenerator
from nba_model.backtest import DevigCalculator, KellyCalculator

generator = SignalGenerator(
    devig_calculator=DevigCalculator(),
    kelly_calculator=KellyCalculator(fraction=0.25),
    min_edge=0.02,  # 2% minimum edge
    bankroll=10000.0,
)

signals = generator.generate_signals(predictions, market_odds)

for signal in signals:
    print(f"{signal.matchup}: {signal.bet_type} {signal.side}")
    print(f"  Edge: {signal.edge:.1%}")
    print(f"  Stake: {signal.recommended_stake_pct:.2%}")
```

### BettingSignal

```python
@dataclass
class BettingSignal:
    game_id: GameId
    game_date: date
    matchup: str
    bet_type: str           # "moneyline", "spread", "total"
    side: str               # "home", "away", "over", "under"
    line: float | None      # Spread/total line
    model_prob: float       # Model probability
    market_prob: float      # Devigged market probability
    edge: float             # model_prob - market_prob
    recommended_odds: float # Decimal odds
    kelly_fraction: float   # Full Kelly fraction
    recommended_stake_pct: float  # After caps
    confidence: str         # "high", "medium", "low"
```

## CLI Commands

```bash
# Predict today's games
nba-model predict today

# Single game prediction
nba-model predict game 0022300123

# Specific date
nba-model predict date 2024-01-15

# Generate betting signals
nba-model predict signals --min-edge 0.02

# Use specific model version
nba-model predict today --model-version v1.2.0
```

## Latency Requirements

| Operation | Target | Max |
|-----------|--------|-----|
| Single game | 500ms | 5s |
| 15 games (full day) | 30s | 2 min |
| Signal generation | 100ms | 1s |

## Confidence Classification

| Level | Edge Threshold | Model Confidence |
|-------|---------------|------------------|
| High | ≥ 5% | ≥ 60% |
| Medium | ≥ 3% | ≥ 40% |
| Low | ≥ 2% | Any |

## Integration Points

**Upstream (inputs from):**
- `models/` - TwoTowerFusion, GameFlowTransformer, PlayerInteractionGNN
- `models/registry` - ModelRegistry for version management
- `features/` - ContextFeatureBuilder, RAPM, Spacing
- `data/` - Games, Teams, Players, Stints

**Downstream (outputs to):**
- `output/` - Dashboard display
- `monitor/` - Performance tracking

**Lateral (uses):**
- `backtest/devig` - DevigCalculator for vig removal
- `backtest/kelly` - KellyCalculator for bet sizing

## Anti-Patterns

- ❌ Never generate signals without checking injury reports
- ❌ Never bet on games within 2 hours (lines move significantly)
- ❌ Never use stale model (check model version date)
- ❌ Never ignore confidence scores in sizing
- ❌ Never use implied probability with vig for edge calculation
- ❌ Never bet without minimum edge threshold (2%)
- ❌ Never use full Kelly (use 0.25x fractional)
- ❌ Never skip devigging when comparing to market odds
