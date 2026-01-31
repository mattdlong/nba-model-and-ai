# Backtesting Engine

## Responsibility

Validates model performance through historical simulation. Implements walk-forward validation, Kelly criterion sizing, and vig removal.

## Status

✅ **Phase 5 - Complete**

## Module Structure

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Public API | Exports all public components |
| `engine.py` | Walk-forward validation | `WalkForwardEngine`, `BacktestResult`, `BacktestConfig` |
| `kelly.py` | Bet sizing | `KellyCalculator`, `KellyResult`, `simulate_bankroll` |
| `devig.py` | Vig removal | `DevigCalculator`, `FairProbabilities` |
| `metrics.py` | Performance metrics | `BacktestMetricsCalculator`, `FullBacktestMetrics` |

## Key Algorithms

### Walk-Forward Validation
Strict temporal ordering to prevent look-ahead bias:
```
Fold 1: Train [Game 1-500], Validate [Game 501-600]
Fold 2: Train [Game 1-550], Validate [Game 551-650]
...
```

### Kelly Criterion
Full Kelly: `f* = (bp - q) / b` where:
- b = decimal_odds - 1 (net odds)
- p = model probability
- q = 1 - p

Default: 0.25x Kelly (quarter Kelly) for variance reduction.

### Devigging Methods
1. **Multiplicative**: `p_fair = p_implied / sum(p_implied)`
2. **Power Method**: Solve `Σ(1/odds)^k = 1` via Brent's method
3. **Shin's Method**: Model informed bettor proportion (gold standard)

## Usage Example

```python
from nba_model.backtest import WalkForwardEngine, BacktestConfig, create_mock_trainer

config = BacktestConfig(
    min_train_games=500,
    validation_window_games=100,
    kelly_fraction=0.25,
    max_bet_pct=0.02,
    min_edge_pct=0.02,
    devig_method="power",
)

engine = WalkForwardEngine(
    min_train_games=config.min_train_games,
    validation_window_games=config.validation_window_games,
)

result = engine.run_backtest(games_df, trainer, config=config)
print(f"ROI: {result.metrics.roi:.2%}")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| ROI | > 5% | > 2% |
| Win Rate | > 53% | > 51% |
| CLV | > 1% | > 0% |
| Max Drawdown | < 15% | < 25% |
| Sharpe Ratio | > 1.0 | > 0.5 |

## Integration Points

- **Upstream:** `models/` provides trained models via `TrainerProtocol`
- **Downstream:** `monitor/` uses backtest results for drift detection

## CLI Commands

```bash
# Run walk-forward backtest
nba-model backtest run --kelly 0.25 --min-edge 0.02

# Generate report from results
nba-model backtest report --file results.json

# Optimize Kelly fraction
nba-model backtest optimize --metric sharpe
```

## Anti-Patterns

- ❌ Avoid full Kelly - use fractional Kelly (recommended max 0.5x, code allows up to 1.0)
- ❌ Never backtest without accounting for vig
- ❌ Never look ahead in walk-forward validation
- ❌ Never bet without minimum edge requirement
- ❌ Never skip devigging when comparing to market odds
- ❌ Never use implied probability (with vig) for edge calculation - use devigged market_prob
