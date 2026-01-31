# Backtesting Engine

## Responsibility

Validates model performance through historical simulation. Implements walk-forward validation, Kelly criterion sizing, and vig removal.

## Status

🔲 **Phase 5 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Key Functions |
|------|---------|---------------|
| `__init__.py` | Public API | - |
| `engine.py` | Walk-forward validation | `BacktestEngine.run()` |
| `kelly.py` | Bet sizing | `calculate_kelly_fraction()` |
| `devig.py` | Vig removal | `power_devig()`, `shin_devig()` |
| `metrics.py` | Performance metrics | ROI, Sharpe, CLV, drawdown |

## Key Algorithms

1. **Walk-Forward:** Train on N seasons, validate on next, roll forward
2. **Kelly Criterion:** `f* = (bp - q) / b` with fractional adjustment (0.25x)
3. **Devigging:** Power method `Σ(1/odds)^k = 1` or Shin method

## Performance Targets

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| ROI | > 3% | > 1% |
| Sharpe | > 1.0 | > 0.5 |
| Max Drawdown | < 15% | < 25% |
| CLV | Positive | Any positive |
| Brier Score | < 0.24 | < 0.25 |

## Integration Points

- **Upstream:** `models/` provides trained models
- **Downstream:** `monitor/` uses backtest results for drift detection

## Anti-Patterns

- ❌ Never use full Kelly (always fractional, max 0.5x)
- ❌ Never backtest without transaction costs (vig)
- ❌ Never look ahead in walk-forward validation
- ❌ Never report metrics without confidence intervals
