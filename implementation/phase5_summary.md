# Phase 5: Backtesting Engine - Implementation Summary

## Overview

Phase 5 implements the backtesting infrastructure for validating model performance and bet sizing strategies. The system prevents look-ahead bias through walk-forward validation, converts market odds to fair probabilities via devigging, applies Kelly Criterion for optimal bet sizing, and generates comprehensive performance metrics.

## Completed Components

### 1. Walk-Forward Validation Engine (`engine.py`)

**WalkForwardEngine Class**
- Temporal cross-validation that maintains strict chronological ordering
- Configurable parameters:
  - `min_train_games`: Minimum games in training set (default: 500)
  - `validation_window_games`: Games per validation fold (default: 100)
  - `step_size_games`: Advancement between folds (default: 50)

**Key Methods:**
- `generate_folds(games_df)`: Returns list of (train_df, val_df, fold_info) tuples
- `run_backtest(trainer, config)`: Execute full walk-forward backtest

**BacktestResult Container:**
- `bets`: List of all Bet objects
- `bankroll_history`: Bankroll values over time
- `fold_results`: Per-fold results
- `metrics`: Computed performance metrics

**BacktestConfig Dataclass:**
- All configuration in one frozen dataclass
- Supports moneyline, spread, and total bet types

### 2. Devigging Methods (`devig.py`)

**Three devigging methods implemented:**

1. **Multiplicative** (simple baseline)
   - `p_fair = implied_p / sum(implied_p)`
   - Fast but doesn't handle longshot bias

2. **Power Method** (recommended for most cases)
   - Solves `(1/O_home)^k + (1/O_away)^k = 1` via Brent's method
   - Better handling of longshot bias

3. **Shin's Method** (theoretical gold standard)
   - Models market with informed bettors
   - Uses normalized weighting formula

**Helper Functions:**
- `implied_probability(odds)`: Convert odds to probability
- `calculate_overround(odds_home, odds_away)`: Calculate vig
- `solve_power_k(odds)`: Find power method exponent

### 3. Kelly Criterion Sizing (`kelly.py`)

**KellyCalculator Class:**
- Full Kelly formula: `f* = (bp - q) / b`
- Fractional Kelly multiplier (default: 0.25 = quarter Kelly)
- Maximum bet cap (default: 2% of bankroll)
- Minimum edge requirement (default: 2%)

**Key Methods:**
- `calculate_full_kelly(model_prob, odds)`: Raw Kelly fraction
- `calculate_bet_size(bankroll, prob, odds)`: Final bet amount
- `calculate(...)`: Full KellyResult with all details
- `simulate_bankroll(bets)`: Simulate historical performance
- `optimize_fraction(bets, fractions)`: Find optimal Kelly fraction

**KellyResult Container:**
- `full_kelly`: Uncapped Kelly fraction
- `adjusted_kelly`: After fractional multiplier
- `bet_fraction`: Final capped fraction
- `bet_amount`: Currency amount
- `edge`: Model edge over market
- `has_edge`: Whether to place bet

### 4. Performance Metrics (`metrics.py`)

**BacktestMetricsCalculator Class:**

**Returns Metrics:**
- `total_return`: Total percentage return
- `cagr`: Compound annual growth rate
- `avg_bet_return`: Average return per bet

**Risk Metrics:**
- `volatility`: Standard deviation of returns
- `sharpe_ratio`: Risk-adjusted return
- `sortino_ratio`: Downside risk-adjusted return
- `max_drawdown`: Largest peak-to-trough decline
- `max_drawdown_duration`: Days in drawdown

**Betting Metrics:**
- `total_bets`, `win_rate`, `avg_edge`, `avg_odds`, `roi`

**Calibration Metrics:**
- `brier_score`: Probability calibration
- `log_loss`: Cross-entropy loss

**CLV Metrics:**
- `avg_clv`: Average closing line value
- `clv_positive_rate`: Percentage with positive CLV

**Additional Features:**
- `metrics_by_type`: Segmented by bet type
- `generate_report(metrics)`: Human-readable text report
- `calculate_calibration_curve()`: Calibration curve data

### 5. CLI Commands

**`nba-model backtest run`**
- Execute walk-forward backtest
- Options: `--kelly`, `--max-bet`, `--min-edge`, `--devig`, `--bankroll`, `--season`
- Shows fold progress and final metrics table

**`nba-model backtest report`**
- Generate detailed performance report
- Can load from file or show example

**`nba-model backtest optimize`**
- Find optimal Kelly fraction
- Options: `--fractions`, `--metric` (sharpe/growth)
- Displays results table with best fraction

## Test Coverage

95 unit tests covering:
- All devigging methods and convergence
- Kelly calculations and constraints
- Metrics calculations (Sharpe, drawdown, CLV, etc.)
- Walk-forward fold generation and temporal ordering
- Full backtest pipeline integration

## Files Created/Modified

### New Files:
- `nba_model/backtest/engine.py` (~500 lines)
- `nba_model/backtest/kelly.py` (~300 lines)
- `nba_model/backtest/devig.py` (~400 lines)
- `nba_model/backtest/metrics.py` (~350 lines)
- `tests/unit/backtest/test_engine.py`
- `tests/unit/backtest/test_kelly.py`
- `tests/unit/backtest/test_devig.py`
- `tests/unit/backtest/test_metrics.py`

### Modified Files:
- `nba_model/backtest/__init__.py` - Updated public API
- `nba_model/cli.py` - Implemented backtest commands
- `nba_model/backtest/CLAUDE.md` - Updated documentation
- `CLAUDE.md` - Updated Phase 5 status

## Usage Examples

### Basic Backtest
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

### Kelly Calculation
```python
from nba_model.backtest import KellyCalculator

calc = KellyCalculator(fraction=0.25, max_bet_pct=0.02, min_edge_pct=0.02)
result = calc.calculate(bankroll=10000, model_prob=0.55, decimal_odds=1.91)
print(f"Bet: ${result.bet_amount:.2f} ({result.edge:.1%} edge)")
```

### Devigging
```python
from nba_model.backtest import DevigCalculator

devig = DevigCalculator()
fair_probs = devig.power_method_devig(1.91, 1.91)
print(f"Home: {fair_probs.home:.3f}, Away: {fair_probs.away:.3f}")
```

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| ROI | > 5% | > 2% |
| Win Rate | > 53% | > 51% |
| CLV | > 1% | > 0% |
| Max Drawdown | < 15% | < 25% |
| Sharpe Ratio | > 1.0 | > 0.5 |

## Success Criteria Met

1. ✅ Walk-forward engine generates temporally valid folds with no data leakage
2. ✅ All three devigging methods produce probabilities summing to 1.0
3. ✅ Kelly calculator respects fraction, cap, and minimum edge constraints
4. ✅ Backtest can run end-to-end with trained models and produce BacktestResult
5. ✅ All metrics calculate correctly from bet history
6. ✅ CLI commands execute without errors

## Integration Points

- **Upstream**: `nba_model.models` provides trained models via `TrainerProtocol`
- **Downstream**: `nba_model.monitor` will use backtest results for drift detection

## Next Steps (Phase 6)

1. Implement drift detection module
2. Add retraining trigger logic
3. Integrate backtest metrics into monitoring
