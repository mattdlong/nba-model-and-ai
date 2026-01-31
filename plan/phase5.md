# Phase 5: Backtesting

## Overview

This phase implements the backtesting infrastructure for validating model performance and bet sizing strategies. The system must prevent look-ahead bias through walk-forward validation, convert market odds to fair probabilities via devigging, apply Kelly Criterion for optimal bet sizing, and generate comprehensive performance metrics.

## Dependencies

- Phase 4: Trained fusion models (Transformer, GNN, Two-Tower architecture)
- Phase 2: Historical odds data in `odds` table (if available)
- Database tables: `games`, `odds`, model weights in `data/models/`

## Objectives

1. Implement walk-forward validation engine respecting temporal ordering
2. Build devigging calculators (Multiplicative, Power Method, Shin's Method)
3. Implement Kelly Criterion bet sizing with fractional Kelly and caps
4. Create comprehensive backtest metrics and result containers

## Module Structure

```
nba_model/backtest/
    __init__.py
    engine.py      # Walk-forward validation
    kelly.py       # Kelly criterion sizing
    devig.py       # Vig removal methods
    metrics.py     # Performance metrics
```

## Task 5.1: Walk-Forward Validation Engine

**Location:** `nba_model/backtest/engine.py`

### WalkForwardEngine Class

Implement temporal cross-validation that prevents future information leakage. Unlike k-fold CV, walk-forward validation maintains strict chronological ordering.

**Fold Generation Logic:**
- Fold 1: Train on [Season 1-2], Validate on [Season 3 first half]
- Fold 2: Train on [Season 1-3 first half], Validate on [Season 3 second half]
- Fold 3: Train on [Season 1-3], Validate on [Season 4 first half]
- Continue expanding training window while sliding validation window forward

**Configuration Parameters:**
- `min_train_games`: Minimum games in training set (default: 500)
- `validation_window_games`: Games per validation fold (default: 100)
- `step_size_games`: How far to advance between folds (default: 50)

**Required Methods:**
- `generate_folds(games_df)`: Return list of (train_df, val_df) tuples sorted by game_date
- `run_backtest(trainer, kelly_calculator, devig_method, initial_bankroll)`: Execute full walk-forward backtest

**Backtest Execution Flow:**
1. Generate chronological folds
2. For each fold:
   - Train model on training set
   - Generate predictions on validation set
   - Fetch corresponding market odds
   - Apply devigging to get fair probabilities
   - Calculate edge (model_prob - market_prob)
   - Apply Kelly sizing for bets with positive edge
   - Simulate bet outcomes
   - Track profit/loss and bankroll
3. Aggregate results across all folds
4. Return BacktestResult container

### BacktestResult Class

Container for aggregated backtest outcomes.

**Required Properties:**
- `bets`: List of all Bet objects
- `bankroll_history`: List of bankroll values over time
- `metrics`: Dictionary of computed metrics
- `total_return`: Final bankroll / initial bankroll - 1
- `sharpe_ratio`: Risk-adjusted return calculation
- `max_drawdown`: Largest peak-to-trough decline
- `win_rate`: Winning bets / total bets
- `clv`: Average closing line value across bets

### Bet Dataclass

**Required Fields:**
- `game_id`: Game identifier
- `timestamp`: When bet was placed
- `bet_type`: 'moneyline' | 'spread' | 'total'
- `side`: 'home' | 'away' | 'over' | 'under'
- `model_prob`: Model's probability estimate
- `market_odds`: Decimal odds from market
- `market_prob`: Fair probability after devigging
- `edge`: model_prob - market_prob
- `kelly_fraction`: Calculated Kelly fraction
- `bet_amount`: Actual bet size in currency units
- `result`: 'win' | 'loss' | 'push'
- `profit`: Net profit/loss from bet

## Task 5.2: Devigging Methods

**Location:** `nba_model/backtest/devig.py`

### DevigCalculator Class

Remove bookmaker vig (margin) to extract true implied probabilities.

**Method 1: Multiplicative Devig**
- Simple approach: `p_fair = implied_p / sum(implied_p)`
- Fast but inferior accuracy
- Use as baseline comparison

**Method 2: Power Method**
- Solve for exponent k such that: `(1/O_home)^k + (1/O_away)^k = 1`
- Fair probability = `(1/O)^k`
- Better handling of longshot bias than multiplicative
- Implement via binary search to find k within tolerance (default: 1e-6)

**Method 3: Shin's Method (Gold Standard)**
- Models market containing informed bettors
- Iteratively solve for z (proportion of informed bettors)
- Derive true probabilities from z
- Most accurate for liquid markets
- Required for production use

**Required Methods:**
- `multiplicative_devig(odds_home, odds_away)`: Return (fair_prob_home, fair_prob_away)
- `power_method_devig(odds_home, odds_away)`: Return (fair_prob_home, fair_prob_away)
- `shin_method_devig(odds_home, odds_away)`: Return (fair_prob_home, fair_prob_away)
- `calculate_edge(model_prob, market_prob)`: Return model_prob - market_prob

### Helper Function

`solve_power_k(odds, tol)`: Binary search implementation to find k for power method devigging.

## Task 5.3: Kelly Criterion Sizing

**Location:** `nba_model/backtest/kelly.py`

### KellyCalculator Class

Implement Kelly Criterion bet sizing with variance reduction via fractional Kelly.

**Full Kelly Formula:**
```
f* = (bp - q) / b
where:
    b = decimal_odds - 1 (net odds)
    p = model probability of winning
    q = 1 - p
```

**Configuration Parameters:**
- `fraction`: Kelly fraction multiplier (default: 0.25 for quarter Kelly)
- `max_bet_pct`: Maximum bet as percentage of bankroll (default: 0.02 = 2%)
- `min_edge_pct`: Minimum edge required to place bet (default: 0.02 = 2%)

**Required Methods:**
- `calculate_bet_size(bankroll, model_prob, decimal_odds)`: Return bet amount in currency units (0 if no edge or negative Kelly)
- `calculate_full_kelly(model_prob, decimal_odds)`: Return full Kelly fraction (can be negative)
- `optimize_fraction(historical_bets, fractions)`: Find optimal Kelly fraction by simulating different fractions on historical data, maximizing Sharpe ratio or geometric growth rate

**Bet Size Calculation Flow:**
1. Calculate full Kelly fraction
2. If negative or below min_edge_pct, return 0
3. Apply fractional Kelly multiplier
4. Cap at max_bet_pct of bankroll
5. Return final bet amount

## Task 5.4: Performance Metrics

**Location:** `nba_model/backtest/metrics.py`

### BacktestMetrics Class

Calculate comprehensive performance metrics from backtest results.

**Required Method:** `calculate_all(result: BacktestResult) -> dict`

**Return Metrics:**

*Returns:*
- `total_return`: Total percentage return over backtest period
- `cagr`: Compound annual growth rate
- `avg_bet_return`: Average return per bet

*Risk:*
- `volatility`: Standard deviation of returns
- `sharpe_ratio`: Risk-adjusted return (excess return / volatility)
- `sortino_ratio`: Downside risk-adjusted return
- `max_drawdown`: Largest peak-to-trough percentage decline
- `max_drawdown_duration`: Days spent in maximum drawdown

*Betting:*
- `total_bets`: Count of all bets placed
- `win_rate`: Percentage of winning bets
- `avg_edge`: Average edge across all bets
- `avg_odds`: Average decimal odds of bets placed
- `roi`: Return on investment (profit / total wagered)

*Calibration:*
- `brier_score`: Mean squared error of probability predictions
- `log_loss`: Cross-entropy loss of predictions

*CLV (Closing Line Value):*
- `avg_clv`: Average closing line value (key profitability indicator)
- `clv_positive_rate`: Percentage of bets with positive CLV

*Segmentation:*
- `metrics_by_type`: Same metrics computed separately for moneyline, spread, and total bets

**Additional Methods:**
- `calculate_clv(bet, closing_odds)`: Compute CLV for single bet as `(closing_implied_prob - bet_implied_prob) / bet_implied_prob`
- `generate_report(result)`: Produce human-readable text report of backtest results

## CLI Commands

Register the following commands under the `backtest` subcommand group:

- `nba-model backtest run`: Execute walk-forward backtest with configurable parameters
- `nba-model backtest report`: Generate backtest report from stored results
- `nba-model backtest optimize`: Run Kelly fraction optimization on historical bets

## Performance Targets

| Metric | Target | Acceptable |
|--------|--------|------------|
| ROI | > 5% | > 2% |
| Win Rate | > 53% | > 51% |
| CLV | > 1% | > 0% |
| Max Drawdown | < 15% | < 25% |
| Sharpe Ratio | > 1.0 | > 0.5 |

## Success Criteria

1. Walk-forward engine generates temporally valid folds with no data leakage
2. All three devigging methods produce probabilities summing to 1.0
3. Kelly calculator respects fraction, cap, and minimum edge constraints
4. Backtest can run end-to-end with trained models and produce BacktestResult
5. All metrics calculate correctly from bet history
6. CLI commands execute without errors

## Testing Requirements

### Unit Tests

**engine.py:**
- Verify fold generation maintains chronological ordering (no future games in training set)
- Verify train/validation sets are disjoint
- Verify minimum training size constraint is respected
- Test backtest execution with mock trainer and calculator

**devig.py:**
- Verify multiplicative method: given odds 1.91/1.91, fair probs should each be ~0.5
- Verify power method converges within tolerance
- Verify Shin's method produces valid probabilities for various odds combinations
- Test edge calculation returns correct sign and magnitude

**kelly.py:**
- Verify full Kelly calculation matches formula
- Verify fractional Kelly applies multiplier correctly
- Verify bet size capped at max_bet_pct
- Verify zero return when edge below threshold
- Verify zero return when Kelly fraction negative

**metrics.py:**
- Verify Sharpe ratio calculation against known values
- Verify max drawdown correctly identifies peak-to-trough
- Verify win rate, ROI, and other basic metrics
- Verify CLV calculation with known closing odds

### Integration Tests

- Run complete backtest on sample of historical data (minimum 200 games)
- Verify bankroll history is monotonically tracked
- Verify bet count matches expected signals
- Verify no NaN or infinite values in metrics
- Confirm metrics fall within plausible ranges

### Validation Tests

- Compare devigging methods: Power and Shin should produce similar results for balanced odds
- Verify CLV positive rate correlates with profitability
- Check that higher Kelly fractions increase both return and volatility
- Validate Brier score indicates calibrated probabilities (< 0.25 for reasonable models)

---

Upon completion, commit all changes with message "feat: implement phase 5 - backtesting" and push to origin
