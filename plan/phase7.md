# Phase 7: Predictions

## Overview

This phase implements the production inference pipeline for generating game predictions, Bayesian injury probability adjustments, and actionable betting signal generation. The system must handle uncertainty from game-time decisions and late roster scratches while maintaining sub-5-second latency for single-game predictions.

## Dependencies

- Phase 3: Feature calculators (RAPM, spacing, fatigue, normalization)
- Phase 4: Trained model ensemble (Transformer, GNN, Two-Tower Fusion)
- Phase 5: DevigCalculator and KellyCalculator implementations

## Objectives

1. Implement production-grade inference pipeline capable of generating predictions for individual games and full daily slates
2. Build Bayesian injury adjustment system that accounts for player availability uncertainty using prior probabilities and contextual updates
3. Create betting signal generator that identifies positive expected value opportunities across moneyline, spread, and total markets
4. Provide prediction explainability through top contributing factors for each game

## Module Structure

Create the following modules under `nba_model/predict/`:

- `__init__.py` - Package exports
- `inference.py` - Core prediction pipeline
- `injuries.py` - Bayesian injury probability adjustments
- `signals.py` - Betting signal generation

## Task 7.1: Inference Pipeline

### Location
`nba_model/predict/inference.py`

### Components to Implement

#### InferencePipeline Class
Orchestrates prediction generation by:
- Loading models from ModelRegistry (default to 'latest' version)
- Building context features for Tower A input
- Constructing lineup graphs for GNN input
- Generating sequence representations for Transformer input (zeros for pre-game predictions)
- Running the fusion model
- Applying injury adjustments as post-processing

Required methods:
- `predict_game(game_id: str) -> GamePrediction` - Full prediction for single game
- `predict_today() -> list[GamePrediction]` - All games scheduled for current date
- `predict_date(date: date) -> list[GamePrediction]` - All games on specified date

#### GamePrediction Dataclass
Container storing:
- Game identifiers (game_id, game_date, home_team, away_team)
- Raw model outputs (home_win_prob, predicted_margin, predicted_total)
- Injury-adjusted outputs (home_win_prob_adjusted, predicted_margin_adjusted, predicted_total_adjusted)
- Uncertainty quantification (confidence score, injury_uncertainty score)
- Explainability data (top_factors as list of (feature_name, importance) tuples)
- Metadata (model_version, prediction_timestamp, expected lineups)

### Latency Requirements
- Single game prediction: < 5 seconds including feature construction
- Full day predictions (~15 games max): < 2 minutes

## Task 7.2: Bayesian Injury Adjustments

### Location
`nba_model/predict/injuries.py`

### Components to Implement

#### InjuryAdjuster Class
Implements Bayesian updating for player availability probabilities.

Prior probabilities (base rates from historical data):
- Probable: 93%
- Questionable: 55%
- Doubtful: 3%
- Out: 0%
- Available: 100%

Posterior update factors:
- Player historical patterns (some players rest more frequently, others play through injury)
- Team context (tanking scenarios, back-to-back games, playoff implications)
- Injury type classification

Required methods:
- `get_play_probability(player_id, injury_status, injury_type, team_context) -> float`
- `calculate_player_history_likelihood(player_id, injury_status) -> float`
- `adjust_prediction(base_prediction, injury_report) -> GamePrediction`
- `calculate_replacement_impact(player_id, replacement_id, team_id) -> float`

Adjustment algorithm:
1. For each player with uncertain status, compute play probability via Bayesian update
2. Run model for both scenarios (player plays / player sits)
3. Compute expected values: `E[win_prob] = P(plays) * P(win|plays) + P(sits) * P(win|sits)`
4. Calculate uncertainty score based on variance of possible outcomes

Replacement impact uses RAPM differential between expected starter and projected replacement.

#### InjuryReportFetcher Class
Retrieves current injury status data.

Required method:
- `get_current_injuries() -> DataFrame` with columns: player_id, team_id, status, injury_description, report_date

Data source: NBA API injury endpoints or web scraping fallback.

## Task 7.3: Betting Signal Generation

### Location
`nba_model/predict/signals.py`

### Components to Implement

#### SignalGenerator Class
Transforms predictions into actionable betting signals.

Initialization parameters:
- DevigCalculator instance (from Phase 5)
- KellyCalculator instance (from Phase 5)
- min_edge threshold (default 2%)

Required methods:
- `generate_signals(predictions, current_odds) -> list[BettingSignal]`
- `generate_game_signals(prediction, odds) -> list[BettingSignal]`

Signal generation logic for each game:
1. Obtain market odds for moneyline, spread, and total
2. Apply devigging to extract fair market probabilities
3. Compare model probability against fair market probability
4. Calculate edge as: `edge = model_prob - market_prob`
5. Filter to signals where edge exceeds min_edge threshold
6. Compute Kelly fraction for position sizing
7. Assign confidence level (high/medium/low) based on model confidence and edge magnitude

#### BettingSignal Dataclass
Container storing:
- Game identifiers (game_id, game_date, matchup string)
- Bet specification (bet_type, side, line if applicable)
- Probability comparison (model_prob, market_prob, edge)
- Sizing recommendations (recommended_odds, kelly_fraction, recommended_stake_pct)
- Confidence classification
- Context (key_factors list, injury_notes list)

### Edge Calculation Details
For each bet type:
- Moneyline: Compare home_win_prob against devigged market implied probability
- Spread: Convert predicted_margin to cover probability, compare against spread market
- Total: Convert predicted_total to over/under probabilities, compare against totals market

## CLI Integration

Add the following commands under `nba-model predict`:

- `predict today` - Generate and display predictions for today's games
- `predict game <game_id>` - Generate prediction for specific game
- `predict signals` - Generate betting signals with current market odds

Output format: Structured display using Rich library showing predictions, confidence levels, and recommended actions.

## Testing Requirements

### Unit Tests

#### InferencePipeline Tests
- Verify `predict_game` returns valid GamePrediction with all fields populated
- Verify probability outputs are in valid range [0, 1]
- Verify margin and total predictions are reasonable (margin: -30 to +30, total: 180 to 260)
- Test handling of missing lineup data (should use expected starters)
- Test model version loading (specific version and 'latest')

#### InjuryAdjuster Tests
- Verify prior probabilities match specification for each status level
- Test Bayesian update produces probabilities in valid range
- Verify adjustment direction is correct (missing star player should lower team win probability)
- Test replacement impact calculation using known RAPM values
- Verify uncertainty score increases with more questionable players

#### SignalGenerator Tests
- Verify no signals generated when edge < min_edge
- Verify signals generated when edge >= min_edge
- Test Kelly fraction calculation matches expected formula
- Verify confidence classification logic
- Test all bet types generate appropriate signal structures

### Integration Tests

#### End-to-End Prediction Flow
- Load trained models from registry
- Fetch game data for a historical date
- Generate predictions
- Verify predictions match expected format
- Confirm latency within specified bounds

#### Injury Integration
- Create mock injury report with known player statuses
- Verify adjusted predictions differ from base predictions
- Confirm adjustment magnitude correlates with player importance (RAPM)

#### Signal Generation Integration
- Provide mock odds data
- Generate signals for full day slate
- Verify signal filtering works correctly
- Confirm Kelly calculations are consistent with Phase 5 implementation

### Validation Criteria
- All probability outputs normalized to [0, 1]
- Adjusted predictions have higher uncertainty scores than base predictions when injuries present
- Signals only recommend bets with positive edge above threshold
- GamePrediction metadata correctly identifies model version used
- Prediction timestamps accurate to current system time

## Success Criteria

1. InferencePipeline successfully generates predictions using trained models
2. Single-game predictions complete within 5-second latency target
3. Full-day predictions complete within 2-minute latency target
4. InjuryAdjuster correctly applies Bayesian updates with specified priors
5. Injury adjustments appropriately modify win probabilities based on player RAPM
6. SignalGenerator identifies positive-edge betting opportunities
7. All CLI commands execute without errors
8. Test suite achieves >90% coverage of prediction module code

## Completion Instructions

Upon completion, commit all changes with message "feat: implement phase 7 - predictions" and push to origin.
