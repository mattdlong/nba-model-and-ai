# Phase 6: Self-Improvement Iterations

## Overview

This phase implements the continuous learning and model monitoring infrastructure. The system must autonomously detect when model performance degrades, identify distribution shifts in input features, trigger retraining pipelines, and manage model versioning with rollback capabilities.

## Objectives

1. Implement covariate drift detection using statistical tests (KS, PSI)
2. Implement concept drift detection via prediction-outcome divergence monitoring
3. Create automatic retraining triggers with configurable thresholds
4. Build model versioning system with A/B comparison and promotion workflow
5. Enable continuous learning pipeline with scheduled and event-driven retraining

## Dependencies

- Phase 4: Trained models, model registry infrastructure, training pipeline
- Phase 5: Backtest metrics calculation, Bet dataclass, performance evaluation

## Module Structure

All components reside in `nba_model/monitor/`:
- `drift.py` - Covariate and concept drift detection
- `triggers.py` - Retraining trigger evaluation logic
- `versioning.py` - Model version management and comparison

## Task 6.1: Covariate Drift Detection

### DriftDetector Class

Implement a `DriftDetector` class that monitors input feature distributions for significant shifts from the reference (training) distribution.

**Initialization Parameters:**
- `reference_data: pd.DataFrame` - Feature values from training period
- `p_value_threshold: float = 0.05` - KS test significance threshold
- `psi_threshold: float = 0.2` - PSI threshold for significant drift

**Monitored Features (constant list):**
- `pace`, `offensive_rating`, `fg3a_rate`, `rest_days`, `travel_distance`, `rapm_mean`

**Methods to Implement:**

1. `ks_test(feature: str, recent_data: pd.DataFrame) -> tuple[float, float]`
   - Execute two-sample Kolmogorov-Smirnov test between reference and recent distributions
   - Return (test_statistic, p_value)
   - Use `scipy.stats.ks_2samp`

2. `calculate_psi(feature: str, recent_data: pd.DataFrame, n_bins: int = 10) -> float`
   - Compute Population Stability Index
   - PSI formula: `sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))`
   - Bin reference and recent data into `n_bins` quantile buckets
   - Handle edge cases where bin counts are zero (add small epsilon)
   - Interpretation thresholds: <0.1 stable, 0.1-0.2 moderate, >=0.2 significant

3. `check_drift(recent_data: pd.DataFrame, window_days: int = 30) -> dict`
   - Execute both KS and PSI tests for all monitored features
   - Return structure:
     ```
     {
       'has_drift': bool,
       'features_drifted': list[str],
       'details': {feature_name: {'ks_stat': float, 'p_value': float, 'psi': float}}
     }
     ```
   - Feature is flagged as drifted if KS p-value < threshold OR PSI > threshold

### ConceptDriftDetector Class

Implement a `ConceptDriftDetector` class that monitors the relationship between predictions and outcomes.

**Initialization Parameters:**
- `accuracy_threshold: float = 0.48` - Minimum acceptable accuracy
- `calibration_threshold: float = 0.1` - Maximum acceptable calibration error

**Methods to Implement:**

1. `check_prediction_drift(predictions: list[float], actuals: list[int], window_size: int = 100) -> dict`
   - Compute rolling accuracy over most recent `window_size` predictions
   - Compute Brier score for probability calibration
   - Return structure:
     ```
     {
       'accuracy_degraded': bool,
       'calibration_degraded': bool,
       'recent_accuracy': float,
       'recent_brier_score': float
     }
     ```
   - Accuracy degraded if below threshold
   - Calibration degraded if Brier score exceeds `0.25 + calibration_threshold`

## Task 6.2: Retraining Triggers

### RetrainingTrigger Class

Implement a `RetrainingTrigger` class that evaluates multiple conditions to determine if model retraining is warranted.

**Initialization Parameters:**
- `scheduled_interval_days: int = 7` - Automatic retrain interval
- `min_new_games: int = 50` - Minimum games to justify data-based retrain
- `roi_threshold: float = -0.05` - ROI below this triggers retrain
- `accuracy_threshold: float = 0.48` - Accuracy below this triggers retrain

**Trigger Types:**

1. **Scheduled Trigger** - Time-based periodic retraining
   - Method: `check_scheduled_trigger(last_train_date: date) -> bool`
   - Returns True if current_date - last_train_date >= scheduled_interval_days

2. **Drift Trigger** - Statistical distribution shift detected
   - Method: `check_drift_trigger(drift_detector: DriftDetector, recent_data: pd.DataFrame) -> bool`
   - Returns True if drift_detector.check_drift() indicates drift

3. **Performance Trigger** - Model predictions degrading
   - Method: `check_performance_trigger(recent_bets: list[Bet]) -> bool`
   - Calculate ROI and accuracy from recent_bets (minimum 50 bets)
   - Returns True if ROI < roi_threshold OR win_rate < accuracy_threshold

4. **Data Trigger** - Sufficient new training data available
   - Method: `check_data_trigger(games_since_training: int) -> bool`
   - Returns True if games_since_training >= min_new_games

**Aggregation Method:**

`evaluate_all_triggers(context: dict) -> dict`

Context dict contains: `last_train_date`, `drift_detector`, `recent_data`, `recent_bets`, `games_since_training`

Return structure:
```
{
  'should_retrain': bool,
  'reason': str,  # e.g., 'drift_detected', 'scheduled', 'performance_degraded', 'new_data'
  'priority': str,  # 'high' (performance/drift), 'medium' (scheduled), 'low' (data)
  'trigger_details': {trigger_name: bool for each trigger type}
}
```

Priority assignment:
- High: drift or performance triggers activated
- Medium: scheduled trigger activated
- Low: only data trigger activated

## Task 6.3: Model Versioning and Comparison

### ModelVersionManager Class

Implement a `ModelVersionManager` class that handles model lifecycle management.

**Storage Structure:**
```
data/models/
  v1.0.0/
    transformer.pt
    gnn.pt
    fusion.pt
    metadata.json
  v1.0.1/
    ...
  latest -> v1.0.1 (symlink or metadata pointer)
```

**Version Format:** `v{major}.{minor}.{patch}`
- Major: Architecture changes (new model type, layer modifications)
- Minor: Retraining with new data (same architecture)
- Patch: Hyperparameter tuning only

**Metadata Schema:**
```
{
  'version': str,
  'created_at': ISO datetime,
  'training_data_start': date,
  'training_data_end': date,
  'hyperparameters': dict,
  'validation_metrics': dict,
  'git_commit': str (optional),
  'parent_version': str (optional),
  'status': str  # 'active', 'deprecated', 'promoted'
}
```

**Methods to Implement:**

1. `create_version(models: dict, config: dict, metrics: dict, parent_version: str = None) -> str`
   - Determine next version number based on parent and change type
   - Save model weights to versioned directory
   - Write metadata.json
   - Return version string

2. `compare_versions(version_a: str, version_b: str, test_data: pd.DataFrame) -> dict`
   - Load both model versions
   - Run inference on identical test_data
   - Calculate metrics for both: accuracy, Brier score, margin MAE, total MAE
   - Return structure:
     ```
     {
       'version_a_metrics': dict,
       'version_b_metrics': dict,
       'winner': str,  # version with better overall performance
       'improvement': {metric_name: (version_b_value - version_a_value)}
     }
     ```
   - Winner determined by: accuracy > Brier score (inverse) > margin MAE (inverse)

3. `promote_version(version: str) -> None`
   - Set version as 'latest' (production model)
   - Update version metadata status to 'promoted'
   - Update previous promoted version status to 'deprecated'

4. `rollback(to_version: str) -> None`
   - Validate version exists
   - Set specified version as 'latest'
   - Log rollback event in metadata

5. `get_lineage(version: str) -> list[str]`
   - Traverse parent_version links
   - Return ordered list from oldest ancestor to specified version

6. `list_versions() -> list[dict]`
   - Return metadata for all versions, sorted by creation date descending

## CLI Integration

Add the following commands to the CLI under the `monitor` group:

1. `nba-model monitor drift`
   - Instantiate DriftDetector with reference data from last training period
   - Run check_drift on data from past 30 days
   - Output drift status and details to console
   - Exit code 1 if drift detected

2. `nba-model monitor trigger`
   - Load all required context (last train date, recent bets, recent data)
   - Run evaluate_all_triggers
   - Output trigger evaluation results
   - If should_retrain is True, output recommended action

3. `nba-model monitor versions`
   - List all model versions with key metrics
   - Indicate which version is currently promoted
   - Support `--compare v1.0.0 v1.0.1` flag for version comparison

## Technical Specifications

| Component | Check Frequency | Thresholds |
|-----------|-----------------|------------|
| Covariate Drift | Daily | KS p < 0.05, PSI > 0.2 |
| Concept Drift | Per 100 predictions | Accuracy < 48%, Brier > 0.35 |
| Scheduled Retrain | Weekly | 7 days since last train |
| Performance Trigger | Daily | ROI < -5% over 50 bets |
| Data Trigger | Daily | 50 new games since last train |

## Testing Requirements

### Unit Tests

1. **DriftDetector Tests** (`tests/test_monitor/test_drift.py`)
   - Test KS test returns valid statistics for identical distributions (high p-value)
   - Test KS test detects synthetic shifted distribution (low p-value)
   - Test PSI calculation returns 0 for identical distributions
   - Test PSI calculation returns >0.2 for heavily shifted distribution
   - Test check_drift aggregates results correctly
   - Test edge case handling for empty dataframes

2. **ConceptDriftDetector Tests**
   - Test accuracy calculation with known predictions/actuals
   - Test Brier score calculation matches expected formula
   - Test degradation flags set correctly at threshold boundaries

3. **RetrainingTrigger Tests** (`tests/test_monitor/test_triggers.py`)
   - Test scheduled trigger activates at exactly N days
   - Test drift trigger passes through drift detector result
   - Test performance trigger activates when ROI below threshold
   - Test data trigger activates at minimum game count
   - Test evaluate_all_triggers priority assignment logic
   - Test trigger combination scenarios (multiple triggers active)

4. **ModelVersionManager Tests** (`tests/test_monitor/test_versioning.py`)
   - Test version string generation follows semantic versioning
   - Test metadata is correctly written and read
   - Test compare_versions loads models and computes metrics
   - Test promote_version updates symlink/pointer and metadata
   - Test rollback restores previous version state
   - Test get_lineage traverses parent chain correctly
   - Test list_versions returns sorted results

### Integration Tests

1. **Drift Detection Pipeline**
   - Load actual training data sample
   - Inject synthetic drift into test features
   - Verify drift detection identifies correct features
   - Verify no false positives on stable features

2. **Retrain Trigger Pipeline**
   - Create mock context with various trigger conditions
   - Verify correct priority assignment
   - Verify reason string accurately describes triggering condition

3. **Version Lifecycle**
   - Create initial version v1.0.0
   - Create derived version v1.1.0
   - Compare versions on test data
   - Promote better version
   - Rollback and verify state restoration
   - Verify lineage shows correct ancestry

### Validation Criteria

- All drift detection tests pass with scipy 1.11+
- Trigger evaluation completes in <1 second
- Version comparison completes in <30 seconds (including model loading)
- No memory leaks in repeated drift checks (test 1000 iterations)
- Metadata JSON is valid and parseable after all operations

## Success Criteria

1. DriftDetector correctly identifies 90%+ of synthetically induced distribution shifts
2. RetrainingTrigger evaluates all conditions without false positives on stable periods
3. ModelVersionManager maintains data integrity across create/promote/rollback cycles
4. CLI commands execute successfully and return appropriate exit codes
5. All unit and integration tests pass

---

Upon completion, commit all changes with message "feat: implement phase 6 - self-improvement" and push to origin
