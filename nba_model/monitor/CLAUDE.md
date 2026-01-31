# Model Monitoring

## Responsibility

Detects model degradation through drift detection and triggers automated retraining. Manages model versioning and performance tracking.

## Status

**Phase 6 - Complete**

## Structure

| File | Purpose | Key Classes |
|------|---------|-------------|
| `__init__.py` | Public API | All exports |
| `drift.py` | Drift detection | `DriftDetector`, `ConceptDriftDetector` |
| `triggers.py` | Retrain logic | `RetrainingTrigger`, `TriggerContext`, `TriggerResult` |
| `versioning.py` | Model management | `ModelVersionManager`, `VersionMetadata` |

## Drift Detection Methods

| Type | Method | Threshold | Class |
|------|--------|-----------|-------|
| Covariate | KS Test | p < 0.05 | `DriftDetector` |
| Covariate | PSI | > 0.2 | `DriftDetector` |
| Concept | Accuracy degradation | < 48% | `ConceptDriftDetector` |
| Concept | Calibration drift | Brier > 0.35 | `ConceptDriftDetector` |

## Monitored Features

The following features are monitored for covariate drift:
- `pace` - Game pace (possessions per 48 minutes)
- `offensive_rating` - Points per 100 possessions
- `fg3a_rate` - Three-point attempt rate
- `rest_days` - Days since last game
- `travel_distance` - Miles traveled
- `rapm_mean` - Average RAPM of lineup

## Retraining Triggers

| Trigger | Condition | Priority |
|---------|-----------|----------|
| Scheduled | 7 days since last train | Medium |
| Drift | PSI > 0.2 for any feature | High |
| Performance | ROI < -5% over 50 bets | High |
| Data | 50+ new games since last train | Low |

## Usage Examples

### Drift Detection

```python
from nba_model.monitor import DriftDetector, MONITORED_FEATURES

# Initialize with training data
detector = DriftDetector(
    reference_data=training_df,
    p_value_threshold=0.05,
    psi_threshold=0.2,
)

# Check recent data for drift
result = detector.check_drift(recent_df)
if result.has_drift:
    print(f"Drifted features: {result.features_drifted}")
    for feature, detail in result.details.items():
        print(f"  {feature}: KS={detail.ks_stat:.3f}, PSI={detail.psi:.3f}")
```

### Concept Drift Detection

```python
from nba_model.monitor import ConceptDriftDetector

detector = ConceptDriftDetector(
    accuracy_threshold=0.48,
    calibration_threshold=0.10,
)

result = detector.check_prediction_drift(
    predictions=[0.55, 0.62, 0.48, ...],
    actuals=[1, 1, 0, ...],
    window_size=100,
)

if result.accuracy_degraded or result.calibration_degraded:
    print(f"Concept drift detected!")
    print(f"Recent accuracy: {result.recent_accuracy:.2%}")
    print(f"Recent Brier score: {result.recent_brier_score:.4f}")
```

### Retraining Triggers

```python
from datetime import date, timedelta
from nba_model.monitor import RetrainingTrigger, TriggerContext

trigger = RetrainingTrigger(
    scheduled_interval_days=7,
    min_new_games=50,
    roi_threshold=-0.05,
    accuracy_threshold=0.48,
)

context = TriggerContext(
    last_train_date=date.today() - timedelta(days=10),
    drift_detector=detector,
    recent_data=recent_df,
    recent_bets=bets,
    games_since_training=75,
)

result = trigger.evaluate_all_triggers(context)
if result.should_retrain:
    print(f"Retraining recommended: {result.reason}")
    print(f"Priority: {result.priority}")
```

### Model Version Management

```python
from nba_model.monitor import ModelVersionManager

manager = ModelVersionManager()

# Create new version
version = manager.create_version(
    models={"transformer": t, "gnn": g, "fusion": f},
    config={"d_model": 128},
    metrics={"accuracy": 0.58, "brier_score": 0.22},
    parent_version="1.0.0",
    bump="minor",
)

# Compare versions
comparison = manager.compare_versions("1.0.0", "1.1.0")
print(f"Winner: {comparison.winner}")

# Promote version
manager.promote_version("1.1.0")

# Rollback if needed
manager.rollback("1.0.0")

# Get version lineage
lineage = manager.get_lineage("1.2.0")  # ["1.0.0", "1.1.0", "1.2.0"]
```

## CLI Commands

```bash
# Check for covariate drift
nba-model monitor drift --window 30

# Evaluate retraining triggers
nba-model monitor trigger

# List model versions
nba-model monitor versions

# Compare specific versions
nba-model monitor versions --compare v1.0.0,v1.1.0
```

## Integration Points

- **Upstream:** `backtest/` provides performance metrics and bet history
- **Downstream:** `models/` for retraining, `output/` for alerts

## Anti-Patterns

- Never retrain on concept drift without root cause analysis
- Never auto-deploy retrained models without validation
- Never ignore covariate drift (early warning of concept drift)
- Never use single metric for retraining decisions
- Never skip version comparison before promotion
