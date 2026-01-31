# Model Monitoring

## Responsibility

Detects model degradation through drift detection and triggers automated retraining. Manages model versioning and performance tracking.

## Status

🔲 **Phase 6 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Key Functions |
|------|---------|---------------|
| `__init__.py` | Public API | - |
| `drift.py` | Drift detection | `detect_covariate_drift()`, `detect_concept_drift()` |
| `triggers.py` | Retrain logic | `should_retrain()`, `RetrainTrigger` |
| `versioning.py` | Model management | `ModelRegistry`, version metadata |

## Drift Detection Methods

| Type | Method | Threshold |
|------|--------|-----------|
| Covariate | KS Test | p < 0.05 |
| Covariate | PSI | > 0.2 |
| Concept | Performance degradation | ROI < 1% for 30 days |
| Concept | Calibration drift | Brier > 0.25 |

## Retraining Triggers

1. **Scheduled:** Weekly full retrain on Mondays
2. **Drift-based:** Auto-trigger on PSI > 0.2 for 3+ features
3. **Performance:** Auto-trigger on 14-day rolling ROI < 0%
4. **Manual:** CLI command for forced retrain

## Integration Points

- **Upstream:** `backtest/` provides performance metrics
- **Downstream:** `models/` for retraining, `output/` for alerts

## Anti-Patterns

- ❌ Never retrain on concept drift without root cause analysis
- ❌ Never auto-deploy retrained models without validation
- ❌ Never ignore covariate drift (early warning of concept drift)
- ❌ Never use single metric for retraining decisions
