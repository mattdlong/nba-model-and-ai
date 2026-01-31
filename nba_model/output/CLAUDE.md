# Output Generation

## Responsibility

Generates human-readable outputs: GitHub Pages dashboard, daily reports, and performance summaries. Transforms predictions and metrics into static HTML/JSON for deployment.

## Status

**Phase 8 - Complete**

## Module Structure

| File | Purpose | Key Classes |
|------|---------|-------------|
| `__init__.py` | Public API | All exports |
| `reports.py` | Report generation | `ReportGenerator` |
| `charts.py` | Chart data generation | `ChartGenerator` |
| `dashboard.py` | Static site builder | `DashboardBuilder` |

## Key Classes

### ReportGenerator

Generates structured reports for predictions, performance, and model health.

```python
from nba_model.output import ReportGenerator

generator = ReportGenerator()

# Daily predictions report
report = generator.daily_predictions_report(predictions, signals)
# Returns: {date, games, signals, summary}

# Performance report
perf = generator.performance_report("week", metrics=backtest_metrics)
# Returns: {period, accuracy, roi, clv, calibration_curve, ...}

# Model health report
health = generator.model_health_report(drift_result, recent_metrics)
# Returns: {status, drift_detected, retraining_recommended, ...}
```

### ChartGenerator

Produces Chart.js-compatible data structures for visualization.

```python
from nba_model.output import ChartGenerator

charts = ChartGenerator()

# Bankroll growth chart
bankroll_data = charts.bankroll_chart(history, dates=date_list)
# Returns: {labels, datasets}

# Calibration curve
calibration = charts.calibration_chart(predictions, actuals)
# Returns: {labels, datasets, metadata}

# ROI by month
roi_data = charts.roi_by_month_chart(bets)
# Returns: {labels, datasets, metadata}
```

### DashboardBuilder

Builds complete static site for GitHub Pages deployment.

```python
from nba_model.output import DashboardBuilder

builder = DashboardBuilder(output_dir="docs", template_dir="templates")

# Build full site structure
file_count = builder.build_full_site()

# Update predictions page and JSON
builder.update_predictions(predictions, signals)

# Update performance metrics
builder.update_performance(metrics=metrics, bankroll_history=history)

# Update model health page
builder.update_model_health(drift_results, recent_metrics)

# Archive day to history
builder.archive_day(date(2024, 1, 15))
```

## Dashboard Pages

| Page | Content | Update Frequency |
|------|---------|-----------------|
| `index.html` | Summary stats, top signals | Daily |
| `predictions.html` | All game predictions | Daily pre-game |
| `history.html` | Performance charts, metrics | Daily post-results |
| `model.html` | Drift status, health | On demand |

## Output Formats

### JSON API Files

```
docs/api/
├── today.json        # Current predictions and signals
├── signals.json      # Active betting signals
├── performance.json  # Metrics and chart data
└── history/
    └── YYYY-MM-DD.json  # Archived daily data
```

### HTML Dashboard

```
docs/
├── index.html
├── predictions.html
├── history.html
├── model.html
└── assets/
    ├── style.css
    └── charts.js
```

## CLI Commands

```bash
# Build complete dashboard
nba-model dashboard build

# Build to custom directory
nba-model dashboard build --output docs-staging

# Deploy to GitHub Pages
nba-model dashboard deploy --message "Update predictions"

# Update with latest data
nba-model dashboard update
```

## Daily Workflow

```bash
# Morning workflow
nba-model data update            # Fetch yesterday's results
nba-model features build         # Update features
nba-model monitor drift          # Check for drift
nba-model predict today          # Generate predictions
nba-model dashboard update       # Archive + update dashboard
```

## Chart Types

| Chart | Method | Purpose |
|-------|--------|---------|
| Bankroll Line | `bankroll_chart()` | Track bankroll growth |
| Calibration | `calibration_chart()` | Model probability accuracy |
| ROI Bar | `roi_by_month_chart()` | Monthly performance |
| Win Rate Trend | `win_rate_trend_chart()` | Rolling win rate |
| Edge Distribution | `edge_distribution_chart()` | Edge histogram |

## Health Status Levels

| Status | Condition | Action |
|--------|-----------|--------|
| `healthy` | No drift, metrics OK | None |
| `warning` | Minor drift or metric decline | Monitor closely |
| `critical` | Multiple drift indicators or poor performance | Retrain recommended |

## Integration Points

**Upstream (receives from):**
- `predict/` - GamePrediction, BettingSignal
- `backtest/` - FullBacktestMetrics, Bet
- `monitor/` - DriftCheckResult

**Downstream (outputs to):**
- GitHub Pages (external deployment)
- Local file system (docs/ directory)

## Anti-Patterns

- Never commit API keys to dashboard code
- Never show betting signals before market close
- Never include absolute bet amounts (only percentages)
- Never regenerate history files (append only via archive)
- Never deploy without building first
- Never skip archiving before updating predictions
