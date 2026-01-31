# Tests for Output Module

## Status

**Phase 8 - Complete**

## Test Files

| File | Tests | Source Module |
|------|-------|---------------|
| `test_reports.py` | ReportGenerator tests | `nba_model/output/reports.py` |
| `test_charts.py` | ChartGenerator tests | `nba_model/output/charts.py` |
| `test_dashboard.py` | DashboardBuilder tests | `nba_model/output/dashboard.py` |

## Test Coverage

### ReportGenerator Tests

- `test_daily_predictions_report_returns_dict` - Correct structure
- `test_daily_predictions_report_formats_games` - Game formatting
- `test_daily_predictions_report_formats_signals` - Signal formatting
- `test_daily_predictions_report_calculates_summary` - Summary stats
- `test_daily_predictions_report_empty_lists` - Empty input handling
- `test_performance_report_with_metrics` - Metrics formatting
- `test_performance_report_with_bets` - Bet calculation
- `test_performance_report_invalid_period` - Error handling
- `test_model_health_report_healthy` - Healthy status
- `test_model_health_report_with_drift` - Drift detection
- `test_model_health_report_critical_status` - Critical status

### ChartGenerator Tests

- `test_bankroll_chart_returns_chartjs_structure` - Chart.js format
- `test_bankroll_chart_with_dates` - Date labels
- `test_bankroll_chart_color_for_profit` - Green for gains
- `test_bankroll_chart_color_for_loss` - Red for losses
- `test_calibration_chart_bins_predictions` - Probability binning
- `test_calibration_chart_perfect_line` - Reference line
- `test_roi_by_month_aggregates_correctly` - Monthly aggregation
- `test_roi_by_month_color_codes_positive_negative` - Color coding
- `test_win_rate_trend_rolling_window` - Rolling calculation

### DashboardBuilder Tests

- `test_build_full_site_creates_directories` - Directory structure
- `test_build_full_site_creates_json_files` - JSON file creation
- `test_build_full_site_renders_html` - HTML rendering
- `test_update_predictions_writes_today_json` - Today file update
- `test_update_predictions_renders_predictions_page` - Page rendering
- `test_archive_day_creates_history_file` - Archiving
- `test_build_without_templates` - No templates handling

## Running Tests

```bash
# Run output module tests
pytest tests/unit/output/ -v

# With coverage
pytest tests/unit/output/ --cov=nba_model.output

# Specific test file
pytest tests/unit/output/test_reports.py -v
```

## Mock Objects

Tests use mock dataclasses that mirror production types:
- `MockGamePrediction` - Mimics `predict.inference.GamePrediction`
- `MockBettingSignal` - Mimics `predict.signals.BettingSignal`
- `MockFullBacktestMetrics` - Mimics `backtest.metrics.FullBacktestMetrics`
- `MockBet` - Mimics `types.Bet`

## Fixtures

| Fixture | Purpose |
|---------|---------|
| `generator` | ReportGenerator instance |
| `chart_generator` | ChartGenerator instance |
| `builder` | DashboardBuilder with temp dirs |
| `output_dir` | Temporary output directory |
| `template_dir` | Temp templates with minimal HTML |
| `sample_predictions` | List of MockGamePrediction |
| `sample_signals` | List of MockBettingSignal |
| `sample_bets` | List of MockBet |

## Anti-Patterns

- Never use real file paths in tests
- Never depend on template existence for core logic tests
- Never skip edge case testing (empty inputs, None values)
- Never test implementation details (focus on behavior)
