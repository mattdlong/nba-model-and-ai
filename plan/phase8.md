# Phase 8: Documentation & Polish

## Overview

This phase implements the output layer of the NBA quantitative trading system, encompassing report generation, static dashboard construction for GitHub Pages, and automation workflows for daily operations.

## Objectives

1. Implement report generation system for predictions and performance tracking
2. Build static GitHub Pages dashboard with Jinja2 templating
3. Create chart data generators for dashboard visualizations
4. Establish automation CLI commands for daily and weekly workflows

## Dependencies

- Phase 5: Backtest results and performance metrics
- Phase 6: Drift detection and model health status
- Phase 7: GamePrediction and BettingSignal dataclasses

## Module Specifications

### 8.1 Report Generation Module

**Location:** `nba_model/output/reports.py`

**ReportGenerator Class**

Implements three report generation methods:

1. **daily_predictions_report**: Accepts list of GamePrediction and BettingSignal objects. Returns dictionary containing date, per-game predictions, actionable signals, and aggregate summary statistics.

2. **performance_report**: Accepts period parameter (week/month/season). Queries historical predictions and outcomes. Returns dictionary with total predictions count, accuracy, ROI, CLV, calibration curve data points, and breakdown by bet type.

3. **model_health_report**: Accepts drift detection results and recent metrics. Returns dictionary with model status, drift indicators, feature stability metrics, and retraining recommendations.

All report methods return dictionaries suitable for JSON serialization and template rendering.

### 8.2 Dashboard Builder Module

**Location:** `nba_model/output/dashboard.py`

**DashboardBuilder Class**

Constructor accepts output_dir (default: 'docs') and template_dir (default: 'templates').

**Site Structure to Generate:**
```
docs/
├── index.html           # Main dashboard
├── predictions.html     # Today's predictions
├── history.html         # Historical performance
├── model.html           # Model info and health
├── api/                 # JSON data files
│   ├── today.json
│   ├── signals.json
│   ├── performance.json
│   └── history/
│       └── {YYYY-MM-DD}.json
└── assets/
    ├── style.css
    └── charts.js
```

**Methods:**

1. **build_full_site**: Generates complete static site from templates. Creates all HTML pages, copies static assets, and initializes JSON data directory structure.

2. **update_predictions**: Accepts GamePrediction and BettingSignal lists. Renders predictions.html template and writes today.json and signals.json to api directory.

3. **update_performance**: Accepts metrics dictionary. Renders updated performance data to performance.json.

4. **archive_day**: Accepts date parameter. Moves current day predictions to history directory with date-stamped filename.

**ChartGenerator Class**

Generates JavaScript-compatible chart data structures:

1. **bankroll_chart**: Accepts bankroll history list. Returns dictionary with labels (dates) and datasets (values) for line chart rendering.

2. **calibration_chart**: Accepts predictions (floats) and actuals (integers). Bins predictions into probability ranges and compares to actual win rates. Returns dictionary for calibration plot.

3. **roi_by_month_chart**: Accepts Bet list. Aggregates ROI by calendar month. Returns dictionary for bar chart rendering.

### 8.3 Dashboard Page Content

**index.html (Main Dashboard)**
- Model version badge and health status indicator
- Summary statistics cards: Win Rate, ROI, CLV, Total Bets
- Today's top 3 signals preview
- Bankroll growth chart (if simulation enabled)
- Quick links to other pages

**predictions.html**
- Date selector (default: today)
- Game cards showing: matchup, model predictions vs market lines, edge calculations
- Signal recommendations with confidence levels (high/medium/low)
- Key factors list per prediction
- Injury report integration showing GTD players

**history.html**
- Date range filter controls
- Performance metrics table with sortable columns
- Calibration curve visualization
- ROI time series chart
- Bet type breakdown pie chart
- Searchable/filterable bet history table

**model.html**
- Architecture diagram placeholder
- Current version display with training date
- Feature importance ranking table
- Drift detection status indicators (green/yellow/red)
- Last drift check timestamp
- Next scheduled retraining date

### 8.4 Jinja2 Templates

Create base template with common layout (header, navigation, footer). Child templates extend base and define content blocks.

Template variables passed from Python:
- predictions: List of prediction dictionaries
- signals: List of signal dictionaries
- metrics: Performance metrics dictionary
- model_info: Model metadata dictionary
- charts: Pre-computed chart data dictionaries
- generated_at: Timestamp string

### 8.5 Static Assets

**style.css**
- Responsive grid layout
- Card component styles
- Table styling with alternating rows
- Color coding for positive/negative values
- Status indicator colors (green/yellow/red)

**charts.js**
- Chart.js initialization functions
- Data loading from JSON endpoints
- Chart configuration objects for each chart type

### 8.6 CLI Commands

**dashboard subcommand group:**

1. `nba-model dashboard build`: Invokes DashboardBuilder.build_full_site(). Outputs build status and file count.

2. `nba-model dashboard deploy`: Executes git add docs/ && git commit && git push to deploy to GitHub Pages. Requires clean working directory.

**Automation integration:**

Daily workflow sequence (intended for cron):
```
nba-model data update
nba-model features build
nba-model monitor drift
nba-model predict today
nba-model dashboard build
```

Weekly workflow sequence:
```
nba-model train all
nba-model backtest run
nba-model dashboard build
```

## Technical Specifications

| Component | Output Format | Update Frequency |
|-----------|---------------|------------------|
| today.json | JSON | Daily pre-game |
| signals.json | JSON | Daily pre-game |
| performance.json | JSON | Daily post-results |
| history/{date}.json | JSON | Daily archive |
| HTML pages | Static HTML | On build command |

## Implementation Order

1. ReportGenerator class with all three report methods
2. ChartGenerator class with chart data methods
3. DashboardBuilder class with site generation logic
4. Jinja2 templates (base, index, predictions, history, model)
5. Static assets (CSS, JS)
6. CLI commands integration
7. End-to-end integration testing

## Testing Requirements

### Unit Tests

1. **ReportGenerator Tests**
   - Verify daily_predictions_report returns correct dictionary structure with all required keys
   - Verify performance_report correctly aggregates metrics over specified periods
   - Verify model_health_report includes drift status and recommendations
   - Test with empty input lists to ensure graceful handling

2. **ChartGenerator Tests**
   - Verify bankroll_chart returns valid Chart.js compatible data structure
   - Verify calibration_chart correctly bins predictions into ranges
   - Verify roi_by_month_chart aggregates by calendar month boundaries
   - Test edge cases: single data point, empty data, negative values

3. **DashboardBuilder Tests**
   - Verify build_full_site creates expected directory structure
   - Verify all HTML files are valid (contain doctype, html tags)
   - Verify JSON files are valid JSON and match expected schema
   - Verify update_predictions overwrites existing files correctly
   - Verify archive_day moves files to correct history location with proper naming

### Integration Tests

1. **Template Rendering**
   - Render each template with mock data and verify no Jinja2 errors
   - Verify rendered HTML contains expected dynamic content
   - Verify chart data placeholders are populated correctly

2. **CLI Command Tests**
   - Verify `dashboard build` command executes without error
   - Verify `dashboard build` creates output in correct directory
   - Verify command provides appropriate exit codes (0 success, non-zero failure)

3. **End-to-End Workflow**
   - Generate mock predictions using Phase 7 components
   - Pass through ReportGenerator to create reports
   - Pass through DashboardBuilder to generate site
   - Verify generated site is complete and internally consistent
   - Open generated HTML in headless browser and verify no JavaScript errors

### Manual Verification

1. Open generated index.html in browser and verify visual layout
2. Navigate between all pages and verify links work
3. Verify charts render with sample data
4. Verify responsive design at mobile viewport widths
5. Verify JSON API endpoints return expected data when fetched

## Success Criteria

- All HTML pages generate without template errors
- JSON files pass schema validation
- Charts render correctly with test data
- CLI commands complete with appropriate status messages
- Site builds in under 10 seconds for typical data volume
- Generated site works when served from file:// protocol (no server required)

## Files to Create

- `nba_model/output/__init__.py`
- `nba_model/output/reports.py`
- `nba_model/output/dashboard.py`
- `templates/base.html`
- `templates/index.html`
- `templates/predictions.html`
- `templates/history.html`
- `templates/model.html`
- `docs/assets/style.css`
- `docs/assets/charts.js`
- `tests/test_output/__init__.py`
- `tests/test_output/test_reports.py`
- `tests/test_output/test_dashboard.py`

---

Upon completion, commit all changes with message "feat: implement phase 8 - documentation and polish" and push to origin
