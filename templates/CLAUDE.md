# Jinja2 Templates

## Purpose

HTML templates for dashboard generation. Used by `nba_model/output/dashboard.py` to generate static site.

## Structure

| File | Purpose |
|------|---------|
| `base.html` | Base template with layout, header, navigation, footer |
| `index.html` | Main dashboard with summary stats and top signals |
| `predictions.html` | Today's predictions and betting signals |
| `history.html` | Historical performance, charts, bet history |
| `model.html` | Model health, drift status, feature stability |

## Template Variables

Variables passed by `DashboardBuilder` to each template:

### Common (all templates)
- `generated_at` - ISO format timestamp of generation
- `page_name` - Current page name (derived from template filename)

### index.html
- `summary` - Daily summary dict (total_games, total_signals, high_confidence_signals, avg_edge)
- `top_signals` - Top 3 signals for preview
- `prediction_date` - Date string for predictions
- `performance` - Performance metrics (win_rate, roi, clv, total_bets)
- `health` - Model health status dict
- `model_info` - Model metadata (version, etc.)

### predictions.html
- `predictions` - List of game prediction dicts
- `signals` - List of betting signal dicts
- `summary` - Summary statistics dict
- `injuries` - Dict of team -> player injury list (optional)

### history.html
- `metrics` - Performance metrics dict (period, roi, accuracy, clv, etc.)
- `charts` - Pre-computed chart data dicts
- `date_range` - Dict with start/end dates (optional)
- `bet_history` - List of historical bet dicts (optional)

### model.html
- `health` - Health status dict (status, drift_detected, features_drifted, etc.)
- `model_info` - Model metadata dict (version, training_date, feature_importance, etc.)

## Static Assets

Assets are stored in `docs/assets/` (not in templates directory):
- `style.css` - Dashboard styling
- `charts.js` - Chart.js integration and data loading

## Usage

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("predictions.html")
html = template.render(
    predictions=predictions,
    signals=signals,
    summary=summary,
    generated_at=datetime.now().isoformat(),
)
```

## Anti-Patterns

- Never include business logic in templates (do in Python)
- Never hardcode URLs (use relative paths)
- Never include sensitive data (API keys, credentials)
- Never reference external CDNs without fallback
