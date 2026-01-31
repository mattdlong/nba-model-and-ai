# Output Generation

## Responsibility

Generates human-readable outputs: GitHub Pages dashboard, daily reports, and performance summaries.

## Status

🔲 **Phase 8 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Key Functions |
|------|---------|---------------|
| `__init__.py` | Public API | - |
| `reports.py` | Report generation | `generate_daily_report()` |
| `dashboard.py` | Static site builder | `build_dashboard()`, `update_predictions()` |

## Dashboard Pages

| Page | Content | Update Frequency |
|------|---------|-----------------|
| `index.html` | Today's predictions | Daily 9am |
| `predictions.html` | Active signals | On new predictions |
| `history.html` | Past performance | Daily after games |
| `model.html` | Model metrics | Weekly |

## Output Formats

1. **JSON API** (`docs/api/`): Machine-readable for external tools
2. **HTML Dashboard** (`docs/`): Human-readable GitHub Pages site
3. **Console Reports**: Rich-formatted terminal output

## File Structure

```
docs/
├── index.html
├── predictions.html
├── history.html
├── model.html
├── api/
│   ├── today.json
│   ├── signals.json
│   └── history/
│       └── 2024-01-01.json
└── assets/
    ├── style.css
    └── charts.js
```

## Integration Points

- **Upstream:** `predict/` for signals, `backtest/` for performance
- **Downstream:** GitHub Pages (external)

## Anti-Patterns

- ❌ Never commit API keys to dashboard code
- ❌ Never show signals before market close
- ❌ Never include bet amounts (only percentages)
- ❌ Never regenerate history (append only)
