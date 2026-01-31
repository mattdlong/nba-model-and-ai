# Jinja2 Templates

## Purpose

HTML templates for dashboard generation. Used by `nba_model/output/dashboard.py` to generate static site.

## Structure

| File | Purpose |
|------|---------|
| `base.html` | Base template with layout |
| `predictions.html` | Predictions page |
| `components/` | Reusable template components |

## Template Variables

Common variables passed to all templates:
- `generated_at` - Generation timestamp
- `model_version` - Current model version
- `site_title` - Site title

## Usage

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("predictions.html")
html = template.render(predictions=predictions, generated_at=now)
```

## Anti-Patterns

- ❌ Never include logic in templates (do in Python)
- ❌ Never hardcode URLs
- ❌ Never include sensitive data
