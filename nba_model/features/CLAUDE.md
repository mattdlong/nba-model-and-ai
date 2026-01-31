# Feature Engineering

## Responsibility

Transforms raw NBA data into ML-ready features. Handles RAPM calculation, spatial analytics, fatigue metrics, and event parsing.

## Status

🔲 **Phase 3 - Not Started** (stub `__init__.py` only)

## Planned Structure

| File | Purpose | Key Outputs |
|------|---------|-------------|
| `__init__.py` | Public API | - |
| `rapm.py` | RAPM calculation | ORAPM, DRAPM, total per player |
| `spatial.py` | Convex hull spacing | Hull area, centroid, corner density |
| `fatigue.py` | Rest/travel metrics | B2B flags, travel distance, load |
| `parsing.py` | Event extraction | Regex-parsed play types |
| `normalization.py` | Z-score by season | Normalized feature vectors |

## Key Algorithms

1. **RAPM:** Ridge regression on stint design matrix (λ=5000)
2. **Convex Hull:** scipy.spatial for shot distribution area
3. **Time Decay:** Exponential weighting (τ=180 days half-life)

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| RAPM | 3 | ORAPM, DRAPM, total |
| Spatial | 5+ | Hull area, centroid, density |
| Fatigue | 4+ | B2B, days rest, travel |
| Contextual | 5+ | Home/away, altitude, timezone |

## Integration Points

- **Upstream:** `data/` provides ORM models
- **Downstream:** `models/` consumes feature tensors

## Anti-Patterns

- ❌ Never compute RAPM without time decay weighting
- ❌ Never use raw stats without z-score normalization
- ❌ Never hardcode feature dimensions (use config)
