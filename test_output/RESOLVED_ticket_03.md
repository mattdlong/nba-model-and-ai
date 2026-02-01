Steps to reproduce: 1. Run python -m nba_model.cli predict signals without data collected or models trained

Expected behaviour: Should indicate no current games available or clear error about missing prerequisites

Actual behaviour: Likely fails due to missing data or models

Why it is incorrect: CLI should handle missing prerequisites gracefully, not crash

---

## Fix Description

**Status: RESOLVED**

**Fix Applied:**
Added explicit prerequisite checks to the `predict signals` command in `nba_model/cli.py`:

1. **Database Check** (already existed): Verifies database exists before proceeding
2. **Model Check** (NEW): Added check using `ModelRegistry.get_latest_version()` to verify trained models exist

When no trained models are found, the CLI now displays a clear, helpful error message:
```
Error: No trained models found.
To train a model, run:
  1. python -m nba_model.cli data collect --seasons 2023-24
  2. python -m nba_model.cli features build
  3. python -m nba_model.cli train all
```

**Files Modified:**
- `nba_model/cli.py`: Added model existence check before attempting signal generation

**Additional Changes:**
- Also updated `predict game` and `predict date` commands with the same check for consistency

**Verification:**
Running `python -m nba_model.cli predict signals` without trained models now shows the helpful error message instead of crashing with an exception.
