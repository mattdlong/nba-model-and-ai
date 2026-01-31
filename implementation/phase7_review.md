# Phase 7 Implementation Review (Codex)

Date: 2026-01-31
Reviewer: Codex CLI

## Requirements Compliance Checklist

- [x] `nba_model/predict/` modules exist (`inference.py`, `injuries.py`, `signals.py`, `__init__.py`).
- [x] `InferencePipeline` exposes `predict_game`, `predict_today`, `predict_date`.
- [x] `GamePrediction` includes required identifiers, raw outputs, adjusted outputs, uncertainty, top factors, metadata, expected lineups.
- [x] Signal generation supports moneyline, spread, total markets and applies devig + Kelly sizing.
- [x] CLI commands added for `predict today|game|date|signals`.
- [x] Tests for predict modules exist (unit + integration).
- [ ] Injury adjustments follow the specified Bayesian scenario algorithm (plays/sits expected value) and use real injury data source.
- [ ] Prediction explainability uses explicit feature importances (current uses abs(feature value) proxy only).
- [ ] All code adheres to dependency-injection and type-hint requirements from DEVELOPMENT_GUIDELINES.
- [ ] Coverage for prediction module verified at >90%.

## Test Results and Coverage

- `pytest tests/ -v` failed with sandbox `Signal(6)` (OpenMP/SHM crash). Retried with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 KMP_DUPLICATE_LIB_OK=TRUE` and got the same `Signal(6)`.
- Import check `from nba_model.predict import InferencePipeline, InjuryAdjuster, SignalGenerator` failed with the same `Signal(6)`.
- Coverage not measured due to the crash (requirement is >90% for prediction module).

Note: Per instructions, the OpenMP/Signal 6 crash is a sandbox issue and not attributed to code correctness.

## Issues Found

1) Requirement gap: injury adjustment algorithm and data source
- `nba_model/predict/injuries.py` does not implement the required two-scenario (plays/sits) expectation calculation. Instead it applies a RAPM-based adjustment without running both scenarios.
- `_get_team_injuries` returns an empty list and `InjuryReportFetcher` is a stub, so injuries never affect predictions.

2) Guidelines violation: dependency injection
- `InferencePipeline` instantiates dependencies internally (e.g., `ContextFeatureBuilder`, `InjuryAdjuster`, `LineupGraphBuilder`). DEVELOPMENT_GUIDELINES require dependency injection for non-trivial dependencies.

3) Guidelines violation: missing type hints / strict typing
- Missing return type annotations on methods such as `InferencePipeline._build_player_features_df` and `InjuryAdjuster.adjust_prediction`.
- Some methods return untyped `dict` without TypedDict or concrete dataclass return types.

4) Linting risk: unused import
- `nba_model/predict/inference.py` imports `get_settings` but never uses it (ruff F401).

5) CLAUDE.md accuracy
- `nba_model/predict/CLAUDE.md` implies Transformer output uses play-by-play, while implementation always uses zeros for pre-game; documentation should match actual behavior.

## Overall Assessment

NOT READY

Core structure is in place, but injury adjustment behavior and data fetching do not meet Phase 7 requirements, and there are guideline compliance gaps (dependency injection, typing, lint). Tests and coverage could not be validated due to sandbox OpenMP/Signal 6 errors.
