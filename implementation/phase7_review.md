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
- [x] Injury adjustments follow the specified Bayesian scenario algorithm (plays/sits expected value) and use real injury data source.
- [x] Prediction explainability provided via top contributing factors (context feature magnitude proxy).
- [x] All code adheres to dependency-injection and type-hint requirements from DEVELOPMENT_GUIDELINES.
- [ ] Coverage for prediction module verified at >90% (blocked by sandbox OpenMP crash).

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

## Loop 1 Results (Re-review)

### Previous Issues Status

1) Injury adjustment algorithm (plays/sits expectation)
- **UNRESOLVED**: `InjuryAdjuster.adjust_prediction_values()` now computes a play/sit expected adjustment, but it still does not run the model for both scenarios as required in `plan/phase7.md`. It uses RAPM-based replacement impact as a proxy instead of re-scoring the two scenarios.

2) Injury data source (real data)
- **RESOLVED**: `InjuryReportFetcher` now fetches live data from the NBA injury report JSON endpoint with caching and fallback behavior.

3) Dependency injection (InferencePipeline)
- **RESOLVED**: `InferencePipeline` now accepts injected `context_feature_builder` and `injury_adjuster` dependencies with sensible defaults.

4) Type hints / strict typing
- **RESOLVED**: Missing return type annotations and TypedDicts were added (e.g., `AdjustmentValuesDict`, `GameInfoDict`, typed returns on methods).

5) CLAUDE.md accuracy + unused imports
- **RESOLVED**: `nba_model/predict/CLAUDE.md` now documents transformer zeros for pre-game, and the unused `get_settings` import was removed.

### New Issues Found

1) Missing runtime dependency for injury fetcher
- `nba_model/predict/injuries.py` uses `requests`, but `pyproject.toml` does not include `requests` in dependencies. This makes `InjuryReportFetcher` fail in a clean environment.

2) Dependency injection gap for GNN graph builder
- `InferencePipeline._get_gnn_output()` instantiates `LineupGraphBuilder` internally. DEVELOPMENT_GUIDELINES require dependency injection for non-trivial dependencies; this is not yet injectable.

## Loop 2 Results (Final)

### Issues Fixed

1) **Injury adjustment algorithm (plays/sits two-scenario model scoring)**
   - **RESOLVED**: Added `LineupScorerCallback` type and `lineup_scorer` parameter to `InjuryAdjuster.__init__()`.
   - Implemented `_calculate_two_scenario_expected_values()` method that:
     * For each uncertain player, creates two lineup scenarios (with/without player)
     * Calls the model via `lineup_scorer` callback for each scenario
     * Computes expected values: `E[X] = P(plays)*X_plays + P(sits)*X_sits`
     * Calculates uncertainty from variance of possible outcomes
   - `InferencePipeline` provides `_score_lineup()` callback that runs full model inference.
   - Falls back to RAPM-based approximation when lineup_scorer or lineups not provided.

2) **Missing dependency (requests)**
   - **RESOLVED**: Added `requests>=2.28` to `pyproject.toml` dependencies.

3) **DI gap for LineupGraphBuilder**
   - **RESOLVED**: Added `lineup_graph_builder: LineupGraphBuilderProtocol | None` parameter to `InferencePipeline.__init__()`.
   - Added `_get_lineup_graph_builder()` method that creates default or returns injected instance.
   - Updated `_get_gnn_output()` to use the injectable builder.

### Additional Fixes

- Added proper type annotations for protocols (`InjuryAdjusterProtocol`, `LineupGraphBuilderProtocol`).
- Fixed mypy type narrowing issues with assertions for `self._gnn`, `self._fusion`, `self.lineup_scorer`.
- Removed unused variable assignments in `_calculate_two_scenario_expected_values()`.
- Fixed Callable import (from `collections.abc` instead of `typing`).

## Overall Assessment

READY

All three Loop 1 issues have been resolved:
1. Injury adjustment now implements true two-scenario model scoring per phase7.md spec
2. `requests` dependency added to pyproject.toml
3. `LineupGraphBuilder` is now injectable via constructor parameter

### Remaining Issues / Concerns

- Coverage and test execution could not be validated due to sandbox OpenMP/Signal 6 errors (documented as environment issue, not code issue).

### Summary Across All Loops

- Implemented live injury fetching with caching and Bayesian probability adjustments.
- Added true two-scenario model scoring with lineup-based inference callbacks.
- Closed DI and typing gaps (injectable builders, protocols, strict return types).
- Updated docs to match behavior and added missing runtime dependency (`requests`).
