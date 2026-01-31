# Phase 3 Implementation Review - Loop 1 Re-Review

**Date:** 2026-01-31
**Scope:** Phase 3 feature engineering after Loop 1 fixes

## Requirements Compliance Checklist

- [x] RAPM calculator implemented with sparse design matrix, time-decay weighting, min-minutes filter, ridge regression, and lambda cross-validation. (`nba_model/features/rapm.py`)
- [x] RAPM outputs persisted to `player_rapm` table in CLI pipeline. (`nba_model/cli.py`)
- [x] Spatial convex hull spacing metrics (hull area, centroid, corner density, avg distance), KDE density maps, and gravity overlap implemented. (`nba_model/features/spatial.py`)
- [!] `SpacingCalculator.min_shots` is not enforced in `calculate_lineup_spacing`; calculations proceed for any >=4 shots. (`nba_model/features/spatial.py`)
- [x] Lineup spacing persisted to `lineup_spacing` table in CLI pipeline. (`nba_model/cli.py`)
- [x] Fatigue calculator provides rest days, travel distance, schedule flags (B2B, 3-in-4, 4-in-5), and player load metrics with complete arena coordinates. (`nba_model/features/fatigue.py`)
- [x] Event parser implements turnover classification, shot context parsing, and shot clock usage categorization. (`nba_model/features/parsing.py`)
- [x] Season normalizer supports z-score by season, persistence, and required metrics including `fg3a_rate` and `points_per_game`. (`nba_model/features/normalization.py`)
- [x] CLI `features build`, `features rapm`, and `features spatial` commands present; build order is normalization → RAPM → spacing → fatigue with tqdm progress. (`nba_model/cli.py`)
- [x] Unit + integration tests for RAPM, spatial, fatigue, parsing, normalization, and feature pipeline present and passing. (`tests/unit/features/`, `tests/integration/test_feature_pipeline.py`)
- [x] Coverage requirement met (>= 75% overall).
- [!] Phase 3 spec says `transform()` should raise error on missing season; implementation warns and skips instead. (`nba_model/features/normalization.py`)
- [!] Shot clock usage spec mentions inbound possession starts; implementation only handles rebounds/turnovers (and resets on made shots). (`nba_model/features/parsing.py`)
- [!] CLAUDE.md accuracy: root claims 85% coverage target while test docs/guidelines specify 75% overall and 80% for new code. (`CLAUDE.md`, `tests/CLAUDE.md`, `DEVELOPMENT_GUIDELINES.md`)

## Test Results & Coverage

- `pytest tests/ -v`: **535 passed**, **1 warning** (SQLAlchemy identity conflict warning from `tests/unit/data/test_db.py`).
- `pytest --cov=nba_model --cov-report=term-missing --cov-fail-under=75`: **79.32% total coverage** (requirement met).
- Critical-path 90% coverage target (betting/prediction logic) not explicitly reported by tooling; no evidence of targeted enforcement beyond overall coverage.

## Issues Found

- **UNRESOLVED:** Missing-season handling in season normalization diverges from Phase 3 requirement (should raise, currently warns and skips). (`nba_model/features/normalization.py`)
- **UNRESOLVED:** Shot clock usage does not account for inbound possession starts as specified. (`nba_model/features/parsing.py`)
- **UNRESOLVED:** `SpacingCalculator.min_shots` configuration is unused, so minimum sample thresholds rely on callers rather than the calculator. (`nba_model/features/spatial.py`)
- **UNRESOLVED:** Coverage targets in CLAUDE docs are inconsistent (85% vs 75%/80%); needs alignment. (`CLAUDE.md`, `tests/CLAUDE.md`, `DEVELOPMENT_GUIDELINES.md`)
- **RESOLVED (Loop 1):** Fatigue indicator keys updated to `3_in_4`/`4_in_5`, normalization defaults include `fg3a_rate`/`points_per_game`, features build now includes fatigue step and tqdm progress, and feature pipeline integration test added. (See latest commit)

## Overall Assessment

Phase 3 implementation is functionally complete and tests pass with acceptable coverage. The core feature engineering components (RAPM, spatial metrics, fatigue, parsing, normalization, and CLI integration) meet most requirements. Remaining gaps are limited to spec mismatches in normalization missing-season behavior, inbound possession handling for shot clock usage, and documentation consistency for coverage targets.
