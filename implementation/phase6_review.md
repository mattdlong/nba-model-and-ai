# Phase 6 Implementation Review

Date: 2026-01-31
Reviewer: Codex

## Requirements Compliance Checklist

### Core Phase 6 Requirements
- [x] Covariate drift detection (KS + PSI) implemented in `nba_model/monitor/drift.py`.
- [x] Concept drift detection (accuracy + Brier) implemented in `nba_model/monitor/drift.py`.
- [x] Retraining trigger logic implemented in `nba_model/monitor/triggers.py`.
- [x] Model versioning manager implemented in `nba_model/monitor/versioning.py`.
- [ ] CLI monitor commands fully wired to real data and drift checks.
  - `monitor drift` does not instantiate `DriftDetector` or run drift checks; it only counts games and prints “ready” messaging.
  - `monitor trigger` uses placeholder context (no drift detector, no recent data, no bets, no games since training) and shows incorrect “days since training” description.
- [ ] Version comparison with test data is not implemented.
  - `_compute_metrics_on_data()` is a placeholder and does not load models or run inference when `test_data` is provided.
- [ ] Version list ordering does not follow “creation date descending”.
  - `ModelVersionManager.list_versions()` relies on `ModelRegistry.list_versions()` (sorted by semantic version), not by metadata creation date.
- [ ] Metadata schema does not include `created_at` in `metadata.json`.
  - ModelRegistry writes `training_date` only; `created_at` is not stored as specified in `phase6.md`.

### Development Guidelines (Testing)
- [ ] Coverage requirement not verified (pytest coverage run did not complete).
- [ ] Integration tests required by Phase 6 are missing (no `tests/integration` for monitor pipelines).

### Phase 6 Test Requirements
- [ ] Drift tests missing empty-dataframe edge case.
- [ ] PSI heavy-shift test only asserts `> 0.1`, not `> 0.2` as required.
- [ ] Version comparison tests do not validate live inference on `test_data`.

### CLAUDE.md Accuracy
- [x] CLAUDE.md files exist in expected locations (module + tests).
- [ ] `nba_model/monitor/CLAUDE.md` marked “Phase 6 - Complete,” but CLI commands are still stubbed.
- [ ] `tests/unit/monitor/CLAUDE.md` claims empty-dataframe edge cases are covered; they are not.

## Test Results and Coverage

### Commands Run
- `source .venv/bin/activate && pytest tests/ -v` → failed (Sandbox Signal 6)
- `source .venv/bin/activate && pytest tests/unit/monitor/test_drift.py -v` → failed (Sandbox Signal 6)

### Outcome
- Full test suite did not complete due to runtime abort (Signal 6). No coverage data could be verified.
- Coverage requirement from `DEVELOPMENT_GUIDELINES.md` (>= 75% overall) could not be confirmed.

## Runtime/CLI Verification

- `source .venv/bin/activate && nba-model monitor --help` → OK
- Import checks:
  - Importing `nba_model.monitor` (or any submodule) caused a runtime abort (Signal 6). Root cause appears to be package-level import of `torch` via `nba_model.monitor.__init__`.

## Issues Found

1. **CLI monitor drift is non-functional**
   - `nba-model monitor drift` does not build reference/recent feature data or call `DriftDetector.check_drift`. It also never exits 1 on drift detection as required.
   - Files: `nba_model/cli.py`

2. **CLI monitor trigger uses placeholder context**
   - No drift detector, no recent data, no recent bets, no game counts; output is not based on real signals.
   - The “days since training” display is always 0 due to `last_train_date - last_train_date`.
   - Files: `nba_model/cli.py`

3. **Live version comparison not implemented**
   - `ModelVersionManager.compare_versions(..., test_data=...)` does not load models or compute metrics; it returns stored metrics even when test data is provided.
   - Files: `nba_model/monitor/versioning.py`

4. **Metadata schema mismatch**
   - `created_at` is required in metadata per Phase 6, but metadata currently stores `training_date` only.
   - Files: `nba_model/models/registry.py`, `nba_model/monitor/versioning.py`

5. **Version listing order not per spec**
   - Versions are sorted by semantic version rather than metadata creation date.
   - Files: `nba_model/monitor/versioning.py`, `nba_model/models/registry.py`

6. **Missing Phase 6 integration tests**
   - No integration tests for drift pipeline, trigger pipeline, or version lifecycle as required in `plan/phase6.md`.
   - Files: `tests/integration/*`

7. **Unit test gaps vs Phase 6 requirements**
   - Drift tests do not include empty-dataframe handling.
   - PSI heavy-shift test does not assert `> 0.2` (only `> 0.1`).
   - Files: `tests/unit/monitor/test_drift.py`

8. **CLAUDE.md inaccuracies**
   - Module/test docs claim completion and edge-case coverage that are not yet true.
   - Files: `nba_model/monitor/CLAUDE.md`, `tests/unit/monitor/CLAUDE.md`

## Overall Assessment

Phase 6 has strong foundational class implementations, but key requirements are incomplete: CLI commands are still stubs, version comparison with live inference is not implemented, metadata schema does not match the spec, integration tests are missing, and tests do not fully meet stated requirements. The documented “Phase 6 - Complete” status is premature. Additional work is required before Phase 6 can be considered compliant.
