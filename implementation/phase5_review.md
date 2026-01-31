# Phase 5 Implementation Review (Codex)

Date: 2026-01-31
Reviewer: Codex

## Scope Reviewed
- `plan/implementation_plan.md`
- `DEVELOPMENT_GUIDELINES.md`
- `plan/phase5.md`
- `implementation/phase5_summary.md`
- Latest commit: `git log -1 --stat`
- Phase 5 code and tests in `nba_model/backtest/` and `tests/unit/backtest/`
- CLI backtest commands in `nba_model/cli.py`
- CLAUDE docs (root + backtest + backtest tests)

## Requirements Compliance Checklist (Phase 5)
- Walk-forward validation engine implemented with chronological folds: ✅
- Devigging methods (multiplicative, power, Shin): ⚠️ (Shin uses overround heuristic, not iterative solve)
- Kelly Criterion sizing with fractional Kelly and caps: ✅
- Backtest result container with required fields: ⚠️ (no `clv` property; provides `avg_clv` instead)
- Performance metrics with required calculations: ✅
- CLI commands (`backtest run/report/optimize`): ⚠️ (`report` does not parse results file)
- End-to-end backtest run (odds, devig, kelly, metrics): ⚠️ (no odds provider integration; defaults to synthetic odds)

## Development Guidelines Compliance
- Type hints present and used consistently in new modules: ✅
- Python 3.11+ syntax and dataclasses used: ✅
- Testing requirements (pytest, coverage): ⚠️ (tests crash with Signal 6; coverage not verified)

## Test Results
Commands executed:
- `source .venv/bin/activate && pytest tests/ -v`
  - Result: **FAILED** (process terminated with Signal 6)
- `source .venv/bin/activate && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 KMP_DUPLICATE_LIB_OK=TRUE pytest tests/ -v`
  - Result: **FAILED** (process terminated with Signal 6)

Notes:
- The OpenMP/MKL environment variables recommended in `tests/conftest.py` did not prevent the crash.
- Because the test suite could not complete, coverage could not be collected or verified.

Coverage Requirement:
- `pyproject.toml` sets `fail_under = 75` for overall coverage. Current status **unknown** due to test crash.
- `tests/CLAUDE.md` additionally states 90% coverage for critical betting/prediction paths; not verifiable at this time.

## Runtime/CLI Verification
- `python -m nba_model.cli backtest --help`: ✅ (command group and subcommands display correctly)
- Import backtest modules: ✅
  - `from nba_model.backtest import WalkForwardEngine, DevigCalculator, KellyCalculator, BacktestMetricsCalculator`

## CLAUDE.md Checks
- CLAUDE files exist in expected locations (root, backtest, tests): ✅
- Accuracy checks:
  - `nba_model/backtest/CLAUDE.md` states “Never use full Kelly (always fractional, max 0.5x)” but `KellyCalculator` permits `fraction` up to 1.0. ⚠️
  - Root `CLAUDE.md` references “75% minimum coverage (80% for new code)” while enforcement in `pyproject.toml` is `fail_under = 75` only. ⚠️

## Issues Found
1. **`backtest report` command is a stub**
   - File: `nba_model/cli.py`
   - The command prints “Result file parsing not yet implemented.” This does not meet Phase 5 requirement to generate reports from stored results.

2. **Shin devigging method not implemented as specified**
   - File: `nba_model/backtest/devig.py`
   - The method uses a simplified heuristic (`z ≈ overround`) rather than an iterative solve for `z` as required.

3. **Kelly edge check uses implied odds, not devigged market probability**
   - File: `nba_model/backtest/kelly.py`
   - `KellyCalculator.calculate` uses implied probability (includes vig) to compute edge, which can reject bets that pass the devigged edge threshold from Phase 5.

4. **CLV calculation only supports home moneyline odds**
   - File: `nba_model/backtest/engine.py`
   - Closing odds collection only uses `home_ml` and does not support away/spread/total bets, so CLV metrics are incomplete.

5. **Phase 5 integration tests missing**
   - Requirements call for integration tests running the backtest on a sample dataset (>=200 games). Only unit tests exist for backtest modules.

6. **Backtest metrics API mismatch**
   - File: `nba_model/backtest/metrics.py`
   - Phase 5 required `calculate_all(result: BacktestResult) -> dict`, but implementation takes `bets` + `bankroll_history` and returns a `FullBacktestMetrics` object. This is workable but not aligned with the stated API.

## Overall Assessment
Phase 5 introduces a substantial backtesting implementation and aligns with most core requirements (walk-forward folds, devigging, Kelly sizing, and metrics). However, there are notable gaps: report generation is incomplete, Shin devigging does not follow the specified iterative approach, CLV handling is partial, and the test suite could not be executed due to a Signal 6 crash which blocks coverage verification. These issues should be addressed before Phase 5 can be considered fully complete.

---

## Loop 1 Results (Re-Review)

### Issue Resolution Status
- ✅ **Resolved:** `backtest report` command implemented and parses JSON results (`nba_model/cli.py`).
- ❌ **Unresolved:** Shin devigging still uses `z ≈ overround` approximation (no iterative solve) (`nba_model/backtest/devig.py`).
- ✅ **Resolved:** Kelly edge check now accepts devigged `market_prob` and engine passes it (`nba_model/backtest/kelly.py`, `nba_model/backtest/engine.py`).
- ✅ **Resolved:** CLV uses `(game_id, bet_type, side)` mapping; supports moneyline/spread/total (`nba_model/backtest/engine.py`, `nba_model/backtest/metrics.py`).
- ✅ **Resolved:** Integration tests added with 800-game synthetic dataset (`tests/integration/test_backtest_pipeline.py`).
- ❌ **Unresolved:** Metrics API still not aligned with spec name/signature. `calculate_from_result()` exists but spec requires `calculate_all(result: BacktestResult) -> dict` (`nba_model/backtest/metrics.py`).

### New Issues Found
1. **BacktestResult still lacks required `clv` property**
   - File: `nba_model/backtest/engine.py`
   - Phase 5 spec requires `BacktestResult.clv`, but only `avg_clv` exists (through metrics).

2. **Spread/total devigging still hard-coded to 0.5**
   - File: `nba_model/backtest/engine.py`
   - For spread/total bets, market probability is fixed at 0.5 regardless of odds input. This bypasses the chosen devigging method and may violate the requirement to devig market odds for all bet types.

### Test Results (Loop 1)
- `source .venv/bin/activate && pytest tests/ -v`: **FAILED** (Signal 6)
- `source .venv/bin/activate && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 KMP_DUPLICATE_LIB_OK=TRUE MKL_DYNAMIC=FALSE KMP_INIT_AT_FORK=FALSE KMP_BLOCKTIME=0 KMP_AFFINITY=disabled pytest tests/ -v`: **FAILED** (Signal 6)
- Coverage could not be verified (requirements: ≥75% overall, ≥80% unit per guidelines; `tests/CLAUDE.md` also mentions 90% for critical paths).

### CLAUDE.md Accuracy Check
- Root `CLAUDE.md` claims “75% minimum coverage (80% for new code)” but enforcement in `pyproject.toml` is `fail_under = 75` only.
- `tests/CLAUDE.md` states 90% coverage for critical betting/prediction paths; not enforced in tooling and unverified due to test crash.

## Overall Assessment (Loop 1)
**NOT READY.** Two prior issues remain unresolved (Shin devigging method, metrics API signature), plus new Phase 5 gaps (missing `clv` property on `BacktestResult`, spread/total devigging bypass). Tests continue to crash with Signal 6, blocking coverage verification.
