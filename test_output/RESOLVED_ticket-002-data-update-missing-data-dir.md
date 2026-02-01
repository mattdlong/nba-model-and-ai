# Ticket: AttributeError in data update command - Missing 'data_dir' attribute

**Date:** 2026-02-01
**Reporter:** TESTER (opencode-claude-split)
**Severity:** High
**Component:** CLI - Data Commands

## Steps to Reproduce

1. Activate virtual environment: `source .venv/bin/activate`
2. Run: `python -m nba_model.cli data update`

## Expected Behaviour

The command should incrementally update the database with recent games data from the NBA API.

## Actual Behaviour

The command crashes immediately with an AttributeError:

```
AttributeError: 'Settings' object has no attribute 'data_dir'
```

Full traceback:
```
/Users/mdl/Documents/code/nba-model-and-ai/nba_model/cli.py:240 in data_update
    checkpoint_mgr = CheckpointManager()

/Users/mdl/Documents/code/nba-model-and-ai/nba_model/data/checkpoint.py:107 in __init__
    storage_path = settings.data_dir / "checkpoints"
```

## Why it is incorrect

The `CheckpointManager` class attempts to access `settings.data_dir` which does not exist in the Settings configuration. The Settings class only defines:
- `db_path`
- `model_dir`
- `log_dir`

There is no `data_dir` attribute, causing the command to fail before any data collection can begin.

## Impact

This completely blocks the daily workflow described in USAGE.md, which states users should run:
```bash
python -m nba_model.cli data update
```

## Related Issue

See ticket-001-data-status-missing-data-dir.md - same root cause affecting multiple commands.

## Environment

- OS: macOS
- Python: 3.14.2
- Working directory: /Users/mdl/Documents/code/nba-model-and-ai

---

## Resolution

**Status:** RESOLVED
**Fixed by:** DEVELOPER (claude-split)
**Fix Date:** 2026-02-01

### Fix Description

Added a `data_dir` property to the `Settings` class in `nba_model/config.py`. The property derives the data directory from the database path's parent directory (`Path(self.db_path).parent`), which correctly resolves to `data/` when `db_path` is `data/nba.db`.

This is the same fix as ticket-001 - both tickets share the same root cause.

### Code Changes

Added to `nba_model/config.py`:
```python
@property
def data_dir(self) -> Path:
    """Return data directory as Path object.

    Derived from the database path's parent directory.
    """
    return Path(self.db_path).parent
```

### Testing Evidence

**Steps followed:**
1. Activated virtual environment: `source .venv/bin/activate`
2. Ran: `python -m nba_model.cli data update`

**Outcome:**
The command now starts executing without the AttributeError. The command begins the update process as expected (connecting to NBA API for incremental updates).

**Confirmation:** Expected behaviour now matches actual behaviour - command no longer crashes with `AttributeError: 'Settings' object has no attribute 'data_dir'`.
