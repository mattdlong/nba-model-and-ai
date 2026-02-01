# Ticket: AttributeError in data status command - Missing 'data_dir' attribute

**Date:** 2026-02-01
**Reporter:** TESTER (opencode-claude-split)
**Severity:** High
**Component:** CLI - Data Commands

## Steps to Reproduce

1. Activate virtual environment: `source .venv/bin/activate`
2. Run: `python -m nba_model.cli data status`

## Expected Behaviour

The command should display database statistics including:
- Entity counts (Games, Players, Plays, Shots, Stints)
- Date ranges for data
- Checkpoint status information

## Actual Behaviour

The command shows the database statistics table but then crashes with an AttributeError:

```
AttributeError: 'Settings' object has no attribute 'data_dir'
```

Full traceback:
```
/Users/mdl/Documents/code/nba-model-and-ai/nba_model/cli.py:297 in data_status
    checkpoint_mgr = CheckpointManager()

/Users/mdl/Documents/code/nba-model-and-ai/nba_model/data/checkpoint.py:107 in __init__
    storage_path = settings.data_dir / "checkpoints"
```

## Why it is incorrect

The code attempts to access `settings.data_dir` but the `Settings` class only has `db_path`, `model_dir`, and `log_dir` attributes. There is no `data_dir` configuration option defined in the settings.

This causes the CLI to crash when trying to display checkpoint status after successfully showing the database statistics.

## Environment

- OS: macOS
- Python: 3.14.2
- Working directory: /Users/mdl/Documents/code/nba-model-and-ai
