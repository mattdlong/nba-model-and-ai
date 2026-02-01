# Ticket: Dashboard build fails - Same file copy error

**Date:** 2026-02-01
**Reporter:** TESTER (opencode-claude-split)
**Severity:** High
**Component:** Dashboard Generation

## Steps to Reproduce

1. Activate virtual environment: `source .venv/bin/activate`
2. Run: `python -m nba_model.cli dashboard build`

## Expected Behaviour

The command should generate a static dashboard site in the docs/ directory with HTML/JSON files for viewing predictions and performance metrics.

## Actual Behaviour

The command fails with an error:

```
Error building dashboard: PosixPath('docs/assets/charts.js') and 
PosixPath('docs/assets/charts.js') are the same file
```

## Why it is incorrect

The dashboard builder is attempting to copy a file to the same destination path. This suggests the copy logic is not checking if the source and destination are the same file before attempting the copy operation. This is likely happening when:
1. The source and destination directories are the same
2. The relative path resolution results in the same absolute path
3. The copy operation is being performed on files that are already in place

## Impact

Users cannot generate the dashboard site to view predictions in a web browser, which is a key output feature described in USAGE.md.

## Environment

- OS: macOS
- Python: 3.14.2
- Output directory: docs (default)
- Templates directory: templates (default)

---

## Resolution

**Status:** RESOLVED
**Fixed by:** DEVELOPER (claude-split)
**Fix Date:** 2026-02-01

### Fix Description

Updated the `_copy_static_assets` method in `nba_model/output/dashboard.py` to check if the source and destination directories/files are the same before attempting to copy. When the output directory is `docs` (the default), the assets source is also `docs/assets`, causing `shutil.copy2()` to fail when trying to copy a file to itself.

### Code Changes

Modified `_copy_static_assets()` in `nba_model/output/dashboard.py`:
```python
if source_dir:
    # Skip if source and destination are the same directory
    if source_dir.resolve() == dest_dir.resolve():
        logger.debug("Source and destination are the same, skipping asset copy")
        return 0

    for asset_file in source_dir.iterdir():
        if asset_file.is_file():
            dest_file = dest_dir / asset_file.name
            # Skip if source and destination files are the same
            if asset_file.resolve() == dest_file.resolve():
                logger.debug("Skipping asset (same file): {}", asset_file.name)
                continue
            shutil.copy2(asset_file, dest_file)
            assets_copied += 1
            logger.debug("Copied asset: {}", asset_file.name)
```

### Testing Evidence

**Steps followed:**
1. Activated virtual environment: `source .venv/bin/activate`
2. Ran: `python -m nba_model.cli dashboard build`

**Outcome:**
The command now executes successfully without the "same file" error:
```
╭────────────────────────────── Dashboard Build ───────────────────────────────╮
│ Building Dashboard                                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
Dashboard built successfully!
Created 7 files in 'docs/'

To view locally, open: docs/index.html
```

**Confirmation:** Expected behaviour now matches actual behaviour.
