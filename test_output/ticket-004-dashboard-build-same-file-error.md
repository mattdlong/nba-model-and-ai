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
