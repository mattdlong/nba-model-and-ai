Steps to reproduce: 1. Install dependencies with pip install -e ".[dev]" 2. Check if torch-geometric installs correctly

Expected behaviour: All dependencies including torch-geometric should install successfully

Actual behaviour: Installation may fail for torch-geometric on Apple Silicon (M1/M2) due to binary wheels not being available

Why it is incorrect: Project should document platform-specific installation issues and provide alternatives for torch-geometric on macOS

---

## Fix Description

**Status: RESOLVED**

**Fix Applied:**
Added comprehensive documentation for torch-geometric installation on Apple Silicon (M1/M2/M3 Macs) in two locations:

### 1. USAGE.md (User Documentation)
Added a new "Platform-Specific Notes" section under "Getting Started" with:
- **Option 1**: Install PyTorch first, then torch-geometric with specific wheel index
- **Option 2**: Use Conda (recommended for Apple Silicon) - provides better ARM64 support
- **Option 3**: Skip torch-geometric (limited functionality) - system still works without GNN

### 2. CLAUDE.md (Developer Documentation)
Added a new "torch-geometric Installation on Apple Silicon" troubleshooting section with:
- Wheel index installation command
- Conda-based installation (recommended)
- Fallback option when installation fails

**Files Modified:**
- `USAGE.md`: Added "Platform-Specific Notes" section with Apple Silicon instructions
- `CLAUDE.md`: Added "torch-geometric Installation on Apple Silicon" troubleshooting section

**Verification:**
Documentation now clearly explains:
1. Why torch-geometric may fail on Apple Silicon
2. Three different approaches to resolve the issue
3. That the system works with reduced functionality if installation fails
