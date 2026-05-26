# Running Scripts Guide

Note: backend training/data-generation scripts have been removed. Training and
dataset generation now belong to the standalone `training_system` and
`preprocessor` services.

## Overview

This guide explains how to run scripts in the NeuroLab project correctly, avoiding import errors.

## The Import Issue

Python scripts inside the `src/` directory need to import modules using the `src.` prefix. However, when running scripts directly, Python doesn't automatically add the project root to the path.

## Solutions

### Solution 1: Use Wrapper Scripts (Recommended)

We've created wrapper scripts in the `scripts/` directory that handle the path setup automatically:

```bash
# Test voice API
python scripts/test_voice.py
```

### Solution 2: Run from Project Root

Always run scripts from the project root directory:

```bash
# ✓ Correct - from project root
python src/scripts/generation/generate_test_audio.py

# ✗ Wrong - from inside nested src/ directories without project root context
cd src/scripts/generation
python generate_test_audio.py  # This may fail!
```

### Solution 3: Set PYTHONPATH

Set the PYTHONPATH environment variable:

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "C:\Users\pc\Documents\Neurolab\neurolab_model"
python src/scripts/generation/generate_test_audio.py
```

**Windows (CMD):**
```cmd
set PYTHONPATH=C:\Users\pc\Documents\Neurolab\neurolab_model
python src/scripts/generation/generate_test_audio.py
```

**Linux/Mac:**
```bash
export PYTHONPATH=/path/to/neurolab_model
python src/scripts/generation/generate_test_audio.py
```

## Available Wrapper Scripts

### 1. Test Voice API
```bash
python scripts/test_voice.py
```
- Tests voice processing endpoints
- Requires: API server running (`uvicorn main:app`)
- Tests: health, emotions, audio analysis

## Direct Script Execution

If you prefer to run scripts directly from `src/`, they now include automatic path setup:

```python
# This is added to all scripts in src/scripts/
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

So you can run supported source-side scripts directly when needed.

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup directories
python setup_directories.py
```

### Test Data Generation
```bash
# Generate test audio
python src/scripts/generation/generate_test_audio.py
```

### Testing
```bash
# Test voice API
python scripts/test_voice.py

# Run all tests
python -m pytest src/tests/

# Run specific test
python -m pytest src/tests/test_voice_api.py
```

### API Server
```bash
# Start server
uvicorn main:app --reload

# Start on specific port
uvicorn main:app --port 8080

# Start with host binding
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'src'

**Cause:** Running script from wrong directory or PYTHONPATH not set

**Solutions:**
1. Run supported scripts from the project root
2. Set PYTHONPATH (see Solution 3 above)
3. Prefer documented backend scripts only; training scripts were removed

### Error: No such file or directory

**Cause:** Output directories don't exist

**Solution:**
```bash
python setup_directories.py
```

### Error: Model file not found

**Cause:** backend inference models have not been produced/promoted yet

**Solution:** train via `training_system`, then promote/sync via `model_platform`

### Error: Connection refused (testing voice API)

**Cause:** API server not running

**Solution:**
```bash
# Start server in another terminal
uvicorn main:app --reload

# Then run tests
python scripts/test_voice.py
```

## Project Structure

```
neurolab_model/
├── scripts/              # Wrapper scripts (use these!)
│   └── test_voice.py
├── src/
│   ├── scripts/         # Actual implementation
│   │   └── generation/
│   ├── tests/           # Test files
│   └── ...
├── main.py              # API entry point
└── setup_directories.py # Directory setup
```

## Best Practices

1. **Always use wrapper scripts** when possible
2. **Run from project root** if running directly
3. **Check current directory** before running scripts:
   ```bash
   pwd  # Linux/Mac
   cd   # Windows
   ```
4. **Use virtual environment**:
   ```bash
   # Activate venv first
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

## IDE Configuration

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}"
    }
}
```

### PyCharm
1. Right-click project root
2. Mark Directory as → Sources Root

## Summary

✓ **Use wrapper scripts** in `scripts/` directory (easiest)
✓ **Run from project root** when using direct paths
✓ **Set PYTHONPATH** for advanced usage
✓ **Check documentation** when in doubt

For more information, see:
- [QUICK_START.md](QUICK_START.md) - Quick reference
- [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md) - Import details
- [README.md](README.md) - Main documentation
