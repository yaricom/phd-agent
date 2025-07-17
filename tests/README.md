# Tests

This directory contains unit tests for the PhD Agent system.

## Running Tests

### Using pytest (recommended)

Install the development dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
python -m pytest tests
```

Run a specific test file:
```bash
python -m pytest tests/unit/test_file_utils.py
```

Run with coverage:
```bash
python -m pytest --cov=phd_agent
```


## Test Structure

- `test_file_utils.py` - Tests for the file_utils module
  - Tests all format writing functions (TXT, PDF, DOCX)
  - Tests format detection and error handling
  - Tests dependency availability detection
  - Uses mocking for external dependencies

## Test Coverage

The tests cover:
- ✅ TXT file writing (success and error cases)
- ✅ PDF file writing (with mocked reportlab)
- ✅ DOCX file writing (with mocked python-docx)
- ✅ Format auto-detection
- ✅ Directory creation
- ✅ Unicode content handling
- ✅ Empty content handling
- ✅ Dependency availability detection
- ✅ Error handling for missing dependencies 