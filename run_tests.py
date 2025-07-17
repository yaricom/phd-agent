#!/usr/bin/env python3
"""
Simple test runner for file_utils module.

This script runs the unit tests for the file_utils module without requiring
pytest to be installed globally.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def run_file_utils_tests():
    """Run the file_utils tests."""
    try:
        import pytest
        from tests.test_file_utils import TestFileUtils
        
        print("Running file_utils tests...")
        
        # Create test instance
        test_instance = TestFileUtils()
        
        # Run basic tests that don't require external dependencies
        print("\n1. Testing TXT file writing...")
        test_instance.test_write_essay_txt_success(test_instance.sample_essay(), test_instance.temp_dir())
        test_instance.test_write_essay_txt_without_sources(test_instance.temp_dir())
        test_instance.test_write_essay_txt_unicode_content(test_instance.temp_dir())
        test_instance.test_write_essay_txt_empty_content(test_instance.temp_dir())
        
        print("\n2. Testing format detection...")
        test_instance.test_write_essay_auto_detect_txt(test_instance.sample_essay(), test_instance.temp_dir())
        test_instance.test_write_essay_auto_detect_unknown_extension(test_instance.sample_essay(), test_instance.temp_dir())
        test_instance.test_write_essay_auto_detect_no_extension(test_instance.sample_essay(), test_instance.temp_dir())
        test_instance.test_write_essay_creates_directory(test_instance.sample_essay(), test_instance.temp_dir())
        
        print("\n3. Testing supported formats...")
        test_instance.test_get_supported_formats_all_available()
        test_instance.test_get_supported_formats_no_optional_deps()
        test_instance.test_get_supported_formats_mixed_availability()
        
        print("\n4. Testing explicit format specification...")
        test_instance.test_write_essay_explicit_format_txt(test_instance.sample_essay(), test_instance.temp_dir())
        test_instance.test_write_essay_unsupported_format(test_instance.sample_essay(), test_instance.temp_dir())
        
        print("\n✅ All basic tests passed!")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install pytest: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_file_utils_tests()
    sys.exit(0 if success else 1) 