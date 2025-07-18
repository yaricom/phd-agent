"""
Unit tests for file_utils module.

Tests all functions exposed by the file_utils module including format detection,
file writing, and error handling.
"""

import pytest
import os
from unittest.mock import patch
from datetime import datetime

from phd_agent.models import Essay, DocumentSource, DocumentType, EssayOutline
from phd_agent.file_utils import (
    write_essay_txt,
    write_essay_pdf,
    write_essay_docx,
    write_essay,
    get_supported_formats,
)


@pytest.fixture
def sample_essay():
    """Create a sample essay for testing."""
    sources = [
        DocumentSource(
            id="1",
            title="Test Source 1",
            content="This is test content 1",
            source_type=DocumentType.PDF,
            url="https://example.com/1",
        ),
        DocumentSource(
            id="2",
            title="Test Source 2",
            content="This is test content 2",
            source_type=DocumentType.WEB,
            url="https://example.com/2",
        ),
    ]

    outline = EssayOutline(
        title="Test Essay",
        introduction="Test introduction",
        main_points=["Point 1", "Point 2"],
        conclusion="Test conclusion",
        sources=["1", "2"],
    )

    return Essay(
        id="test-essay-1",
        title="Test Essay",
        content="This is the first paragraph.\n\nThis is the second paragraph.",
        word_count=15,
        sources=sources,
        outline=outline,
        created_at=datetime(2023, 1, 1, 12, 0, 0),
    )


def test_write_essay_txt_success(sample_essay, tmpdir):
    """Test successful TXT file writing."""
    output_path = os.path.join(tmpdir, "test_essay.txt")

    result = write_essay_txt(sample_essay, output_path)

    assert result is True
    assert os.path.exists(output_path)

    # Verify file content
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Title: Test Essay" in content
    assert "Word Count: 15" in content
    assert "Sources: 2" in content
    assert "This is the first paragraph." in content
    assert "This is the second paragraph." in content
    assert "1. Test Source 1 (pdf)" in content
    assert "2. Test Source 2 (web)" in content
    assert "URL: https://example.com/1" in content
    assert "URL: https://example.com/2" in content


def test_write_essay_txt_without_sources(tmpdir):
    """Test TXT file writing with essay that has no sources."""
    essay = Essay(
        id="test-essay-no-sources",
        title="Test Essay No Sources",
        content="Test content",
        word_count=2,
        sources=[],
        outline=EssayOutline(
            title="Test Essay No Sources",
            introduction="Intro",
            main_points=[],
            conclusion="Conclusion",
            sources=[],
        ),
        created_at=datetime(2023, 1, 1, 12, 0, 0),
    )

    output_path = os.path.join(tmpdir, "test_essay_no_sources.txt")

    result = write_essay_txt(essay, output_path)

    assert result is True
    assert os.path.exists(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Sources: 0" in content
    assert "SOURCES:" in content


def test_write_essay_txt_permission_error(sample_essay):
    """Test TXT file writing with permission error."""
    # Try to write to a directory that doesn't exist and can't be created
    output_path = "/nonexistent/directory/test.txt"

    result = write_essay_txt(sample_essay, output_path)

    assert result is False


def test_write_essay_pdf_success(sample_essay, tmpdir):
    """Test successful PDF file writing."""
    output_path = os.path.join(tmpdir, "test_essay.pdf")

    result = write_essay_pdf(sample_essay, output_path)

    assert result is True
    assert os.path.exists(output_path)


def test_write_essay_docx_success(sample_essay, tmpdir):
    """Test successful DOCX file writing."""
    output_path = os.path.join(tmpdir, "test_essay.docx")

    result = write_essay_docx(sample_essay, output_path)

    assert result is True
    assert os.path.exists(output_path)


def test_write_essay_auto_detect_txt(sample_essay, tmpdir):
    """Test automatic format detection for TXT files."""
    output_path = os.path.join(tmpdir, "test_essay.txt")

    result = write_essay(sample_essay, output_path)

    assert result is True
    assert os.path.exists(output_path)


def test_write_essay_auto_detect_pdf(sample_essay, tmpdir):
    """Test automatic format detection for PDF files."""
    with patch("phd_agent.file_utils.write_essay_pdf", return_value=True) as mock_pdf:
        output_path = os.path.join(tmpdir, "test_essay.pdf")

        result = write_essay(sample_essay, output_path)

        assert result is True
        mock_pdf.assert_called_once_with(sample_essay, output_path)


def test_write_essay_auto_detect_docx(sample_essay, tmpdir):
    """Test automatic format detection for DOCX files."""
    with patch("phd_agent.file_utils.write_essay_docx", return_value=True) as mock_docx:
        output_path = os.path.join(tmpdir, "test_essay.docx")

        result = write_essay(sample_essay, output_path)

        assert result is True
        mock_docx.assert_called_once_with(sample_essay, output_path)


def test_write_essay_auto_detect_unknown_extension(sample_essay, tmpdir):
    """Test automatic format detection for unknown extensions."""
    output_path = os.path.join(tmpdir, "test_essay.unknown")

    result = write_essay(sample_essay, output_path)

    assert result is True
    # Should create .txt file
    expected_path = output_path + ".txt"
    assert os.path.exists(expected_path)


def test_write_essay_auto_detect_no_extension(sample_essay, tmpdir):
    """Test automatic format detection for files without extension."""
    output_path = os.path.join(tmpdir, "test_essay")

    result = write_essay(sample_essay, output_path)

    assert result is True
    # Should create .txt file
    expected_path = output_path + ".txt"
    assert os.path.exists(expected_path)


def test_write_essay_explicit_format_txt(sample_essay, tmpdir):
    """Test explicit format specification for TXT."""
    output_path = os.path.join(tmpdir, "test_essay.custom")

    result = write_essay(sample_essay, output_path, output_format="txt")

    assert result is True
    assert os.path.exists(output_path)


def test_write_essay_explicit_format_pdf(sample_essay, tmpdir):
    """Test explicit format specification for PDF."""
    with patch("phd_agent.file_utils.write_essay_pdf", return_value=True) as mock_pdf:
        output_path = os.path.join(tmpdir, "test_essay.custom")

        result = write_essay(sample_essay, output_path, output_format="pdf")

        assert result is True
        mock_pdf.assert_called_once_with(sample_essay, output_path)


def test_write_essay_explicit_format_docx(sample_essay, tmpdir):
    """Test explicit format specification for DOCX."""
    with patch("phd_agent.file_utils.write_essay_docx", return_value=True) as mock_docx:
        output_path = os.path.join(tmpdir, "test_essay.custom")

        result = write_essay(sample_essay, output_path, output_format="docx")

        assert result is True
        mock_docx.assert_called_once_with(sample_essay, output_path)


def test_write_essay_unsupported_format(sample_essay, tmpdir):
    """Test writing with unsupported format."""
    output_path = os.path.join(tmpdir, "test_essay.xyz")

    result = write_essay(sample_essay, output_path, output_format="xyz")

    assert result is False


def test_write_essay_creates_directory(sample_essay, tmpdir):
    """Test that write_essay creates parent directories if they don't exist."""
    nested_dir = os.path.join(tmpdir, "nested", "subdirectory")
    output_path = os.path.join(nested_dir, "test_essay.txt")

    result = write_essay(sample_essay, output_path)

    assert result is True
    assert os.path.exists(output_path)
    assert os.path.exists(nested_dir)


def test_get_supported_formats_all_available():
    """Test get_supported_formats when all dependencies are available."""
    formats = get_supported_formats()

    assert "txt" in formats
    assert "pdf" in formats
    assert "docx" in formats
    assert len(formats) == 3


def test_write_essay_txt_unicode_content(tmpdir):
    """Test TXT file writing with Unicode content."""
    essay = Essay(
        id="test-essay-unicode",
        title="Test Essay with Unicode: 测试",
        content="This is content with Unicode: αβγδε",
        word_count=8,
        sources=[],
        outline=EssayOutline(
            title="Test Essay with Unicode: 测试",
            introduction="Intro",
            main_points=[],
            conclusion="Conclusion",
            sources=[],
        ),
        created_at=datetime(2023, 1, 1, 12, 0, 0),
    )

    output_path = os.path.join(tmpdir, "test_unicode.txt")

    result = write_essay_txt(essay, output_path)

    assert result is True
    assert os.path.exists(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Test Essay with Unicode: 测试" in content
    assert "This is content with Unicode: αβγδε" in content


def test_write_essay_txt_empty_content(tmpdir):
    """Test TXT file writing with empty content."""
    essay = Essay(
        id="test-essay-empty",
        title="Empty Essay",
        content="",
        word_count=0,
        sources=[],
        outline=EssayOutline(
            title="Empty Essay",
            introduction="",
            main_points=[],
            conclusion="",
            sources=[],
        ),
        created_at=datetime(2023, 1, 1, 12, 0, 0),
    )

    output_path = os.path.join(tmpdir, "test_empty.txt")

    result = write_essay_txt(essay, output_path)

    assert result is True
    assert os.path.exists(output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Title: Empty Essay" in content
    assert "Word Count: 0" in content
    assert "SOURCES:" in content
