#!/usr/bin/env python3
"""
System Test Script

This script tests the basic functionality of the PhD Agent multi-agent research system
without requiring external services like Milvus or OpenAI API.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import phd_agent.config  # noqa: F401

        print("✅ config module imported")
    except ImportError as e:
        print(f"⚠️  config import failed (expected if package not installed): {e}")
        return False
    except Exception as e:
        print(f"❌ config import failed: {e}")
        return False

    try:
        from phd_agent.models import (
            ResearchTask,  # noqa: F401
            DocumentSource,  # noqa: F401
            DocumentType,  # noqa: F401
        )

        print("✅ models module imported")
    except ImportError as e:
        print(f"⚠️  models import failed (expected if package not installed): {e}")
        return False
    except Exception as e:
        print(f"❌ models import failed: {e}")
        return False

    try:
        from phd_agent.agents.supervisor_agent import SupervisorAgent  # noqa: F401

        print("✅ supervisor_agent module imported")
    except ImportError as e:
        print(
            f"⚠️  supervisor_agent import failed (expected if package not installed): {e}"
        )
        return False
    except Exception as e:
        print(f"❌ supervisor_agent import failed: {e}")
        return False

    try:
        from phd_agent.agents.pdf_agent import PDFAgent  # noqa: F401

        print("✅ pdf_agent module imported")
    except ImportError as e:
        print(f"⚠️  pdf_agent import failed (expected if package not installed): {e}")
        return False
    except Exception as e:
        print(f"❌ pdf_agent import failed: {e}")
        return False

    try:
        from phd_agent.agents.web_search_agent import WebSearchAgent  # noqa: F401

        print("✅ web_search_agent module imported")
    except ImportError as e:
        print(
            f"⚠️  web_search_agent import failed (expected if package not installed): {e}"
        )
        return False
    except Exception as e:
        print(f"❌ web_search_agent import failed: {e}")
        return False

    try:
        from phd_agent.agents.analyst_agent import AnalystAgent  # noqa: F401

        print("✅ analyst_agent module imported")
    except ImportError as e:
        print(
            f"⚠️  analyst_agent import failed (expected if package not installed): {e}"
        )
        return False
    except Exception as e:
        print(f"❌ analyst_agent import failed: {e}")
        return False

    try:
        from phd_agent.agents.essay_writer_agent import EssayWriterAgent  # noqa: F401

        print("✅ essay_writer_agent module imported")
    except ImportError as e:
        print(
            f"⚠️  essay_writer_agent import failed (expected if package not installed): {e}"
        )
        return False
    except Exception as e:
        print(f"❌ essay_writer_agent import failed: {e}")
        return False

    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        from phd_agent.config import config

        # Test basic config attributes
        assert hasattr(config, "OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
        assert hasattr(config, "OPENAI_MODEL"), "Missing OPENAI_MODEL"
        assert hasattr(config, "MILVUS_HOST"), "Missing MILVUS_HOST"
        assert hasattr(config, "MILVUS_PORT"), "Missing MILVUS_PORT"

        print("✅ Configuration loaded successfully")
        print(f"   OpenAI Model: {config.OPENAI_MODEL}")
        print(f"   Milvus Host: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        print(f"   API Key configured: {'Yes' if config.OPENAI_API_KEY else 'No'}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_models():
    """Test data models."""
    print("\nTesting data models...")

    try:
        from phd_agent.models import (
            ResearchTask,
            DocumentSource,
            DocumentType,
            AgentState,
        )

        # Test ResearchTask creation
        task = ResearchTask(
            id="test-123",
            topic="Test Topic",
            requirements="Test requirements",
            max_sources=5,
            essay_length="medium",
        )
        assert task.topic == "Test Topic"
        assert task.max_sources == 5

        # Test DocumentSource creation
        doc = DocumentSource(
            id="doc-123",
            title="Test Document",
            content="Test content",
            source_type=DocumentType.PDF,
            file_path="/test/path.pdf",
        )
        assert doc.title == "Test Document"
        assert doc.source_type == DocumentType.PDF

        # Test AgentState creation
        state = AgentState(task=task)
        assert state.task == task
        assert len(state.documents) == 0

        print("✅ Data models working correctly")
        return True

    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False


def test_agent_initialization():
    """Test agent initialization (without external services)."""
    print("\nTesting agent initialization...")

    try:
        # Test supervisor agent initialization
        from phd_agent.agents.supervisor_agent import SupervisorAgent

        SupervisorAgent()
        print("✅ Supervisor agent initialized")

        # Test PDF agent initialization
        from phd_agent.agents.pdf_agent import PDFAgent

        PDFAgent()
        print("✅ PDF agent initialized")

        # Test web search agent initialization
        from phd_agent.agents.web_search_agent import WebSearchAgent

        WebSearchAgent()
        print("✅ Web search agent initialized")

        # Test analyst agent initialization
        from phd_agent.agents.analyst_agent import AnalystAgent

        AnalystAgent()
        print("✅ Analyst agent initialized")

        # Test essay writer agent initialization
        from phd_agent.agents.essay_writer_agent import EssayWriterAgent

        EssayWriterAgent()
        print("✅ Essay writer agent initialized")

        return True

    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "pyproject.toml",
        "dev_setup.py",
        "README.md",
        "src/phd_agent/__init__.py",
        "src/phd_agent/config.py",
        "src/phd_agent/models.py",
        "src/phd_agent/vector_store.py",
        "src/phd_agent/vector_store_mock.py",
        "src/phd_agent/api.py",
        "src/phd_agent/agents/__init__.py",
        "src/phd_agent/agents/supervisor_agent.py",
        "src/phd_agent/agents/pdf_agent.py",
        "src/phd_agent/agents/web_search_agent.py",
        "src/phd_agent/agents/analyst_agent.py",
        "src/phd_agent/agents/essay_writer_agent.py",
        "examples/basic_research.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PhD Agent - System Test")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Models", test_models),
        ("Agent Initialization", test_agent_initialization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Start Milvus if you want to use vector storage")
        print("3. Run: python examples/basic_research.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed and files are present.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
