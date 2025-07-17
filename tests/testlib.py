import sys
from contextlib import contextmanager
from unittest import mock


@contextmanager
def mock_system_import(module_name: str):
    """Mock a system import."""
    mocked_module = mock.MagicMock()

    with mock.patch.dict(sys.modules, {module_name: mocked_module}):
        yield mocked_module
