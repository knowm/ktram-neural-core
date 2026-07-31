import pytest

from _congruence import write_report


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Write tests/congruence_report.md from whatever the battery recorded this run."""
    write_report()
