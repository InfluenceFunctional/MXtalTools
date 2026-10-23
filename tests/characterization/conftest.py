"""Fixtures of the stage-0 characterization suite (see README.md).

Every test module here runs under ``charlib.deterministic()`` (one thread, deterministic
algorithms, float32 default dtype), entered per module and restored afterwards, so the pins do not
depend on the thread count or on a default dtype another module set at import, and nothing leaks
out of this directory.
"""
import json

import pytest

import charlib


@pytest.fixture(autouse=True, scope='module')
def _deterministic_cpu():
    with charlib.deterministic():
        yield


@pytest.fixture(scope='session')
def golden_manifest():
    """The goldens' manifest (fails when missing). Whether a pin applies in this environment is judged per
    golden file, from the environment that file records (test_characterization_pins.py), so files written
    at different times are each judged by their own record."""
    if not charlib.GOLDEN_MANIFEST.exists():
        pytest.fail(f'{charlib.GOLDEN_MANIFEST} is missing; build the goldens (README.md)')
    with open(charlib.GOLDEN_MANIFEST) as f:
        return json.load(f)
