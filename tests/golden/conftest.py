"""Golden-value suite: absolute references, and the policy that keeps them honest.

WHY THIS SUITE EXISTS
---------------------
Every quantity crossing the GFN boundary is a DIFFERENCE or a ROUNDTRIP: lattice
energy is `crystal_pot - gas_pot`, latent transforms are `f-inverse(f(x))`,
standardization is `destd(std(x))`, and the MLIP suites assert `path A == path B`.
Each of those passes for an entire FAMILY of wrong absolute values. That is not a
gap in the tests; it is a property of their shape. A real instance: MACE lattice
energy carries a +11836.127 kJ/mol per-molecule offset that cancels in every
difference and was noticed downstream rather than by any test.

These tests pin ABSOLUTE values, so that family is no longer invisible.

REGENERATION POLICY
-------------------
`pytest tests/golden --regen-golden` is the ONLY way a refs file is written, and
**a regeneration run can never report success** -- it exits non-zero at teardown.
Otherwise the suite quietly re-baselines the very bug it exists to catch.

Only `value` is ever rewritten. `call`, `fixture`, `rtol`, `atol` and `note` are
preserved, so changing a tolerance is a hand edit that reads AS a tolerance change
in the diff. It can never hide inside a value regeneration.
"""

import json
from pathlib import Path

import pytest

REFS = Path(__file__).parent / 'refs'
_PENDING = {}


@pytest.fixture(autouse=True)
def _cpu_only():
    """This suite is CPU-ONLY, enforced per test rather than assumed.

    Two reasons. Reference values are float32 results whose low bits depend on
    reduction order, so a value pinned on CPU and checked on GPU is a different
    measurement wearing the same name. And the owner may be using the GPU for real
    work -- a test suite must never contend for it.

    FUNCTION-scoped deliberately. A session-scoped autouse fixture declared here
    still tears down at the end of the WHOLE session, so it would assert against
    allocations made by other suites -- which it did, failing an unrelated
    conformers test that legitimately uses the GPU.
    """
    import torch
    if not torch.cuda.is_available():
        yield
        return
    before = torch.cuda.memory_allocated()
    yield
    leaked = torch.cuda.memory_allocated() - before
    assert leaked == 0, (
        f'a golden test allocated {leaked} bytes on the GPU. This suite must run '
        'entirely on CPU: reference values are device-dependent in their low bits, '
        'and the GPU may be in use for real work.')


def pytest_addoption(parser):
    parser.addoption('--regen-golden', action='store_true', default=False,
                     help='Rewrite golden reference VALUES. The run will fail by design; '
                          'review `git diff tests/golden/refs/` before committing.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'golden: pins an absolute reference value')


@pytest.fixture(scope='session')
def regen(request) -> bool:
    return bool(request.config.getoption('--regen-golden'))


def _load(name: str) -> dict:
    path = REFS / f'{name}.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def golden(regen):
    """`golden(file, key, computed)` -> the reference to compare against.

    On a normal run it returns the stored value and never touches disk. Under
    --regen-golden it records `computed` for writing at teardown and returns it,
    so the assertions trivially pass and the run still fails at the end.
    """
    def _get(file: str, key: str, computed=None):
        data = _load(file)
        entry = data.get(key)
        if regen:
            if computed is None:
                pytest.fail(f'--regen-golden needs a computed value for {file}:{key}')
            _PENDING.setdefault(file, {})[key] = _to_jsonable(computed)
            return computed
        if entry is None:
            pytest.fail(
                f'no golden reference for {file}:{key}. Generate it deliberately with '
                '`pytest tests/golden --regen-golden` and review the diff -- never '
                'let a missing reference silently pass.')
        return entry['value']

    def _tol(file: str, key: str, default_rtol=1e-6, default_atol=1e-8):
        entry = _load(file).get(key) or {}
        return (float(entry.get('rtol', default_rtol)),
                float(entry.get('atol', default_atol)))

    _get.tol = _tol
    return _get


def _to_jsonable(v):
    try:
        import numpy as np
        import torch
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def pytest_sessionfinish(session, exitstatus):
    """Write pending refs, then FAIL the run. A regen is never a pass."""
    if not session.config.getoption('--regen-golden'):
        return
    changed = 0
    for file, values in _PENDING.items():
        path = REFS / f'{file}.json'
        data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        for key, new in values.items():
            entry = data.setdefault(key, {})
            if entry.get('value') != new:
                changed += 1
            entry['value'] = new                      # ONLY the value is rewritten
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    pytest.exit(
        f'GOLDEN REFS REGENERATED -- {changed} value(s) changed across '
        f'{len(_PENDING)} file(s). Review `git diff tests/golden/refs/` before '
        'committing. THIS RUN IS NOT A PASS.', returncode=2)


# --------------------------------------------------------------------------
# fixtures shared across the suite
# --------------------------------------------------------------------------

@pytest.fixture(scope='session')
def mini_csd():
    """The 100-crystal fixture set. 1.6 MB, loaded once per session."""
    import torch
    path = Path(__file__).resolve().parents[2] / 'mini_datasets' / 'mini_new_csd.pt'
    if not path.exists():
        pytest.skip(f'{path} not present')
    return list(torch.load(path, weights_only=False))


@pytest.fixture(scope='session')
def mini_by_id(mini_csd):
    """Keyed BY IDENTIFIER, never by position: a dataset reorder must not
    silently swap which crystal a golden value refers to."""
    out = {}
    for c in mini_csd:
        ident = c.identifier
        if isinstance(ident, (list, tuple)):
            ident = ident[0]
        out[str(ident).strip("[]'\" ")] = c
    return out
