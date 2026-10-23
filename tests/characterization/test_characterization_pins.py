"""Stage 0: today's MXtalTools crystal building and analysis, pinned bit for bit on CPU.

One test per (case, pin group): the group is recomputed by ``charlib.compute_group`` (the code the
goldens were written with) and every value must equal its golden exactly -- same dtype, shape and
bits (NaN == NaN), or the same exception type. Cases and groups: README.md and
``data/goldens_manifest.json``.

A failure here means behaviour changed. That is the point of the suite: later stages change
behaviour ON PURPOSE, and each such change must show up here, be explained, and be re-pinned in
the same change (``build_goldens.py --regenerate``; README.md). A pin that moves without an
intended change is a regression.

Each golden file carries the environment it was written under (``__environment__``: torch, numpy,
torch_geometric, torch_scatter, torch_cluster, MXT_* switches; ``__group_environments__``: the
packages a group kind depends on beyond those, e.g. fairchem / mace / ase for ``mlip``). A test
skips, with the differences named, when its own file's record differs from this environment: a
bit-for-bit pin is then meaningless rather than wrong.
"""
import pytest
import torch

import charlib

PARAMS = [(case, group) for case in charlib.CASES for group in charlib.case_groups(case)]

_GOLDENS = {}


def _golden_file(case):
    if case not in _GOLDENS:
        path = charlib.DATA / f'golden_{case}.pt'
        if not path.exists():
            pytest.fail(f'{path} is missing; build the goldens (README.md)')
        _GOLDENS[case] = torch.load(path, weights_only=True)
    return _GOLDENS[case]


def _golden(case):
    return {k: v for k, v in _golden_file(case).items() if not k.startswith('__')}


def _skip_if_environment_differs(case, group):
    saved = _golden_file(case)
    diff = charlib.environment_mismatch(saved.get('__environment__', {}))
    if diff:
        pytest.skip(f'golden_{case}.pt was written under a different environment; bit-for-bit pins do not '
                    f'transfer. Regenerate deliberately (README.md) if this environment is the new reference. '
                    + '; '.join(f'{k}: golden {a!r}, now {b!r}' for k, a, b in diff))
    kind = charlib.group_kind(group)
    if kind in charlib.GROUP_PACKAGES:
        recorded = saved.get('__group_environments__', {}).get(kind)
        now = charlib.group_environment(kind)
        if recorded != now:
            pytest.skip(f'{kind} pins of golden_{case}.pt were written under {recorded}, this environment has {now}')


@pytest.mark.parametrize('case,group', PARAMS, ids=[f'{c}:{g}' for c, g in PARAMS])
def test_pinned(case, group, golden_manifest):
    _skip_if_environment_differs(case, group)
    golden = {k: v for k, v in _golden(case).items() if k.startswith(group + '/')}
    assert golden, f'no golden pins for {case} {group}: the goldens are older than the pin groups'
    now = charlib.compute_group(case, group)
    problems = []
    for k in sorted(set(golden) - set(now)):
        problems.append(f'{k}: pinned but no longer produced (golden {charlib.describe(golden[k])})')
    for k in sorted(set(now) - set(golden)):
        problems.append(f'{k}: produced but not pinned (now {charlib.describe(charlib.encode(now[k]))})')
    for k in golden:
        if k in now:
            ok, detail = charlib.compare(golden[k], now[k])
            if not ok:
                problems.append(f'{k}: {detail}')
    assert not problems, f'{case} {group}: {len(problems)} pin(s) moved\n  ' + '\n  '.join(problems)


def test_every_case_is_pinned(golden_manifest):
    """The golden files cover exactly the cases and groups the suite defines, and no group they no longer define."""
    for case in charlib.CASES:
        keys = set(_golden(case))
        groups = charlib.case_groups(case)
        for group in groups:
            assert any(k.startswith(group + '/') for k in keys), f'{case}: group {group} has no pins'
        stray = sorted(k for k in keys if not any(k.startswith(g + '/') for g in groups))
        assert not stray, f'{case}: pins of no defined group: {stray[:5]}'
