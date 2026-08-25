"""CIF -> MolCrystalData -> rebuild round trip, against the CCDC reader.

This is the incumbent path's own correctness check, run as a test rather than
as a counter printed inside a chunk job.  It is the reference the CCDC-free
reader in `mxtaltools/dataset_utils/construction/cif/` must match; see
`docs/design/cif_reader_design.md` (Stage 0 / Stage 4).

Requires a CCDC licence and a local CIF corpus.  Both are skipped cleanly when
absent -- but note that a skip here is NOT evidence the path works.  The
licence-free half of this contract lives in `tests/cif/`, which tests the
native reader and does not skip.

Replaces a 25-line stub that ended mid-statement and had never run:
its corpus path was missing a drive colon and `cif_path` was undefined.
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ccdc

CORPUS_CANDIDATES = (
    Path(r"D:\crystal_datasets\CSD_dump"),
    Path(r"D:\crystal_datasets\acridine"),
)
N_CIFS = 20


def _corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.is_dir():
            return p
    pytest.skip(f"no local CIF corpus; looked in {[str(p) for p in CORPUS_CANDIDATES]}")


@pytest.fixture(scope="module")
def chunk() -> list:
    pytest.importorskip("ccdc", reason="CCDC licence required")
    root = _corpus()
    cifs = sorted(str(root / f) for f in os.listdir(root) if f.endswith(".cif"))
    if len(cifs) < N_CIFS:
        pytest.skip(f"corpus {root} holds {len(cifs)} cifs, need {N_CIFS}")
    return cifs[:N_CIFS]


def test_cif_roundtrip(chunk):
    """Every crystal that survives filtering must rebuild to its deposited unit cell.

    `process_chunk` internally asserts the rebuild via `crystal_rebuild_checks`
    and drops anything that fails, so a non-empty return means every returned
    crystal passed.  We therefore assert on the yield, which is the quantity a
    silent regression in the builder would move.
    """
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import process_chunk

    data_list = process_chunk(
        chunk,
        chunk_ind=0,
        use_filenames_for_identifiers=False,
        protonation_state="deprotonated",
        max_z_prime=1,
    )

    assert data_list is not None, "process_chunk returned None"
    # Not every CIF survives filtering (polymers, metals, Z' > 1, size caps).
    # A yield of zero means the path is broken, not that the corpus is hard.
    assert len(data_list) > 0, (
        f"0 of {len(chunk)} cifs survived the full read->build->rebuild path; "
        "the incumbent CCDC path is broken"
    )

    for crystal in data_list:
        assert crystal.num_nodes > 0
        assert crystal.sg_ind is not None
        assert int(crystal.sym_mult) >= 1
        # NOTE: `unit_cell_pos` is deliberately NOT populated on the emitted object --
        # it is rebuilt on demand via mol2ucell()/build_unit_cell().  Asserting it
        # here would be asserting a field that is not part of this contract.
        assert crystal.z_prime is not None


def test_zprime_matches_symmetry_multiplicity(chunk):
    """Z' * sym_mult * (atoms per aunit) == atoms in the built unit cell.

    This is the arithmetic the native reader derives Z' from (design M1), stated
    against the incumbent so that both readers are held to the same identity.
    """
    from mxtaltools.dataset_utils.construction.featurize_cif_chunks import process_chunk

    data_list = process_chunk(
        chunk,
        chunk_ind=0,
        use_filenames_for_identifiers=False,
        protonation_state="deprotonated",
        max_z_prime=1,
    )
    if not data_list:
        pytest.skip("no crystals survived filtering on this corpus slice")

    from mxtaltools.dataset_utils.utils import collate_data_list

    batch = collate_data_list(data_list)
    batch.mol2ucell()          # unit_cell_pos is built on demand, not stored
    n_aunit = int(batch.num_nodes)
    n_ucell = int(batch.unit_cell_pos.shape[0])
    expected = int(sum(int(c.num_nodes) * int(c.sym_mult) for c in data_list))
    assert n_ucell == expected, (
        f"built unit cells hold {n_ucell} atoms across {len(data_list)} crystals, "
        f"expected sum(n_aunit * sym_mult) = {expected} (aunit total {n_aunit})"
    )
