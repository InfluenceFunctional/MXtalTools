"""Nonstandard settings re-expressed in the standard setting (``crystal_fd.settings``, owner decision D1) and the
mirrored reference molecule for off-sheet Sohncke rows (``io_mxt``, owner decision D2), 2026-10-16.

1. The standard setting of every one of the 230 groups is its own setting: ``find_setting`` gives (I, 0).
2. Every setting class present in D:/crystal_datasets/new_prot_csd.pt (``CLASSES``: the stored operator sets of the
   rows whose operators are not ``SYM_OPS[sg]`` as a set, written as xyz strings, labelled by their raw CIF's H-M and
   Hall symbols, with the row count): ``find_setting`` resolves it, the re-expressed operators equal ``SYM_OPS[sg]``
   as a set, the result does not depend on the operators' order, and det P > 0.
3. Same crystal, float64 synthetic, per class: a random standard description is carried into the stored setting by
   its literal atoms (``x_s = P x_n + p``, then a heavy-atom centroid and an O(3) Kabsch fit, no ``re_express``).
   (a) That stored description built literally with the class's own stored operators in the stored cell is the
   standard description's crystal (molecules matched both ways, each one way modulo the target's lattice) to 1e-9
   Angstrom: this is what makes (P, p) correct for the operators. (b) ``re_express`` of it is the standard description
   again (``checks.same_crystal``, 1e-9 Angstrom). (c) Controls: an origin moved by a generic vector and a doubled third
   basis vector are refused by ``setting_matches``, and the moved origin puts the stored-operator crystal more than
   0.1 Angstrom off.
4. ``re_express`` with an integer P is ``ops.change_basis`` (then the wrap); centred P1 settings (I1, B1) halve the
   cell at Z' = 1; a primitive setting of a centred group (det P = 2) is resolved too.
   4b (2026-10-17, after verifier finding v0-F2). Settings the grid pass leaves unresolved resolve to the same crystal
   (synthetic, float64): rhombohedral axes with an origin on the rhombohedral grid but off the hexagonal one (det P =
   3; the two cases of cctbx's 530 Hall settings the verifier's sweep left unresolved), origins off every rational
   grid (exact congruence solve), unreduced bases (a unimodular change with an entry of 3, 19 groups; reduced-basis
   pass); the congruence solver solves consistent systems and refuses inconsistent ones; the passes added after the
   first never change a result the first gives (every class of the file).
5. One crystal described in several settings of P2_1/c (the standard one, P2_1/n, P2_1/a and the other classes of sg
   14 in the file, each also through a further allowed monoclinic basis change) and of C2/c, reduces and folds to one
   canonical form (``same_crystal`` 1e-9 Angstrom).
6. Refusals: operators of another group are ``unresolved_setting``; ``nonstandard='refuse'`` gives the codes of the
   importer before D1; bad parameter values raise; a re-expressed row whose stored cell its stored operators do not
   preserve (an a-unique 'P 21/c 1 1' row with beta off 90 deg; 6 rows of the file, all in a- or c-unique
   settings) is ``setting_metric_mismatch``, a standard row with that defect is measured (``metric_dev``) but
   imported, and ``metric_tol`` is read. Synthetic float32 rows of every class go through the importer itself.
7. The mini CSD file (local data): default import against the importer before D1 and D2 row by row: standard rows
   unchanged (bitwise, molecule mirrored only where ``mirrored``), nonstandard rows now imported with their (P, p),
   heavy atoms against the raw CIF within 5e-3 Angstrom.
8. Real rows of every class (local data: ``fixtures/settings_rows.pt`` written by the full-file run, see FIXTURE; up
   to 3 imported and 2 refused rows per class, every class present): re-expressed, against the stored atoms
   expanded by the stored operators, and against the raw CIF within 5e-3 Angstrom; reduce_cell (sg 1-15) and fold run
   on them; the refused ones are refused for Z' > 1 or for ``setting_metric_mismatch``, the latter checked again in
   the stored setting.
9. Mirrored reference (D2): off-sheet Sohncke Z' = 1 rows (mini file and fixture) end at h = +1 on the sheet with the
   mirrored molecule and the same literal unit cell (1e-12 fractional) as the unmirrored description, and match their
   CIF; ``mirror_reference=False`` leaves them off the sheet (the parameter is read); a Z' = 2 Sohncke row is not
   mirrored; ``sheet=None`` mirrors nothing. ``ImportInfo.reference_handedness`` (-1 where mirrored) is the handedness
   under which GFN's frame step (MolData, collate, ``orient_molecule(mode='std')``) leaves every imported row's
   reference in place (mini file and fixture); without it, a mirrored reference X' comes back as X (verifier finding
   v1-F1, 2026-10-17: a consumer that re-aligns at +1 builds a different, clashing crystal).

Measured on 2026-10-16 over the whole file (canonical_data/importer/run_import_d1d2.py) and again on 2026-10-17 on
the current code (canonical_data/importer/fix_oct17/run_full; every per-row outcome, (P, p) and deviation identical):
every one of the 64,294 rows refused for their setting before D1 is resolved (85 operator sets); the 64,024
re-expressed Z' = 1 rows that import match their stored atoms under their stored operators to within 4.5e-6 Angstrom
of the importer's own pose deviation; a seeded random sample of 1,307 of them (up to 50 per operator set, 79 sets)
matches the raw CIFs to at most 3.4e-3 Angstrom (median 4.4e-6; the 2026-10-16 run took the first rows in file
order, verifier finding v0-F3), and a verifier's exhaustive check of all 64,024 (verify_importer_0, 2026-10-17) found
none above 5e-3 Angstrom; 38,961 mirrored rows give literal unit cells equal to the unmirrored ones (difference 0).
"""
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from mxtaltools.constants.space_group_info import LATTICE_TYPE, SYM_OPS
from mxtaltools.crystal_fd import checks, domains, io_mxt, ops, settings, so3
from mxtaltools.crystal_fd.description import CrystalDescription, MoleculeRef
from mxtaltools.crystal_search.standardize import setting_shift

F64 = torch.float64
NB = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3).to(F64)
DATA = Path(os.environ.get('MXT_CSD_DATA', 'D:/crystal_datasets'))
CIF_DIR = DATA / 'CSD_dump'
_ROOT = Path(__file__).resolve().parents[2]
#: real rows per setting class, written by the full-file run (gfn_diffusion artifacts; local data)
FIXTURE = Path(os.environ.get('MXT_FD_SETTINGS_ROWS', str(
    _ROOT.parent / 'gfn_diffusion/energy_sampling/artifacts/fd_design_oct15/canonical_data/importer/'
                   'run_oct16b/fixtures/settings_rows.pt')))
LEGACY = dict(nonstandard='refuse', mirror_reference=False)
#: Angstrom
SYN_TOL = 1e-9
CIF_TOL = 5e-3

# CLASSES_BEGIN (generated from the full-file run by canonical_data/importer/make_classes.py; do not edit by hand)
CLASSES = {
    '14|P 21/n|-P 2yn': (14, ['-x,-y,-z', '-x+1/2,y+1/2,-z+1/2', 'x+1/2,-y+1/2,z+1/2', 'x,y,z'], 56605),
    '14|P 21/a|-P 2yab': (14, ['-x,-y,-z', '-x+1/2,y+1/2,-z', 'x+1/2,-y+1/2,z', 'x,y,z'], 3904),
    '15|I 2/a|-I 2ya': (15, ['-x,-y,-z', '-x+1/2,-y+1/2,-z+1/2', '-x,y+1/2,-z+1/2', '-x+1/2,y,-z', 'x,-y+1/2,z+1/2', 'x+1/2,-y,z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 945),
    '7|P n|P -2yac': (7, ['x+1/2,-y,z+1/2', 'x,y,z'], 598),
    '61|P c a b|-P 2bc 2ac': (61, ['-x,-y,-z', '-x,-y+1/2,z+1/2', '-x+1/2,y+1/2,-z', '-x+1/2,y,z+1/2', 'x+1/2,-y,-z+1/2', 'x+1/2,-y+1/2,z', 'x,y+1/2,-z+1/2', 'x,y,z'], 495),
    '5|I 2|I 2y': (5, ['-x,y,-z', '-x+1/2,y+1/2,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 173),
    '33|P n 21 a|P -2ac -2n': (33, ['-x,y+1/2,-z', '-x+1/2,y+1/2,z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z'], 171),
    '13|P 2/n|-P 2yac': (13, ['-x,-y,-z', '-x+1/2,y,-z+1/2', 'x+1/2,-y,z+1/2', 'x,y,z'], 165),
    '9|I a|I -2ya': (9, ['x,-y+1/2,z+1/2', 'x+1/2,-y,z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 161),
    '29|P b c 21|P 2c -2b': (29, ['-x,-y,z+1/2', '-x,y+1/2,z', 'x,-y+1/2,z+1/2', 'x,y,z'], 108),
    '15|I 2/c|-I 2yc': (15, ['-x,-y,-z', '-x+1/2,-y+1/2,-z+1/2', '-x,y,-z+1/2', '-x+1/2,y+1/2,-z', 'x,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 98),
    '33|P 21 c n|P -2n 2a': (33, ['x+1/2,-y,-z', 'x,-y+1/2,z+1/2', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 88),
    '33|P c 21 n|P -2n -2ac': (33, ['-x,y+1/2,-z', '-x+1/2,y,z+1/2', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 79),
    '18|P 2 21 21|P 2bc 2': (18, ['-x,-y+1/2,z+1/2', '-x,y+1/2,-z+1/2', 'x,-y,-z', 'x,y,z'], 67),
    '33|P 21 n b|P -2bc 2a': (33, ['x+1/2,-y,-z', 'x+1/2,-y+1/2,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 66),
    '29|P c 21 b|P -2bc -2c': (29, ['-x,y+1/2,-z', '-x,y,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 50),
    '33|P b n 21|P 2c -2ab': (33, ['-x,-y,z+1/2', '-x+1/2,y+1/2,z', 'x+1/2,-y+1/2,z+1/2', 'x,y,z'], 49),
    '15|A 2/a|-A 2ya': (15, ['-x,-y,-z', '-x,-y+1/2,-z+1/2', '-x+1/2,y,-z', '-x+1/2,y+1/2,-z+1/2', 'x+1/2,-y,z', 'x+1/2,-y+1/2,z+1/2', 'x,y,z', 'x,y+1/2,z+1/2'], 48),
    '56|P n a a|-P 2ac 2bc': (56, ['-x,-y,-z', '-x+1/2,-y,z+1/2', '-x+1/2,y+1/2,-z', '-x,y+1/2,z+1/2', 'x,-y+1/2,-z+1/2', 'x+1/2,-y+1/2,z', 'x+1/2,y,-z+1/2', 'x,y,z'], 30),
    '7|P a|P -2ya': (7, ['x+1/2,-y,z', 'x,y,z'], 30),
    '60|P b n a|-P 2ac 2b': (60, ['-x,-y,-z', '-x+1/2,-y,z+1/2', '-x+1/2,y+1/2,-z+1/2', '-x,y+1/2,z', 'x,-y+1/2,-z', 'x+1/2,-y+1/2,z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z'], 24),
    '29|P 21 a b|P -2b 2a': (29, ['x+1/2,-y,-z', 'x+1/2,-y+1/2,z', 'x,y+1/2,-z', 'x,y,z'], 23),
    '43|F 2 d d|F -2d 2': (43, ['x,-y,-z', 'x,-y+1/2,-z+1/2', 'x+1/2,-y,-z+1/2', 'x+1/2,-y+1/2,-z', 'x+1/4,-y+1/4,z+1/4', 'x+1/4,-y+3/4,z+3/4', 'x+3/4,-y+1/4,z+3/4', 'x+3/4,-y+3/4,z+1/4', 'x+1/4,y+1/4,-z+1/4', 'x+1/4,y+3/4,-z+3/4', 'x+3/4,y+1/4,-z+3/4', 'x+3/4,y+3/4,-z+1/4', 'x,y,z', 'x,y+1/2,z+1/2', 'x+1/2,y,z+1/2', 'x+1/2,y+1/2,z'], 23),
    '13|P 2/a|-P 2ya': (13, ['-x,-y,-z', '-x+1/2,y,-z', 'x+1/2,-y,z', 'x,y,z'], 22),
    '60|P n c a|-P 2a 2n': (60, ['-x,-y,-z', '-x+1/2,-y,z', '-x,y+1/2,-z+1/2', '-x+1/2,y+1/2,z+1/2', 'x+1/2,-y+1/2,-z+1/2', 'x,-y+1/2,z+1/2', 'x+1/2,y,-z', 'x,y,z'], 20),
    '29|P 21 c a|P -2ac 2a': (29, ['x+1/2,-y,-z', 'x,-y,z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z'], 18),
    '19|P 21 21 21|P 2bc 2ac': (19, ['-x,-y+1/2,z+1/2', '-x+1/2,y+1/2,-z', 'x+1/2,-y,-z+1/2', 'x,y,z'], 17),
    '60|P c a n|-P 2n 2c': (60, ['-x,-y,-z', '-x+1/2,-y+1/2,z+1/2', '-x+1/2,y+1/2,-z', '-x,y,z+1/2', 'x,-y,-z+1/2', 'x+1/2,-y+1/2,z', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 15),
    '9|A a|A -2ya': (9, ['x+1/2,-y,z', 'x+1/2,-y+1/2,z+1/2', 'x,y,z', 'x,y+1/2,z+1/2'], 15),
    '60|P n a b|-P 2bc 2n': (60, ['-x,-y,-z', '-x,-y+1/2,z+1/2', '-x+1/2,y,-z', '-x+1/2,y+1/2,z+1/2', 'x+1/2,-y+1/2,-z+1/2', 'x+1/2,-y,z', 'x,y+1/2,-z+1/2', 'x,y,z'], 14),
    '29|P b 21 a|P -2a -2ab': (29, ['-x,y+1/2,-z', '-x+1/2,y+1/2,z', 'x+1/2,y,-z', 'x,y,z'], 13),
    '5|A 2|A 2y': (5, ['-x,y,-z', '-x,y+1/2,-z+1/2', 'x,y,z', 'x,y+1/2,z+1/2'], 13),
    '60|P c n b|-P 2b 2ac': (60, ['-x,-y,-z', '-x,-y+1/2,z', '-x+1/2,y+1/2,-z+1/2', '-x+1/2,y,z+1/2', 'x+1/2,-y,-z+1/2', 'x+1/2,-y+1/2,z+1/2', 'x,y+1/2,-z', 'x,y,z'], 10),
    '9|I c|I -2yc': (9, ['x,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 10),
    '15|C 2/n|-C 2yac': (15, ['-x,-y,-z', '-x+1/2,-y+1/2,-z', '-x,y+1/2,-z+1/2', '-x+1/2,y,-z+1/2', 'x,-y+1/2,z+1/2', 'x+1/2,-y,z+1/2', 'x,y,z', 'x+1/2,y+1/2,z'], 8),
    '18|P 21 2 21|P 2ac 2ac': (18, ['-x+1/2,-y,z+1/2', '-x,y,-z', 'x+1/2,-y,-z+1/2', 'x,y,z'], 8),
    '41|C 2 c b|C -2ac 2': (41, ['x,-y,-z', 'x+1/2,-y+1/2,-z', 'x,-y+1/2,z+1/2', 'x+1/2,-y,z+1/2', 'x,y+1/2,-z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z'], 8),
    '56|P b n b|-P 2bc 2ab': (56, ['-x,-y,-z', '-x,-y+1/2,z+1/2', '-x+1/2,y,-z+1/2', '-x+1/2,y+1/2,z', 'x+1/2,-y+1/2,-z', 'x+1/2,-y,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 8),
    '15|A 2/n|-A 2yab': (15, ['-x,-y,-z', '-x,-y+1/2,-z+1/2', '-x+1/2,y,-z+1/2', '-x+1/2,y+1/2,-z', 'x+1/2,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y,z', 'x,y+1/2,z+1/2'], 7),
    '88|I 41/a|I 4bw -1bw': (88, ['-x,-y+1/2,-z+1/4', '-x+1/2,-y,-z+3/4', '-x,-y,z', '-x+1/2,-y+1/2,z+1/2', '-y,x,-z', '-y+1/2,x+1/2,-z+1/2', '-y,x+1/2,z+1/4', '-y+1/2,x,z+3/4', 'y,-x,-z', 'y+1/2,-x+1/2,-z+1/2', 'y,-x+1/2,z+1/4', 'y+1/2,-x,z+3/4', 'x,y+1/2,-z+1/4', 'x+1/2,y,-z+3/4', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 7),
    '86|P 42/n|P 4n -1n': (86, ['-x+1/2,-y+1/2,-z+1/2', '-x,-y,z', '-y,x,-z', '-y+1/2,x+1/2,z+1/2', 'y,-x,-z', 'y+1/2,-x+1/2,z+1/2', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 6),
    '19|P 21 21 21|P 2c 2n': (19, ['-x,-y,z+1/2', '-x+1/2,y+1/2,-z', 'x+1/2,-y+1/2,-z+1/2', 'x,y,z'], 5),
    '19|P 21 21 21|P 2n 2a': (19, ['-x+1/2,-y+1/2,z+1/2', '-x,y+1/2,-z+1/2', 'x+1/2,-y,-z', 'x,y,z'], 5),
    '45|I 2 c b|I -2a 2': (45, ['x,-y,-z', 'x+1/2,-y+1/2,-z+1/2', 'x,-y+1/2,z+1/2', 'x+1/2,-y,z', 'x,y+1/2,-z+1/2', 'x+1/2,y,-z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 5),
    '7|P n|P -2yn': (7, ['x+1/2,-y+1/2,z+1/2', 'x,y,z'], 5),
    '18|P 21 2 21|P 2ac 2a': (18, ['-x+1/2,-y,z+1/2', '-x,y,-z+1/2', 'x+1/2,-y,-z', 'x,y,z'], 3),
    '19|P 21 21 21|P 2c 2ab': (19, ['-x,-y,z+1/2', '-x+1/2,y+1/2,-z+1/2', 'x+1/2,-y+1/2,-z', 'x,y,z'], 3),
    '14|P 21/c 1 1|-P 2xac': (14, ['-x,-y,-z', '-x+1/2,y,z+1/2', 'x+1/2,-y,-z+1/2', 'x,y,z'], 2),
    '19|P 21 21 21|P 2ac 2n': (19, ['-x+1/2,-y,z+1/2', '-x,y+1/2,-z', 'x+1/2,-y+1/2,-z+1/2', 'x,y,z'], 2),
    '1|I 1|I 1': (1, ['x,y,z', 'x+1/2,y+1/2,z+1/2'], 2),
    '20|B 2 21 2|B 2b 2': (20, ['-x,-y+1/2,z', '-x+1/2,-y+1/2,z+1/2', '-x,y+1/2,-z', '-x+1/2,y+1/2,-z+1/2', 'x,-y,-z', 'x+1/2,-y,-z+1/2', 'x,y,z', 'x+1/2,y,z+1/2'], 2),
    '33|P 21 c n|P -2ab 2n': (33, ['x+1/2,-y+1/2,-z+1/2', 'x,-y,z+1/2', 'x+1/2,y+1/2,-z', 'x,y,z'], 2),
    '34|P 2 n n|P -2n 2': (34, ['x,-y,-z', 'x+1/2,-y+1/2,z+1/2', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 2),
    '4|P 21 1 1|P 2xa': (4, ['x+1/2,-y,-z', 'x,y,z'], 2),
    '4|P 21|P 2ybc': (4, ['-x,y+1/2,-z+1/2', 'x,y,z'], 2),
    '4|P 21|P 2yn': (4, ['-x+1/2,y+1/2,-z+1/2', 'x,y,z'], 2),
    '9|A n|A -2yab': (9, ['x+1/2,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y,z', 'x,y+1/2,z+1/2'], 2),
    '9|C n|C -2yac': (9, ['x,-y+1/2,z+1/2', 'x+1/2,-y,z+1/2', 'x,y,z', 'x+1/2,y+1/2,z'], 2),
    '14|P 1 1 21/a|-P 2ac': (14, ['-x,-y,-z', '-x+1/2,-y,z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z'], 1),
    '14|P 1 1 21/b|-P 2bc': (14, ['-x,-y,-z', '-x,-y+1/2,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 1),
    '14|P 1 1 21/n|-P 2n': (14, ['-x,-y,-z', '-x+1/2,-y+1/2,z+1/2', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 1),
    '14|P 21/n 1 1|-P 2xn': (14, ['-x,-y,-z', '-x+1/2,y+1/2,z+1/2', 'x+1/2,-y+1/2,-z+1/2', 'x,y,z'], 1),
    '15|I 1 1 2/a|-I 2a': (15, ['-x,-y,-z', '-x+1/2,-y+1/2,-z+1/2', '-x,-y+1/2,z+1/2', '-x+1/2,-y,z', 'x,y+1/2,-z+1/2', 'x+1/2,y,-z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 1),
    '1|B 1|B 1': (1, ['x,y,z', 'x+1/2,y,z+1/2'], 1),
    '27|P 2 a a|P -2a 2': (27, ['x,-y,-z', 'x+1/2,-y,z', 'x+1/2,y,-z', 'x,y,z'], 1),
    '32|P 2 c b|P -2bc 2': (32, ['x,-y,-z', 'x,-y+1/2,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 1),
    '33|P b n 21|P 2bc -2ab': (33, ['-x,-y+1/2,z+1/2', '-x+1/2,y+1/2,z', 'x+1/2,-y,z+1/2', 'x,y,z'], 1),
    '33|P c 21 n|P -2ab -2ac': (33, ['-x,y+1/2,-z+1/2', '-x+1/2,y,z+1/2', 'x+1/2,y+1/2,-z', 'x,y,z'], 1),
    '33|P n 21 a|P -2a -2bc': (33, ['-x+1/2,y+1/2,-z+1/2', '-x,y+1/2,z+1/2', 'x+1/2,y,-z', 'x,y,z'], 1),
    '33|P n a 21|P 2n -2bc': (33, ['-x+1/2,-y+1/2,z+1/2', '-x,y+1/2,z+1/2', 'x+1/2,-y,z', 'x,y,z'], 1),
    '41|A c 2 a|A -2ab -2ab': (41, ['-x,y,-z', '-x,y+1/2,-z+1/2', '-x+1/2,y,z+1/2', '-x+1/2,y+1/2,z', 'x+1/2,y,-z+1/2', 'x+1/2,y+1/2,-z', 'x,y,z', 'x,y+1/2,z+1/2'], 1),
    '41|C c 2 a|C -2ac -2ac': (41, ['-x,y,-z', '-x+1/2,y+1/2,-z', '-x,y+1/2,z+1/2', '-x+1/2,y,z+1/2', 'x,y+1/2,-z+1/2', 'x+1/2,y,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z'], 1),
    '43|F d 2 d|F -2d -2d': (43, ['-x,y,-z', '-x,y+1/2,-z+1/2', '-x+1/2,y,-z+1/2', '-x+1/2,y+1/2,-z', '-x+1/4,y+1/4,z+1/4', '-x+1/4,y+3/4,z+3/4', '-x+3/4,y+1/4,z+3/4', '-x+3/4,y+3/4,z+1/4', 'x+1/4,y+1/4,-z+1/4', 'x+1/4,y+3/4,-z+3/4', 'x+3/4,y+1/4,-z+3/4', 'x+3/4,y+3/4,-z+1/4', 'x,y,z', 'x,y+1/2,z+1/2', 'x+1/2,y,z+1/2', 'x+1/2,y+1/2,z'], 1),
    '45|I 2 c b|I -2b 2a': (45, ['x,-y+1/2,-z+1/2', 'x+1/2,-y,-z', 'x,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y+1/2,-z', 'x+1/2,y,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 1),
    '45|I c 2 a|I -2b -2b': (45, ['-x,y,-z', '-x+1/2,y+1/2,-z+1/2', '-x,y+1/2,z', '-x+1/2,y,z+1/2', 'x,y+1/2,-z', 'x+1/2,y,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 1),
    '4|P 21|P 2yab': (4, ['-x+1/2,y+1/2,-z', 'x,y,z'], 1),
    '52|P n a n|-P 2n 2bc': (52, ['-x,-y,-z', '-x+1/2,-y+1/2,z+1/2', '-x+1/2,y,-z', '-x,y+1/2,z+1/2', 'x,-y+1/2,-z+1/2', 'x+1/2,-y,z', 'x+1/2,y+1/2,-z+1/2', 'x,y,z'], 1),
    '52|P n c n|-P 2ab 2n': (52, ['-x,-y,-z', '-x+1/2,-y+1/2,z', '-x,y,-z+1/2', '-x+1/2,y+1/2,z+1/2', 'x+1/2,-y+1/2,-z+1/2', 'x,-y,z+1/2', 'x+1/2,y+1/2,-z', 'x,y,z'], 1),
    '54|P b c b|-P 2bc 2b': (54, ['-x,-y,-z', '-x,-y+1/2,z+1/2', '-x,y,-z+1/2', '-x,y+1/2,z', 'x,-y+1/2,-z', 'x,-y,z+1/2', 'x,y+1/2,-z+1/2', 'x,y,z'], 1),
    '5|C 2|C 2ya': (5, ['-x,y+1/2,-z', '-x+1/2,y,-z', 'x,y,z', 'x+1/2,y+1/2,z'], 1),
    '5|C 2|C 2yc': (5, ['-x,y,-z+1/2', '-x+1/2,y+1/2,-z+1/2', 'x,y,z', 'x+1/2,y+1/2,z'], 1),
    '5|I 2|I 2yc': (5, ['-x,y,-z+1/2', '-x+1/2,y+1/2,-z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 1),
    '73|I c a b|-I 2a 2b': (73, ['-x,-y,-z', '-x+1/2,-y+1/2,-z+1/2', '-x,-y+1/2,z+1/2', '-x+1/2,-y,z', '-x,y,-z+1/2', '-x+1/2,y+1/2,-z', '-x,y+1/2,z', '-x+1/2,y,z+1/2', 'x,-y+1/2,-z', 'x+1/2,-y,-z+1/2', 'x,-y,z+1/2', 'x+1/2,-y+1/2,z', 'x,y+1/2,-z+1/2', 'x+1/2,y,-z', 'x,y,z', 'x+1/2,y+1/2,z+1/2'], 1),
    '7|P c|P -2ybc': (7, ['x,-y+1/2,z+1/2', 'x,y,z'], 1),
    '86|P 42/n|-P 4ac': (86, ['-x,-y,-z', '-x+1/2,-y+1/2,z', '-y,x+1/2,-z+1/2', '-y+1/2,x,z+1/2', 'y+1/2,-x,-z+1/2', 'y,-x+1/2,z+1/2', 'x+1/2,y+1/2,-z', 'x,y,z'], 1),
}
# CLASSES_END


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_xyz(ops_xyz) -> np.ndarray:
    """[n, 4, 4] float64 affine operators from CIF-style strings such as ``'-x+1/2,y+1/2,-z+1/2'``."""
    out = []
    for s in ops_xyz:
        m = np.eye(4)
        for i, comp in enumerate(s.split(',')):
            f = lambda x, y, z, c=comp: eval(c, {'__builtins__': {}}, dict(x=x, y=y, z=z))  # noqa: E731
            f0 = Fraction(f(Fraction(0), Fraction(0), Fraction(0)))
            m[i, :3] = [float(Fraction(f(*e)) - f0) for e in ((Fraction(1), 0, 0), (0, Fraction(1), 0),
                                                              (0, 0, Fraction(1)))]
            m[i, 3] = float(f0)
        out.append(m)
    return np.stack(out)


def _class_ops(label):
    sg, xyz, _ = CLASSES[label]
    return sg, parse_xyz(xyz)


def _mol(chiral_molecule) -> MoleculeRef:
    return MoleculeRef.from_coords(torch.as_tensor(chiral_molecule['pos']), torch.as_tensor(chiral_molecule['z']))


def _random_std(sg, k, mol, seed) -> CrystalDescription:
    g = torch.Generator().manual_seed(seed)
    cell = checks.random_cells(LATTICE_TYPE[sg], k, g)
    t = torch.rand(k, 3, generator=g, dtype=F64)
    R = so3.random_rotations(k, g)
    h = torch.where(torch.rand(k, generator=g) < 0.5, -1, 1)
    return CrystalDescription.from_parts(sg, cell, t, R, h)


def _to_stored(d_std: CrystalDescription, mol: MoleculeRef, P, p) -> CrystalDescription:
    """The crystal of ``d_std`` described in the stored setting ``x_s = P x_n + p``, from its literal atoms: stored
    cell from the metric, heavy-atom centroid, O(3) Kabsch fit of the molecule (independent of ``re_express``)."""
    P = torch.as_tensor(P, dtype=F64)
    p = torch.as_tensor(p, dtype=F64)
    X = mol.pos
    heavy = mol.z > 1
    Pi = torch.linalg.inv(P)
    Gs = Pi.T @ ops.metric_tensor(d_std.cell) @ Pi
    cell_s = ops.cell_from_metric_tensor(Gs)
    Tn, Ts = d_std.cell_matrix(), ops.cell_matrices(cell_s)
    f_n = d_std.t[:, 0, None] + torch.linalg.solve(Tn, d_std.pose[:, 0] @ X.T).transpose(1, 2)  # [k, n, 3]
    f_s = f_n @ P.T + p
    t_s = f_s[:, heavy].mean(1)
    Y = torch.einsum('kij,knj->kni', Ts, f_s - t_s[:, None])
    U, _, Vt = torch.linalg.svd(X.T[None] @ Y)
    pose = Vt.transpose(1, 2) @ U.transpose(1, 2)                   # pose X ~ Y, det +-1
    return CrystalDescription.from_poses(d_std.sg, cell_s, t_s, pose)


def _images(f, W, w):
    """[..., G, n, 3] molecules W_g f + w_g of the fractional molecule f [..., n, 3]."""
    return torch.einsum('gij,...nj->...gni', W, f) + w[:, None]


def _one_way(A, Bm, T):
    """[k] max over molecules of A of min over molecules of Bm of the largest atom distance (Angstrom), atoms in order,
    each difference at its minimum image modulo the lattice of T [k, 3, 3] (rounded, then 27 neighbours)."""
    d = A[:, :, None] - Bm[:, None]                                    # [k, Ga, Gb, n, 3]
    d = d - torch.round(d)
    cart = torch.einsum('kij,kabnsj->kabnsi', T, d[..., None, :] + NB)
    dist = cart.norm(dim=-1).amin(-1).amax(-1)                        # [k, Ga, Gb]
    return dist.amin(2).amax(1)


def _stored_vs_std(d_s, ops_s, d_std, mol, P, p):
    """[k] Angstrom: the stored description built with the stored operators ``ops_s`` in the stored cell against the
    standard description built with SYM_OPS[sg], molecules matched both ways, each one way modulo the target's
    lattice (the coarser lattice side sees the other side's extra molecules as lattice images)."""
    P = torch.as_tensor(P, dtype=F64)
    p = torch.as_tensor(p, dtype=F64)
    o = torch.as_tensor(ops_s, dtype=F64)
    X = mol.pos
    Ts, Tn = d_s.cell_matrix(), d_std.cell_matrix()
    f_s = d_s.t[:, 0, None] + torch.linalg.solve(Ts, d_s.pose[:, 0] @ X.T).transpose(1, 2)
    A_s = _images(f_s, o[:, :3, :3], o[:, :3, 3])                     # [k, Gs, n, 3] stored frame
    A_n = checks.unit_cell_fractional(d_std, mol)                     # [k, Gn, n, 3] standard frame
    n_in_s = A_n @ P.T + p
    s_in_n = (A_s - p) @ torch.linalg.inv(P).T
    return torch.maximum(_one_way(n_in_s, A_s, Ts), _one_way(s_in_n, A_n, Tn))


def _set_dev(fa, za, fb, zb, T) -> float:
    """Largest distance (Angstrom) from an atom of either set to the nearest atom of the same element in the other."""
    out = 0.0
    for (f1, z1, f2, z2) in ((fa, za, fb, zb), (fb, zb, fa, za)):
        for el in torch.unique(z1).tolist():
            a, b = f1[z1 == el], f2[z2 == el]
            if len(b) == 0:
                return float('inf')
            for lo in range(0, len(a), 256):
                d = a[lo:lo + 256, None] - b[None]
                d = d - torch.round(d)
                cart = torch.einsum('ij,abnj->abni', T, d[:, :, None] + NB)
                out = max(out, float(cart.norm(dim=-1).amin(-1).amin(1).max()))
    return out


def _cif_dev(ident, desc_row, mol, P, p):
    """Heavy atoms of the literal unit cell against the raw CIF (its own operators), compared in the description's cell
    (CIF fractional coordinates mapped by x_n = P^-1 (x_s - p)); None when the CIF is not on disk."""
    from mxtaltools.dataset_utils.construction.cif import read_cif
    path = CIF_DIR / f'{ident}.cif'
    if not path.exists():
        return None
    cif = read_cif(path, require_bonds=False, require_integer_z_prime=False)[0]
    P = torch.as_tensor(P, dtype=F64)
    p = torch.as_tensor(p, dtype=F64)
    cc = torch.cat([torch.as_tensor(cif.cell_lengths), torch.as_tensor(cif.cell_angles)]).to(F64)[None]
    cell_n = ops.cell_from_metric_tensor(P.T @ ops.metric_tensor(cc) @ P)
    assert torch.allclose(cell_n, desc_row.cell, rtol=1e-5), (ident, cell_n, desc_row.cell)
    heavy = torch.as_tensor(cif.sites.z) > 1
    f0 = torch.as_tensor(cif.sites.frac, dtype=F64)[heavy]
    cops = torch.as_tensor(np.stack(cif.symmetry.operators), dtype=F64)
    f_cif = (torch.einsum('gij,nj->gni', cops[:, :3, :3], f0) + cops[:, None, :3, 3]).reshape(-1, 3)
    f_cif = (f_cif - p) @ torch.linalg.inv(P).T
    z_cif = torch.as_tensor(cif.sites.z)[heavy].repeat(len(cops))
    f_lit = checks.unit_cell_fractional(desc_row, mol)[0]
    z_lit = mol.z.repeat(f_lit.shape[0])
    f_lit = f_lit.reshape(-1, 3)
    keep = z_lit > 1
    return _set_dev(f_lit[keep], z_lit[keep], f_cif - torch.floor(f_cif), z_cif, desc_row.cell_matrix()[0])


def _load(path):
    if not Path(path).exists():
        pytest.skip(f'{path} not on this machine')
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# 1-2: operator sets
# ---------------------------------------------------------------------------

def test_standard_settings_are_identity():
    rng = np.random.default_rng(0)
    for sg in range(1, 231):
        std = settings.std_ops(sg)
        for o in (std, std[rng.permutation(len(std))]):
            r = settings.find_setting(o, sg)
            assert r is not None and np.array_equal(r.P, np.eye(3)) and np.array_equal(r.p, np.zeros(3)), sg
        assert settings.setting_matches(std, sg, np.eye(3), np.zeros(3))


def test_every_class_of_the_file_is_listed():
    assert len(CLASSES) == 85, len(CLASSES)
    assert sum(c[2] for c in CLASSES.values()) == 64294, 'rows of the file refused for their setting before D1'


@pytest.mark.parametrize('label', sorted(CLASSES))
def test_class_reexpresses_to_standard(label):
    sg, o = _class_ops(label)
    assert not io_mxt.operator_set_match(o, sg)[0] or len(o) != len(SYM_OPS[sg])
    r = settings.find_setting(o, sg)
    assert r is not None, label
    assert np.linalg.det(r.P) > 0
    assert settings.operator_sets_equal(settings.transform_ops(o, r.P, r.p), settings.std_ops(sg)), label
    assert settings.setting_matches(o, sg, r.P, r.p)
    perm = np.random.default_rng(1).permutation(len(o))
    r2 = settings.find_setting(o[perm], sg)
    assert np.array_equal(r.P, r2.P) and np.array_equal(r.p, r2.p)


# ---------------------------------------------------------------------------
# 3: same crystal, synthetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('label', sorted(CLASSES))
def test_class_same_crystal(label, chiral_molecule):
    sg, o = _class_ops(label)
    mol = _mol(chiral_molecule)
    r = settings.find_setting(o, sg)
    d_std = _random_std(sg, 4, mol, seed=sum(map(ord, label)))
    d_s = _to_stored(d_std, mol, r.P, r.p)
    # (a) the stored description, built with the class's own operators, is the standard crystal
    dev = _stored_vs_std(d_s, o, d_std, mol, r.P, r.p)
    assert float(dev.max()) < SYN_TOL, (label, dev)
    # (b) re_express brings it back
    back = settings.re_express(d_s, r.P, r.p)
    assert float(checks.same_crystal(back, d_std, mol, cell_rtol=1e-9).max()) < SYN_TOL
    assert bool(((back.t >= 0) & (back.t < 1)).all())
    # (c) controls
    delta = r.P @ np.array([0.1234, 0.0567, 0.0891])
    assert not settings.setting_matches(o, sg, r.P @ np.diag([1.0, 1.0, 2.0]), r.p)
    if sg > 1:                                                        # P1 has every origin
        assert not settings.setting_matches(o, sg, r.P, r.p + delta)
        d_bad = _to_stored(d_std, mol, r.P, r.p + delta)
        bad = _stored_vs_std(d_bad, o, d_std, mol, r.P, r.p + delta)
        assert float(bad.min()) > 0.1, (label, bad)


# ---------------------------------------------------------------------------
# 4: re_express itself; centred settings
# ---------------------------------------------------------------------------

def test_re_express_is_change_basis_for_integer_P(chiral_molecule):
    mol = _mol(chiral_molecule)
    d = _random_std(14, 16, mol, seed=3)
    N = torch.tensor([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    s = torch.tensor([0.25, 0.5, 0.0], dtype=F64)
    a = settings.re_express(d, N, s, wrap=False)
    b = ops.change_basis(d, N, s)
    for f in ('cell', 't', 'R', 'h'):
        assert torch.equal(getattr(a, f), getattr(b, f)), f
    w = settings.re_express(d, N, s)
    assert torch.equal(w.t, b.t - torch.floor(b.t))
    with pytest.raises(ValueError):
        settings.re_express(d, torch.diag(torch.tensor([-1.0, 1.0, 1.0])), s)


def _op(W, w):
    m = np.eye(4)
    m[:3, :3] = W
    m[:3, 3] = w
    return m


@pytest.mark.parametrize('name,sg,centring', [('I1', 1, [0.5, 0.5, 0.5]), ('B1', 1, [0.5, 0.0, 0.5]),
                                              ('C1', 1, [0.5, 0.5, 0.0]), ('I-1', 2, [0.5, 0.5, 0.5])])
def test_centred_triclinic_settings_halve_the_cell(name, sg, centring, chiral_molecule):
    mol = _mol(chiral_molecule)
    rots = [np.eye(3)] if sg == 1 else [np.eye(3), -np.eye(3)]
    o = np.stack([_op(W, c) for W in rots for c in ([0, 0, 0], centring)])
    r = settings.find_setting(o, sg)
    assert r is not None and abs(np.linalg.det(r.P) - 0.5) < 1e-12, name
    assert settings.operator_sets_equal(settings.transform_ops(o, r.P, r.p), settings.std_ops(sg))
    d_std = _random_std(sg, 4, mol, seed=5)
    d_s = _to_stored(d_std, mol, r.P, r.p)
    assert float(_stored_vs_std(d_s, o, d_std, mol, r.P, r.p).max()) < SYN_TOL
    back = settings.re_express(d_s, r.P, r.p)
    vol = lambda c: torch.linalg.det(ops.cell_matrices(c))  # noqa: E731
    assert torch.allclose(vol(back.cell), vol(d_s.cell) / 2, rtol=1e-12)
    assert float(checks.same_crystal(back, d_std, mol, cell_rtol=1e-9).max()) < SYN_TOL
    # the centring dropped from the stored list is not the same group
    assert not settings.setting_matches(o[:len(rots) * 2:2], sg, r.P, r.p)


def test_primitive_setting_of_a_centred_group(chiral_molecule):
    """C2/c given in a primitive cell (4 operators, x_C = P0 x_prim): resolved with det P = 2, the same crystal."""
    mol = _mol(chiral_molecule)
    P0 = np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    o = settings.transform_ops(settings.std_ops(15), P0, np.zeros(3))
    assert len(o) == 4
    r = settings.find_setting(o, 15)
    assert r is not None and abs(np.linalg.det(r.P) - 2) < 1e-12
    d_std = _random_std(15, 4, mol, seed=6)
    d_s = _to_stored(d_std, mol, r.P, r.p)
    assert float(_stored_vs_std(d_s, o, d_std, mol, r.P, r.p).max()) < SYN_TOL
    assert float(checks.same_crystal(settings.re_express(d_s, r.P, r.p), d_std, mol, cell_rtol=1e-9).max()) < SYN_TOL


# ---------------------------------------------------------------------------
# 4b: settings the first pass misses (origin off the grid, unreduced basis)
# ---------------------------------------------------------------------------

#: x_hex = HEX_FROM_RH x_rh: columns are the obverse rhombohedral basis vectors in hexagonal coordinates (det 1/3)
HEX_FROM_RH = np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]], dtype=np.float64) / 3


def _exact_std(sg) -> np.ndarray:
    """SYM_OPS[sg] with translations snapped to multiples of 1/24 (SYM_OPS stores thirds to 8 decimals, 1e-8 off,
    which a basis change with entries up to 15 amplifies to 1e-6 Angstrom; the CSD featurizer stores exact
    fractions)."""
    o = settings.std_ops(sg).copy()
    o[:, :3, 3] = np.round(o[:, :3, 3] * 24) / 24
    return o


def _resolved_same_crystal(o, sg, mol, seed, det=None, tol=SYN_TOL):
    """find_setting resolves the stored operators ``o``, re-expresses them as SYM_OPS[sg] as a set, and a float64
    synthetic crystal written in the stored setting is the standard crystal (stored operators, ``tol`` Angstrom) and
    comes back through re_express (same_crystal, ``tol`` Angstrom)."""
    r = settings.find_setting(o, sg)
    assert r is not None, sg
    assert settings.operator_sets_equal(settings.transform_ops(o, r.P, r.p), settings.std_ops(sg)), sg
    assert settings.setting_matches(o, sg, r.P, r.p)
    if det is not None:
        assert abs(np.linalg.det(r.P) - det) < 1e-9, (sg, np.linalg.det(r.P))
    d_std = _random_std(sg, 3, mol, seed=seed)
    d_s = _to_stored(d_std, mol, r.P, r.p)
    assert float(_stored_vs_std(d_s, o, d_std, mol, r.P, r.p).max()) < tol, sg
    back = settings.re_express(d_s, r.P, r.p)
    assert float(checks.same_crystal(back, d_std, mol, cell_rtol=1e-9).max()) < tol, sg
    return r


@pytest.mark.parametrize('sg,shift', [(146, [0.5, 0.0, 5 / 6]), (148, [5 / 12, 2 / 3, 0.0]), (148, [0.0, 0.0, 0.0]),
                                      (155, [1 / 8, 3 / 8, 1 / 2])])
def test_rhombohedral_axes_with_origin_off_the_hexagonal_grid(sg, shift, chiral_molecule):
    """R 3 :R / R -3 :R / R 3 2 :R with an origin shift on the rhombohedral cell's 1/24 grid, which maps off the
    hexagonal cell's grid (the cases the grid pass left unresolved, verifier sweep 2026-10-17): resolved by the exact
    origin solve with det P = 3, the same crystal."""
    s = np.asarray(shift)
    o = settings.transform_ops(_exact_std(sg), HEX_FROM_RH, HEX_FROM_RH @ s)
    assert len(o) * 3 == len(SYM_OPS[sg])
    _resolved_same_crystal(o, sg, _mol(chiral_molecule), seed=sg, det=3.0)


def test_origin_off_every_grid(chiral_molecule):
    """An origin shift that is no small fraction (P2_1/c, Pbca, I4_1/a, P-1, C2/c): the exact origin solve finds it."""
    mol = _mol(chiral_molecule)
    s = np.array([0.1234, 0.0567, 0.3891])
    for sg in (2, 14, 15, 61, 88):
        o = settings.transform_ops(_exact_std(sg), np.eye(3), s)
        assert not io_mxt.operator_set_match(o, sg)[0]
        _resolved_same_crystal(o, sg, mol, seed=100 + sg, det=1.0)


def _unimodular_span3(rng, sg):
    """A unimodular K with an entry of 3 under which SYM_OPS[sg] is no longer the same operator set (bounded draw)."""
    for _ in range(10000):
        K = rng.integers(-3, 4, size=(3, 3))
        if round(np.linalg.det(K)) == 1 and np.abs(K).max() == 3:
            o = settings.transform_ops(_exact_std(sg), K.astype(np.float64), np.zeros(3))
            if not io_mxt.operator_set_match(o, sg)[0]:
                return K.astype(np.float64)
    raise RuntimeError(f'no unimodular K moves the operators of sg {sg}')


@pytest.mark.parametrize('sg', [3, 5, 7, 9, 13, 14, 15, 19, 20, 33, 43, 61, 88, 146, 148, 167, 176, 205, 227])
def test_unreduced_basis(sg, chiral_molecule):
    """The standard setting carried into a basis of the same lattice by a unimodular K with an entry of 3 (outside
    the search span; verifier sweep 2026-10-17: 45 of 530 Hall settings resolved before the reduced-basis pass), then
    also given a grid origin shift: resolved, the same crystal. The stored cells of such a basis are long and skewed
    (sg 15 here: lengths up to 194 Angstrom, an angle of 1.5 deg), so float64 literal builds in them carry rounding up
    to 1.3e-9 Angstrom (measured); 1e-8 Angstrom is still seven orders below the wrong-origin control of
    test_class_same_crystal (> 0.1 Angstrom)."""
    rng = np.random.default_rng(sg)
    mol = _mol(chiral_molecule)
    K = _unimodular_span3(rng, sg)
    o = settings.transform_ops(_exact_std(sg), K, np.zeros(3))
    _resolved_same_crystal(o, sg, mol, seed=200 + sg, det=1.0, tol=1e-8)
    o2 = settings.transform_ops(_exact_std(sg), K, K @ np.array([0.25, 0.5, 0.125]))
    _resolved_same_crystal(o2, sg, mol, seed=300 + sg, det=1.0, tol=1e-8)


def test_solve_mod1():
    """The congruence solver: random consistent systems are solved (every row within 1e-9 modulo 1), inconsistent
    ones refused, rank-deficient ones solved with the free unknowns at 0."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        m = int(rng.integers(3, 13))
        M = rng.integers(-2, 3, size=(m, 3))
        y0 = rng.random(3)
        b = (M @ y0 + rng.integers(-3, 4, size=m)) % 1.0
        y = settings._solve_mod1(M, b, 1e-9)
        assert y is not None
        res = M @ y - b
        assert np.abs(res - np.round(res)).max() < 1e-9
    M = np.array([[2, 0, 0], [0, 0, 0]])
    assert settings._solve_mod1(M, np.array([0.5, 0.25]), 1e-9) is None          # 0 = 1/4 mod 1
    y = settings._solve_mod1(M, np.array([0.5, 0.0]), 1e-9)
    assert y is not None and abs(y[0] - 0.25) < 1e-12 and y[1] == 0 and y[2] == 0


def test_passes_after_the_first_do_not_change_its_results():
    """For every class of the file the first pass (grid origins, unreduced basis) already resolves, and the passes
    added after it run only when it fails: the search result equals the first pass's."""
    for label in sorted(CLASSES):
        sg, o = _class_ops(label)
        first = settings._search_pass(o, sg, 1e-6, 'grid')
        r = settings.find_setting(o, sg)
        assert first is not None and np.array_equal(first.P, r.P) and np.array_equal(first.p, r.p), label


# ---------------------------------------------------------------------------
# 5: several settings of one crystal, one canonical form
# ---------------------------------------------------------------------------

def _canonical(d):
    return domains.fold(ops.reduce_cell(d).desc).desc


@pytest.mark.parametrize('sg', [14, 15, 7, 9, 13, 5])
def test_settings_of_one_crystal_have_one_canonical_form(sg, chiral_molecule):
    labels = [k for k, c in CLASSES.items() if c[0] == sg]
    if not labels:
        pytest.skip(f'no class of sg {sg} in the file')
    mol = _mol(chiral_molecule)
    d_std = _random_std(sg, 8, mol, seed=sg)
    ref = _canonical(d_std)
    cls = ops.mono_class(torch.tensor([sg]))
    K = ops._square_neighbourhood(1)
    K = K[ops.mono_parity(K, cls)]
    for label in labels:
        o = _class_ops(label)[1]
        r = settings.find_setting(o, sg)
        for M in (K[0], K[len(K) // 2], K[-1]):
            # describe the crystal in the standard setting through an allowed basis change first, then in the class
            N = ops._embed_ac(M[None])[0]
            s = setting_shift(N.numpy(), sg)
            d_n = ops.change_basis(d_std, N, torch.as_tensor(s, dtype=F64))
            d_s = _to_stored(d_n, mol, r.P, r.p)
            back = settings.re_express(d_s, r.P, r.p)
            can = _canonical(back)
            assert torch.allclose(can.cell, ref.cell, rtol=1e-9, atol=1e-9), label
            assert float(checks.same_crystal(can, ref, mol, cell_rtol=1e-8).max()) < SYN_TOL, label


# ---------------------------------------------------------------------------
# 6: refusals and parameters
# ---------------------------------------------------------------------------

def test_unresolved_and_legacy_codes(chiral_molecule):
    row = _p21c_row(chiral_molecule)
    p21n = next((_class_ops(k)[1] for k, c in CLASSES.items() if c[0] == 14), None)
    wrong_group = _mutated(row, symmetry_operators=settings.std_ops(4))           # P2_1 operators stored as sg 14
    irrational = settings.std_ops(14).copy()
    irrational[1, 0, 3] += 0.1234                                                  # not a setting of anything
    cases = [row, wrong_group, _mutated(row, symmetry_operators=irrational)]
    _, info = io_mxt.from_mol_crystal_data(cases)
    assert info.reason == [None, 'unresolved_setting', 'unresolved_setting'], info.reason
    assert info.setting_reexpressed.tolist() == [False, False, False]
    _, info = io_mxt.from_mol_crystal_data(cases, nonstandard='refuse')
    assert info.reason == [None, 'operator_count_mismatch', 'nonstandard_setting'], info.reason
    if p21n is not None:
        _, info = io_mxt.from_mol_crystal_data([_mutated(row, symmetry_operators=p21n)], nonstandard='refuse')
        assert info.reason == ['nonstandard_setting']
    with pytest.raises(ValueError):
        io_mxt.from_mol_crystal_data(cases, nonstandard='bogus')
    with pytest.raises(ValueError):
        io_mxt.from_mol_crystal_data(cases, mirror_reference='yes')


class _Row(dict):
    """A minimal MolCrystalData-like row: the fields the importer reads, as attributes."""
    __getattr__ = dict.__getitem__

    def keys(self):
        return list(super().keys())


def _mutated(row, **fields):
    r = _Row(row)
    r.update(fields)
    return r


def _row_from_desc(d: CrystalDescription, mol: MoleculeRef, ops_s, dtype=torch.float32) -> _Row:
    """A Z' = 1 row whose stored atoms are the literal asymmetric unit of ``d`` (float32, as the CSD files store them),
    whose pose is re-derived in the importer's aligned frame, and whose stored operators are ``ops_s``."""
    from mxtaltools.crystal_fd.io_mxt import _Units
    from mxtaltools.crystal_building.utils import align_mol_batch_to_standard_axes
    T = d.cell_matrix()[0]
    pos = (T @ d.t[0, 0] + (d.pose[0, 0] @ mol.pos.T).T).to(dtype)
    u = _Units(pos.clone().float(), mol.z, torch.zeros(len(mol.z), dtype=torch.long), 1)
    align_mol_batch_to_standard_axes(u, handedness=None)
    X = u.pos.double()
    Y = pos.double() - pos.double()[mol.z > 1].mean(0)
    U, _, Vt = torch.linalg.svd(X.T @ Y)
    Pz = Vt.T @ U.T
    R, h = so3.split_o3(Pz[None])
    c = torch.linalg.solve(T, pos.double()[mol.z > 1].mean(0))
    return _Row(sg_ind=torch.tensor(int(d.sg[0])), z_prime=torch.tensor([1]),
                cell_lengths=d.cell[0, :3].to(dtype)[None], cell_angles=d.cell[0, 3:].to(dtype)[None],
                aunit_centroid=c.to(dtype)[None], aunit_orientation=so3.matrix_to_rotvec(R).to(dtype),
                aunit_handedness=h.reshape(1, 1).to(dtype), pos=pos, z=mol.z.clone(),
                symmetry_operators=np.asarray(ops_s, dtype=np.float64), identifier='SYNTH')


def _p21c_row(chiral_molecule):
    mol = _mol(chiral_molecule)
    return _row_from_desc(_random_std(14, 1, mol, seed=11), mol, settings.std_ops(14))


def test_importer_on_synthetic_nonstandard_rows(chiral_molecule):
    """Synthetic float32 rows, one per class: imported, (P, p) recorded, the same crystal as the standard description
    they were written from (float32 storage: 1e-4 Angstrom)."""
    if not CLASSES:
        pytest.skip('no classes')
    mol = _mol(chiral_molecule)
    rows, stds = [], []
    for label in sorted(CLASSES):
        sg, o = _class_ops(label)
        r = settings.find_setting(o, sg)
        d_std = _random_std(sg, 1, mol, seed=sum(map(ord, label)) + 1)
        d_s = _to_stored(d_std, mol, r.P, r.p)
        d_s = d_s.replace(t=d_s.t - torch.floor(d_s.t))
        rows.append(_row_from_desc(d_s, mol, o))
        stds.append((d_std, r))
    desc, info = io_mxt.from_mol_crystal_data(rows, sheet=None, special_tol=0.0)
    assert info.imported.all(), [x for x in zip(sorted(CLASSES), info.reason) if x[1] is not None]
    assert info.setting_reexpressed.all()
    for j, (d_std, r) in enumerate(stds):
        assert np.array_equal(info.setting_P[j].numpy(), r.P) and np.array_equal(info.setting_p[j].numpy(), r.p)
        assert torch.allclose(desc.cell[j], d_std.cell[0], rtol=1e-5)
        lit = checks.unit_cell_fractional(desc[j], info.mols[j])[0]
        ref = checks.unit_cell_fractional(d_std, mol)[0]
        dev = _set_dev(lit.reshape(-1, 3), info.mols[j].z.repeat(lit.shape[0]), ref.reshape(-1, 3),
                       mol.z.repeat(ref.shape[0]), desc[j].cell_matrix()[0])
        assert dev < 1e-3, (sorted(CLASSES)[j], dev)


# ---------------------------------------------------------------------------
# 7: the mini CSD file, default import against the importer before D1 and D2
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def prot():
    return _load(DATA / 'mini_new_prot_csd.pt')


def _stored_atoms_dev(row, desc_row, mol, P, p) -> float:
    """Angstrom: the stored atoms (Z' = 1) expanded by the row's stored operators in the stored cell, against the
    description built with SYM_OPS[sg] (molecules matched both ways, atoms in their stored order)."""
    o = torch.as_tensor(np.asarray(row.symmetry_operators), dtype=F64).reshape(-1, 4, 4)
    cell_s = torch.cat([torch.as_tensor(row.cell_lengths).reshape(3), torch.as_tensor(row.cell_angles).reshape(3)])
    Ts = ops.cell_matrices(cell_s.double()[None])
    f_s = torch.linalg.solve(Ts[0], torch.as_tensor(row.pos, dtype=F64).T).T
    A_s = _images(f_s, o[:, :3, :3], o[:, :3, 3])[None]
    A_n = checks.unit_cell_fractional(desc_row, mol)
    P = torch.as_tensor(P, dtype=F64)
    p = torch.as_tensor(p, dtype=F64)
    return float(torch.maximum(_one_way(A_n @ P.T + p, A_s, Ts),
                               _one_way((A_s - p) @ torch.linalg.inv(P).T, A_n, desc_row.cell_matrix())))


def test_mini_file_default_against_legacy(prot):
    d_new, i_new = io_mxt.from_mol_crystal_data(prot)
    d_old, i_old = io_mxt.from_mol_crystal_data(prot, **LEGACY)
    jn = {r: j for j, r in enumerate(i_new.row.tolist())}
    jo = {r: j for j, r in enumerate(i_old.row.tolist())}
    n_re, n_mir = 0, 0
    for i in range(len(prot)):
        if i in jo:                                                   # imported before D1: unchanged
            assert i in jn and not bool(i_new.setting_reexpressed[i])
            a, b = jn[i], jo[i]
            mir = bool(i_new.mirrored[a])
            n_mir += mir
            for f in ('sg', 'cell', 't', 'R'):
                assert torch.equal(getattr(d_new, f)[a], getattr(d_old, f)[b]), (i, f)
            assert torch.equal(d_new.h[a], -d_old.h[b] if mir else d_old.h[b])
            flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64) if mir else torch.ones(3, dtype=F64)
            assert float((i_new.mols[a].pos - i_old.mols[b].pos * flip).abs().max()) < 1e-12
            assert np.array_equal(i_new.setting_P[a].numpy(), np.eye(3))
        elif i_old.reason[i] in ('nonstandard_setting', 'operator_count_mismatch'):
            assert bool(i_new.setting_reexpressed[i]), i
            if i_new.reason[i] is None:
                n_re += 1
                a = jn[i]
                assert not np.array_equal(i_new.setting_P[a].numpy(), np.eye(3)) or \
                    not np.array_equal(i_new.setting_p[a].numpy(), np.zeros(3))
                dev = _stored_atoms_dev(prot[i], d_new[a], i_new.mols[a], i_new.setting_P[a], i_new.setting_p[a])
                assert dev < float(i_new.pose_dev[i]) + 1e-5, (i, dev)
        else:
            assert i_new.reason[i] == i_old.reason[i], i
    assert n_re >= 5 and n_mir >= 5, (n_re, n_mir)


def test_mini_reexpressed_rows_against_cif(prot):
    if not CIF_DIR.exists():
        pytest.skip(f'{CIF_DIR} not on this machine')
    desc, info = io_mxt.from_mol_crystal_data(prot)
    n, worst = 0, 0.0
    for j, i in enumerate(info.row.tolist()):
        if not bool(info.setting_reexpressed[i]):
            continue
        dev = _cif_dev(info.identifier[i], desc[j], info.mols[j], info.setting_P[j], info.setting_p[j])
        if dev is None:
            continue
        n += 1
        worst = max(worst, dev)
    if n == 0:
        pytest.skip('no CIF of a re-expressed row on disk')
    assert n >= 5 and worst < CIF_TOL, (n, worst)


# ---------------------------------------------------------------------------
# 8: real rows of every class (fixture written by the full-file run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fixture_rows():
    return _load(FIXTURE)


def test_real_rows_of_every_class(fixture_rows):
    rows = fixture_rows['rows']
    sel = [k for k, kind in enumerate(fixture_rows['kind']) if kind == 'class_imported']
    desc, info = io_mxt.from_mol_crystal_data([rows[k] for k in sel])
    assert info.imported.all(), [(fixture_rows['note'][sel[i]]['ident'], r) for i, r in enumerate(info.reason) if r]
    assert info.setting_reexpressed.all() and float(info.metric_dev.max()) <= 1e-6
    every = {fixture_rows['class_key'][k] for k, kind in enumerate(fixture_rows['kind'])
             if kind in ('class_imported', 'class_refused')}
    assert every == set(range(len(fixture_rows['classes'])))           # every class has real rows here
    assert len(fixture_rows['classes']) == len(CLASSES)
    worst_s, worst_c, n_cif = 0.0, 0.0, 0
    for j in range(desc.num_rows):
        row = rows[sel[j]]
        dev = _stored_atoms_dev(row, desc[j], info.mols[j], info.setting_P[j], info.setting_p[j])
        worst_s = max(worst_s, dev - float(info.pose_dev[j]))
        if CIF_DIR.exists():
            c = _cif_dev(str(row.identifier), desc[j], info.mols[j], info.setting_P[j], info.setting_p[j])
            if c is not None:
                worst_c, n_cif = max(worst_c, c), n_cif + 1
    assert worst_s < 1e-5, worst_s
    if CIF_DIR.exists():
        assert n_cif >= 0.9 * desc.num_rows and worst_c < CIF_TOL, (n_cif, worst_c)
    # the canonicaliser takes the re-expressed rows as it takes standard ones
    tm = desc.sg <= 15
    if bool(tm.any()):
        red = ops.reduce_cell(desc[torch.nonzero(tm)[:, 0]])
        domains.fold(red.desc)
    if bool((~tm).any()):
        domains.fold(desc[torch.nonzero(~tm)[:, 0]])


def _stored_metric_dev(row) -> float:
    """The stored cell against the stored rotations, in the stored setting (as ``io_mxt.metric_deviation``)."""
    o = np.asarray(row.symmetry_operators, dtype=np.float64).reshape(-1, 4, 4)
    cell = torch.cat([torch.as_tensor(row.cell_lengths).reshape(3), torch.as_tensor(row.cell_angles).reshape(3)])
    G = ops.metric_tensor(cell.double()[None])[0]
    W = torch.as_tensor(np.round(o[:, :3, :3]), dtype=F64)
    n = torch.sqrt(torch.diagonal(G))
    return float(((W.transpose(1, 2) @ G @ W - G).abs() / (n[:, None] * n[None])).max())


def test_real_rows_refused_for_other_reasons_keep_them(fixture_rows):
    """Rows of a class refused in the full-file run are refused for the same reason: Z' > 1 (D3), or a stored cell
    that the stored operators do not preserve (``setting_metric_mismatch``, checked here in the stored setting)."""
    rows = fixture_rows['rows']
    sel = [k for k, kind in enumerate(fixture_rows['kind']) if kind == 'class_refused']
    if not sel:
        pytest.skip('no refused class rows in the fixture')
    _, info = io_mxt.from_mol_crystal_data([rows[k] for k in sel])
    assert [r for r in info.reason] == [fixture_rows['note'][k]['reason'] for k in sel]
    assert set(info.reason) <= {'z_prime_gt1', 'setting_metric_mismatch'}, set(info.reason)
    for i, k in enumerate(sel):
        if info.reason[i] == 'setting_metric_mismatch':
            assert float(info.metric_dev[i]) > 1e-6 and _stored_metric_dev(rows[k]) > 1e-6
        else:
            assert bool(info.setting_reexpressed[i])


def test_setting_metric_mismatch(chiral_molecule):
    """A re-expressed row whose stored cell its stored operators do not preserve is refused; the same row with a
    consistent cell imports; a standard row with the same defect is measured (metric_dev), not refused."""
    label = next((k for k, c in CLASSES.items() if k.startswith('14|P 21/c 1 1')), None)
    if label is None:
        pytest.skip("no 'P 21/c 1 1' class")
    sg, o = _class_ops(label)
    mol = _mol(chiral_molecule)
    r = settings.find_setting(o, sg)
    d_std = _random_std(sg, 1, mol, seed=21)
    d_s = _to_stored(d_std, mol, r.P, r.p)
    d_s = d_s.replace(t=d_s.t - torch.floor(d_s.t))
    good = _row_from_desc(d_s, mol, o)
    cell = d_s.cell.clone()
    cell[0, 4] += math.radians(0.05)                                 # a-unique setting with beta off 90 deg
    bad = _row_from_desc(d_s.replace(cell=cell), mol, o)
    std_cell = d_std.cell.clone()
    std_cell[0, 3] += 1e-4                                            # a standard P2_1/c row with alpha off 90 deg
    std_bad = _row_from_desc(d_std.replace(cell=std_cell, t=d_std.t - torch.floor(d_std.t)), mol,
                             settings.std_ops(sg))
    _, info = io_mxt.from_mol_crystal_data([good, bad, std_bad], special_tol=0.0)
    assert info.reason == [None, 'setting_metric_mismatch', None], info.reason
    assert float(info.metric_dev[0]) < 1e-6 < float(info.metric_dev[1])
    assert float(info.metric_dev[2]) > 5e-5
    _, info = io_mxt.from_mol_crystal_data([bad], special_tol=0.0, metric_tol=1e-2)
    assert info.reason == [None]


# ---------------------------------------------------------------------------
# 9: the mirrored reference (D2)
# ---------------------------------------------------------------------------

def _check_mirrored(rows):
    d_new, i_new = io_mxt.from_mol_crystal_data(rows)
    d_off, i_off = io_mxt.from_mol_crystal_data(rows, mirror_reference=False)
    assert torch.equal(i_new.row, i_off.row)
    off = i_off.sohncke & ~i_off.on_sheet
    assert torch.equal(i_new.mirrored, off & (d_off.z_prime == 1))
    assert bool(i_new.on_sheet.all()) and bool((d_new.h[:, 0] == 1).all())
    assert bool((d_off.h[off, 0] == -1).all()) and not bool(i_off.mirrored.any())
    flip = torch.tensor([-1.0, 1.0, 1.0], dtype=F64)
    n = 0
    for j in torch.nonzero(i_new.mirrored)[:, 0].tolist():
        assert torch.allclose(i_new.mols[j].pos, i_off.mols[j].pos * flip, atol=1e-12, rtol=0)
        assert i_new.mols[j].hash != i_off.mols[j].hash
        a = checks.unit_cell_fractional(d_new[j], i_new.mols[j])
        b = checks.unit_cell_fractional(d_off[j], i_off.mols[j])
        assert float((a - b).abs().max()) < 1e-12
        n += 1
    return n, d_new, i_new


def test_mirrored_reference_mini(prot):
    n, d_new, i_new = _check_mirrored(prot)
    assert n >= 5, n
    _, info_none = io_mxt.from_mol_crystal_data(prot, sheet=None)
    assert not bool(info_none.mirrored.any())
    if CIF_DIR.exists():
        for j in torch.nonzero(i_new.mirrored)[:, 0].tolist():
            i = int(i_new.row[j])
            dev = _cif_dev(i_new.identifier[i], d_new[j], i_new.mols[j], torch.eye(3, dtype=F64),
                           torch.zeros(3, dtype=F64))
            assert dev is None or dev < CIF_TOL, (i_new.identifier[i], dev)


def test_mirrored_reference_fixture(fixture_rows):
    rows = fixture_rows['rows']
    sel = [k for k, kind in enumerate(fixture_rows['kind']) if kind == 'mirrored']
    n, _, _ = _check_mirrored([rows[k] for k in sel])
    assert n == len(sel)


def _orient_std(pos, z, handedness):
    """GFN's frame step on one molecule: MolData, collate, ``orient_molecule(mode='std', target_handedness=...)``
    (float32, as the trainer runs it); handedness None is the call without ``target_handedness``."""
    from mxtaltools.dataset_utils.data_classes import MolData
    from mxtaltools.dataset_utils.utils import collate_data_list
    b = collate_data_list([MolData(z=z.clone(), pos=pos.float().clone(), x=torch.zeros(len(z), 1),
                                   do_mol_analysis=True)])
    th = None if handedness is None else torch.tensor([[float(handedness)]])
    b.orient_molecule(mode='std', target_handedness=th)
    return b.pos.double()


def _check_reference_handedness(rows):
    """Every imported row's reference is left in place by the frame step at its ``reference_handedness``; a mirrored
    reference X' taken through it without ``target_handedness`` comes back as X (the unmirrored aligned molecule)."""
    _, info = io_mxt.from_mol_crystal_data(rows)
    _, off = io_mxt.from_mol_crystal_data(rows, mirror_reference=False)
    assert torch.equal(info.row, off.row)
    assert info.reference_handedness.dtype == torch.long
    assert torch.equal(info.reference_handedness, torch.where(info.mirrored, -1, 1))
    worst_fixed, worst_to_x, moved, n_mir = 0.0, 0.0, 0, 0
    for j, m in enumerate(info.mols):
        fixed = _orient_std(m.pos, m.z, int(info.reference_handedness[j]))
        worst_fixed = max(worst_fixed, float((fixed - m.pos).norm(dim=1).max()))
        if bool(info.mirrored[j]):
            n_mir += 1
            default = _orient_std(m.pos, m.z, None)
            worst_to_x = max(worst_to_x, float((default - off.mols[j].pos).norm(dim=1).max()))
            moved += float((default - m.pos).norm(dim=1).max()) > 1.0
    assert worst_fixed < 1e-3, worst_fixed                          # float32 storage: about 1e-5 Angstrom
    assert worst_to_x < 1e-3, worst_to_x
    return n_mir, moved


def test_reference_handedness_keeps_the_reference_mini(prot):
    n_mir, moved = _check_reference_handedness(prot)
    assert n_mir >= 5 and moved >= 0.9 * n_mir, (n_mir, moved)


def test_reference_handedness_keeps_the_reference_fixture(fixture_rows):
    rows = fixture_rows['rows']
    sel = [k for k, kind in enumerate(fixture_rows['kind']) if kind in ('mirrored', 'standard', 'class_imported')]
    n_mir, moved = _check_reference_handedness([rows[k] for k in sel])
    assert n_mir >= 50 and moved >= 0.9 * n_mir, (n_mir, moved)


def test_zp2_sohncke_row_is_not_mirrored(prot):
    """A synthetic Z' = 2 row of P2_1 (unit 2 = unit 1 moved by (0, 0, 1/2), as a 2-unit cell of the same crystal would
    not be, but a valid row for the importer) with both units at h = -1: imported with allow_zp_gt1, left off the
    sheet, not mirrored."""
    _, info = io_mxt.from_mol_crystal_data(prot, sheet=None, **LEGACY)
    j = next((j for j, i in enumerate(info.row.tolist()) if int(info.sg[i]) == 4 and int(info.h_input[j, 0]) == -1),
             None)
    if j is None:
        pytest.skip('no P2_1 row at h = -1 in the mini file')
    row = prot[int(info.row[j])]
    r = row.clone()
    T = torch.as_tensor(row.T_fc, dtype=torch.float32).reshape(3, 3)
    shift = T @ torch.tensor([0.0, 0.0, 0.5])
    n = int(row.num_atoms)
    r.pos = torch.cat([row.pos, row.pos + shift])
    r.z = torch.cat([row.z, row.z])
    r.x = torch.cat([row.x, row.x])
    r.mol_ind = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    r.num_atoms = torch.tensor(2 * n)
    r.z_prime = torch.tensor([2])
    c, o, h = row.aunit_centroid.clone(), row.aunit_orientation.clone(), row.aunit_handedness.clone()
    c[0, 3:6] = torch.remainder(c[0, :3] + torch.tensor([0.0, 0.0, 0.5]), 1.0)
    o[0, 3:6] = o[0, :3]
    h[0, 1] = h[0, 0]
    r.aunit_centroid, r.aunit_orientation, r.aunit_handedness = c, o, h
    desc, info2 = io_mxt.from_mol_crystal_data([r], allow_zp_gt1=True, special_tol=0.0)
    assert info2.reason == [None], info2.reason
    assert not bool(info2.mirrored.any()) and not bool(info2.on_sheet[0]) and bool((desc.h[0] == -1).all())
