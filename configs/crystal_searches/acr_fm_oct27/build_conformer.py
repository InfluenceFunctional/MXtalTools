"""The acridine gas-phase conformer under one energy model, for a campaign under that model (the universal-conformer rule:
one conformer per model, its gas-phase minimum, used everywhere: searches, priors, training).

    python build_conformer.py mace|uma MODEL_PATH LABEL [--out_dir D:/crystal_datasets/acridine] [--write]

Generalises acr_zp1_sep30's build (D:/crystal_datasets/acridine/acr_newmodel_conformer_build/): BFGS to fmax 1e-4 eV/A
on CPU from opt_acridine_conformer.pt (N first: the atom order of every search state and seed file), MACE in float64,
UMA with task omc and no periodicity (the gas leg the lattice energy subtracts). The same relaxation from the old
compressed conformer (acridine_conformer.pt, another atom order) must reach the same energy. The relaxed positions are
Kabsch-aligned onto opt's so the molecule keeps its frame; radius and mol_volume are recomputed, every other field copied
from opt_acridine_conformer.pt. --write saves <out_dir>/acr_<LABEL>_conformer.pt.
"""
import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS

EV = 96.485332
SRC = 'D:/crystal_datasets/acridine/opt_acridine_conformer.pt'
OLD = 'D:/crystal_datasets/acridine/acridine_conformer.pt'


def calculator(kind, path):
    if kind == 'mace':
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=path, device='cpu', default_dtype='float64')
    torch.serialization.add_safe_globals([slice])
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    return FAIRChemCalculator(pretrained_mlip.load_predict_unit(path, device='cpu'), task_name='omc')


def relaxed(m, calc):
    a = Atoms(numbers=m.z.long().numpy(), positions=m.pos.double().numpy(), pbc=False)
    a.calc = calc
    ok = BFGS(a, logfile=None).run(fmax=1e-4, steps=3000)
    return a, bool(ok), float(a.get_potential_energy()), float(np.abs(a.get_forces()).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['mace', 'uma'])
    ap.add_argument('path')
    ap.add_argument('label')
    ap.add_argument('--out_dir', default='D:/crystal_datasets/acridine')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    calc = calculator(a.kind, a.path)
    m = torch.load(SRC, weights_only=False)
    at, ok, e, f = relaxed(m, calc)
    at2, ok2, e2, f2 = relaxed(torch.load(OLD, weights_only=False), calc)
    print(f'{a.label}: from opt E {e:.6f} eV (converged {ok}, max |F| {f:.1e}); from the old compressed conformer '
          f'{e2:.6f} eV (converged {ok2}); difference {(e2 - e) * EV:+.4f} kJ/mol', flush=True)
    old = m.pos.double().numpy()
    P, Q = at.positions - at.positions.mean(0), old - old.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    aligned = P @ R + old.mean(0)
    print(f'RMSD to opt_acridine_conformer after alignment {np.sqrt(((aligned - old) ** 2).sum(1).mean()):.4f} A')
    c = m.clone()
    c.pos = torch.tensor(aligned, dtype=m.pos.dtype)
    c.radius, c.mol_volume = None, None
    c.radius = c.radius_calculation()
    t = c.clone()
    t.z = t.z.long()  # the stored z is float; the volume lookup indexes by it
    c.mol_volume = t.volume_calculation()
    c.identifier = f'acridine_{a.label}_gas_min'
    print(f'radius {float(m.radius):.4f} -> {float(c.radius):.4f} A; mol_volume {float(m.mol_volume):.3f} -> '
          f'{float(c.mol_volume):.3f} A^3')
    if a.write:
        out = os.path.join(a.out_dir, f'acr_{a.label}_conformer.pt')
        torch.save(c, out)
        print('wrote', out, os.path.getsize(out), 'bytes')


if __name__ == '__main__':
    main()
