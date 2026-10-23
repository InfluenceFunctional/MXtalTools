"""
rc_oct22: random-start eLJ searches, SG2 (P-1), Z'=1, on molecules whose geometry is ONE RANDOM CONFORMER per SMILES
(gfn-diffusion energy_sampling/data_processing/random_conformers.py: ETKDG embedding, every rotor spun uniformly, kept
only if valid) in place of a DFT geometry, for the conditional prior. One chunk family per molecule source, lists of
standardized MolData written by gfn-diffusion energy_sampling/prep_qm9_anchor_mols.py, chunk k =
<family>_mols_chunk<k>.pt:
  qm9rc    the qm9full molecules (the current prior's SMILES), single-source geometries
  csdrc    CSD molecules and fragments                        } new SMILES, no QM9 molecule among them, each molecule
  zincrc   ZINC20 (MW row A) molecules and fragments          } in one family only (gfn-diffusion energy_sampling/
  gdbrc    GDB-11 molecules and fragments                     } data_processing/smiles_pool.py, then
  pubrc    PubChem molecules and fragments                    } smiles_mixture.py)
Each family's list was shuffled before chunking, so any run of its chunks is a random sample of it, and the share of
each source in the searched set is chosen by how many of its chunks are submitted. Chunk sizes are read from
chunks.tsv, which --scan writes from the chunk files themselves (on the dev box, before upload).

CPU ONLY. What each array task runs (run_search.py, sampling_mode 'all', device cpu, one core): STARTS_PER_TASK (2)
random starts for EVERY molecule of one chunk file, in fixed batches of BATCH (50); each chunk is SEEDS (5) tasks with
distinct seeds, so every molecule gets 10 starts in all, gathered over the seeds of its chunk. Small batches because
the eLJ route writes out every crystal's image cluster (about 12,000 atoms on QM9-sized molecules): memory grows by
about 16-20 MB per crystal in the batch over a 0.8 GB base. Tasks are ordered family by family, chunk by chunk, seed by
seed, so a family is a contiguous index range and an installment is a set of index ranges. An array index names a task: once INDEX.tsv exists, a rewrite must keep every
row it holds (checked; --renumber overrides, only before anything is submitted). Every family needs chunks unless
--allow-missing names it. A later top-up (more seeds on the same chunks) has to be appended after the last row.

Schedule, as qm9_full_sep29: acr_finish_sep27/0.yaml with optim_target elj; centroid_boundary wrap, no reduction wall
(outputs are NOT in the reduced cell: standardize_cells before use); convergence_eps x10; no early_stop cascade (its
reference is the lowest energy of the whole run, which holds up to 2,000 molecules); raw eLJ; init_target_cp 'std',
init_reduced.

    python make_battery.py --scan D:\\crystal_datasets\\conditional\\rc_oct22\\mols   # chunks.tsv, then as below
    python make_battery.py     # writes tasks/<i>.yaml, INDEX.tsv and the job script's array range; asserts the checks
Outputs on the cluster: <OUT>/<family>_c<k>_<seed>.pt (+ progress/owner side files).
"""
import argparse
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'
SBATCH = HERE / 'submit_rc_oct22.sbatch'
CHUNKS = HERE / 'chunks.tsv'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional'
MOL_PATH = DATA + '/rc_oct22/mols/{fam}_mols_chunk{k}.pt'
OUT_DIR = DATA + '/anchors/rc_oct22'
FAMILIES = ['qm9rc', 'csdrc', 'zincrc', 'gdbrc', 'pubrc']          # task order
SEEDS = 5                       # tasks per chunk, each with its own seed
STARTS_PER_TASK = 2             # random starts per molecule per task: 10 per molecule over the seeds
BATCH = 50                      # crystals relaxed together on one core (fixed; no growth on CPU)
FAMILY_SLOT = {f: i for i, f in enumerate(FAMILIES)}  # opt_seed: a family's task slots never meet another family's
SEED_BASE = 100_000_000_000     # above every qm9_full_sep29 seed; a task's batches use opt_seed + 1e4 * batch_idx
CONCURRENT = 250                # array throttle per submitted array: tasks running at once
ARRAY_MAX = 1000                # Slurm's default MaxArraySize is 1001: tasks from 1000 on run with TASK_OFFSET
RELAX_PER_CORE_S = 0.75         # one core, dev box, batch 50, 200 relaxations of qm9rc chunk 0; for the estimate only


def read_chunks():
    """{family: {chunk: molecules}} from chunks.tsv."""
    out = {f: {} for f in FAMILIES}
    header, *lines = CHUNKS.read_text(encoding='utf-8').splitlines()
    assert header.split('\t')[:3] == ['family', 'chunk', 'molecules'], header
    for line in lines:
        fam, k, n = line.split('\t')[:3]
        out[fam][int(k)] = int(n)
    for fam, ks in out.items():
        assert sorted(ks) == list(range(len(ks))), f'{fam}: chunk numbers are not 0..n-1'
    return out


def keep_old_rows(rows, renumber):
    """An array index names a task: every row of an existing INDEX.tsv must survive a rewrite unchanged."""
    path = HERE / 'INDEX.tsv'
    if renumber or not path.exists():
        return
    old = path.read_text(encoding='utf-8').splitlines()
    changed = [i for i, row in enumerate(old) if i >= len(rows) or rows[i] != row]
    assert not changed, (f'INDEX.tsv rows {changed[:5]} would change or vanish; an array index names a task '
                         f'(pass --renumber only if nothing has been submitted)')


def scan(mol_dir):
    """chunks.tsv from the chunk files: each family's chunk numbers and molecule counts."""
    import torch
    rows = ['family\tchunk\tmolecules\tdistinct_smiles\tfile']
    for fam in FAMILIES:
        files = {}
        for p in Path(mol_dir).glob(f'{fam}_mols_chunk*.pt'):
            m = re.fullmatch(rf'{fam}_mols_chunk(\d+)\.pt', p.name)
            if m:
                files[int(m.group(1))] = p
        assert sorted(files) == list(range(len(files))), f'{fam}: chunk files are not 0..n-1 in {mol_dir}'
        seen = set()
        for k in sorted(files):
            mols = torch.load(files[k], map_location='cpu', weights_only=False)
            ids = [str(m.identifier) for m in mols]
            assert not seen & set(ids), f'{files[k].name}: a molecule is also in an earlier chunk'
            seen |= set(ids)
            rows.append(f'{fam}\t{k}\t{len(mols)}\t{len({str(m.smiles) for m in mols})}\t{files[k].name}')
        print(f'{fam}: {len(files)} chunks, {len(seen):,} molecules')
    CHUNKS.write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')


def task_cfg(fam, k, s, n):
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('mace_predictor_path', None)
    cfg.update(device='cpu', mol_path=MOL_PATH.format(fam=fam, k=k), dataset_path=None, target_path=None,
               target_identifier=None, out_dir=OUT_DIR, run_name=f'{fam}_c{k}_{s}', save_trajs=False,
               uma_predictor_path=None, init_sample_method='random', init_reduced=True, init_target_cp='std',
               force_restart_run=False, mol_seed=0,
               opt_seed=SEED_BASE + (FAMILY_SLOT[fam] * 1_000_000 + 100 * k + s) * 1_000_000,
               sampling_mode='all', mols_to_sample=0, num_samples=n, sgs_to_search=[2],
               zp_to_search=[1], batch_size=BATCH, grow_batch_size=False, oom_ceiling=False)
    for st in cfg['opt']:
        assert st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False
        st['optim_target'] = 'elj'
        st['show_tqdm'] = False
        st['convergence_eps'] = 10 * float(st['convergence_eps'])
        assert 'early_stop' not in st and 'keep_lowest_fraction' not in st
    return cfg


def _paths(node, trail=''):
    if isinstance(node, dict):
        for key, v in node.items():
            yield from _paths(v, f'{trail}.{key}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _paths(v, f'{trail}[{i}]')
    elif isinstance(node, str):
        yield trail, node


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', type=Path, default=None, help='write chunks.tsv from the chunk files in this directory')
    ap.add_argument('--allow-missing', nargs='*', default=[], help='families allowed to have no chunks')
    ap.add_argument('--renumber', action='store_true', help='rewrite INDEX.tsv rows (only before anything is submitted)')
    args = ap.parse_args()
    if args.scan:
        scan(args.scan)
    chunks = read_chunks()
    missing = [f for f in FAMILIES if not chunks[f] and f not in args.allow_missing]
    assert not missing, f'families with no chunks: {missing} (a failed generation?); --allow-missing to proceed'
    tasks = [(fam, k, s, STARTS_PER_TASK) for fam in FAMILIES for k in sorted(chunks[fam]) for s in range(SEEDS)]

    rows = ['task\tfamily\tchunk\tseed\tmolecules\tstarts\trelaxations\trun_name\topt_seed\tmol_path']
    cfgs = []
    for i, (fam, k, s, n) in enumerate(tasks):
        cfg = task_cfg(fam, k, s, n)
        m = chunks[fam][k]
        rows.append(f"{i}\t{fam}\t{k}\t{s}\t{m}\t{n}\t{m * n}\t{cfg['run_name']}\t{cfg['opt_seed']}\t{cfg['mol_path']}")
        cfgs.append(cfg)
    keep_old_rows(rows, args.renumber)
    tasks_dir = HERE / 'tasks'
    tasks_dir.mkdir(exist_ok=True)
    for stale in tasks_dir.glob('*.yaml'):
        stale.unlink()
    for i, cfg in enumerate(cfgs):
        (tasks_dir / f'{i}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    text = SBATCH.read_text(encoding='utf-8')
    last = min(len(tasks), ARRAY_MAX) - 1
    text, hits = re.subn(r'(?m)^#SBATCH --array=\S+$', f'#SBATCH --array=0-{last}%{CONCURRENT}', text)
    assert hits == 1, f'{SBATCH.name}: expected one #SBATCH --array line, found {hits}'
    SBATCH.write_text(text, encoding='utf-8', newline='\n')

    # every task must write its own run and draw its own starts, or tasks overwrite or duplicate each other
    assert len({c['run_name'] for c in cfgs}) == len(cfgs), 'run_name collision'
    assert len({c['opt_seed'] for c in cfgs}) == len(cfgs), 'opt_seed collision'
    assert all(s < 100 for _, _, s, _ in tasks), 'a seed >= 100 would take the next chunk\'s opt_seed'
    assert all(100 * k + 99 < 1_000_000 for _, k, _, _ in tasks), 'a chunk number would take the next family\'s seeds'
    # batch seeds step 1e4 inside a task's 1e6 slot: a task may run at most 100 batches of BATCH
    assert max(chunks[f][k] * n for f, k, _, n in tasks) <= 100 * BATCH, 'a task could run into the next opt_seed slot'
    for c in cfgs:
        for trail, v in _paths(c):
            # a drive letter, a backslash or a relative data path would resolve on the dev box and nowhere else
            assert ':' not in v[:3] and '\\' not in v, f"local path at {c['run_name']}{trail}: {v!r}"
        for key in ('mol_path', 'out_dir'):
            assert c[key].startswith('/scratch/'), (c['run_name'], key, c[key])
    print(f"wrote {len(cfgs)} task configs to {tasks_dir}, INDEX.tsv, and --array=0-{last}%{CONCURRENT} "
          f"in {SBATCH.name}")
    for fam in FAMILIES:
        idx = [i for i, t in enumerate(tasks) if t[0] == fam]
        if not idx:
            continue
        relax = sum(chunks[fam][k] * n for f, k, _, n in tasks if f == fam)
        print(f"  {fam}: tasks {idx[0]}-{idx[-1]}, {len(chunks[fam])} chunks x {SEEDS} seeds, "
              f"{sum(chunks[fam].values()):,} molecules, {SEEDS * STARTS_PER_TASK} starts each; {relax:,} relaxations, "
              f"{relax / RELAX_PER_CORE_S / 3600:,.0f} core-hours at {RELAX_PER_CORE_S:g} per s per core")
    longest = max(chunks[f][k] * n for f, k, _, n in tasks)
    print(f"  longest task {longest:,} relaxations, {longest / RELAX_PER_CORE_S / 3600:.1f} h at {RELAX_PER_CORE_S:g} per s")


if __name__ == '__main__':
    main()
