"""
qm9_full_sep29: the search that builds the full-QM9 conditional prior. Random-start eLJ searches, SG2 (P-1), Z'=1, on
every standardized QM9 molecule, with the current search schedule. One chunk family, qm9_cluster_mols_chunk<k>.pt:
  chunks 0-49   195 molecules each: the set the current prior was built from
  chunks 50-    780 each, the last holding the remainder: the rest of the pool, written by gfn-diffusion
                energy_sampling/prep_qm9_anchor_mols.py --exclude (command in README.md)

What each array task runs (run_search.py, sampling_mode 'all'): its number of random starts for EVERY molecule of one
chunk file.
  chunks 0-49, seed 0   -> STARTS (20) per molecule; the current prior has 10
  chunks 50-, seed 0    -> REST_STARTS (10) per molecule
  chunk 0, seeds 1-3    -> +60 each, so 200 per molecule on chunk 0's 195 molecules
Seeds are independent draws (distinct opt_seed), so a molecule's starts are the union over its seeds. A later top-up is
more seeds on the same chunks, and chunk 0's 200 starts, subsampled, give how often n starts miss a molecule's lowest
minimum for any n up to 200.

Schedule: acr_finish_sep27/0.yaml (the campaign no-wall wrap schedule) with optim_target elj; centroid_boundary wrap,
no reduction wall (outputs are NOT in the reduced cell: standardize_cells before use); convergence_eps x10, as in
zp1_sep28, because the base thresholds predate MXtalTools fe283818. NO early_stop cascade: its reference is the lowest
energy of the whole run, and one run here holds up to 780 molecules, so the deepest molecule would retire every other
molecule's walkers. Energy: raw eLJ (the searcher does not stamp lj_coeff), init_target_cp 'std', init_reduced.

    python make_battery.py     # writes tasks/<i>.yaml, INDEX.tsv and the job script's array range; asserts the checks
Outputs on the cluster: <OUT>/qm9full_c<k>_<seed>.pt (+ progress/owner side files); gfn-diffusion
energy_sampling/data_processing/utils.py::load_search_chunks(<OUT>, 'qm9full_c<k>') gathers one chunk's seeds.
"""
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'
SBATCH = HERE / 'submit_qm9_full.sbatch'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional'
MOL_PATH = DATA + '/priors/qm9_cluster_mols_chunk{k}.pt'
OUT_DIR = DATA + '/anchors/qm9_full_sep29'
# chunk sizes as prep_qm9_anchor_mols.py wrote them: 50 of 195, then REST_MOLS in chunks of 780
REST_MOLS = 120_560
CHUNK_MOLS = [195] * 50 + [min(780, REST_MOLS - 780 * j) for j in range(-(-REST_MOLS // 780))]
STARTS = 20                                  # random starts per molecule, chunks 0-49
REST_STARTS = 10                             # random starts per molecule, chunks 50 onward
SATURATION = [(1, 60), (2, 60), (3, 60)]     # chunk 0's extra (seed, starts)
SEED_BASE = 3_000_000_000       # a task's batches use opt_seed + 1e4 * batch_idx; tasks are 1e6 apart
CONCURRENT = 16                 # array throttle: tasks running at once
TASKS = ([(0, 0, STARTS)] + [(0, s, n) for s, n in SATURATION]
         + [(k, 0, STARTS if k < 50 else REST_STARTS) for k in range(1, len(CHUNK_MOLS))])
RELAX_PER_S = 12.0              # measured on tasks 0-52 (job 18835847), job start to end; for the estimate only


def task_cfg(k, s, n):
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('mace_predictor_path', None)
    cfg.update(device='cuda', mol_path=MOL_PATH.format(k=k), dataset_path=None, target_path=None,
               target_identifier=None, out_dir=OUT_DIR, run_name=f'qm9full_c{k}_{s}', save_trajs=False,
               uma_predictor_path=None, init_sample_method='random', init_reduced=True, init_target_cp='std',
               force_restart_run=False, mol_seed=0, opt_seed=SEED_BASE + (100 * k + s) * 1_000_000,
               sampling_mode='all', mols_to_sample=0, num_samples=n, sgs_to_search=[2],
               zp_to_search=[1], batch_size=2000, grow_batch_size=True, oom_ceiling=True)
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
    tasks_dir = HERE / 'tasks'
    tasks_dir.mkdir(exist_ok=True)
    for stale in tasks_dir.glob('*.yaml'):
        stale.unlink()
    rows = ['task\tchunk\tseed\tmolecules\tstarts\trelaxations\trun_name\topt_seed\tmol_path']
    cfgs = []
    for i, (k, s, n) in enumerate(TASKS):
        cfg = task_cfg(k, s, n)
        (tasks_dir / f'{i}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        rows.append(f"{i}\t{k}\t{s}\t{CHUNK_MOLS[k]}\t{n}\t{CHUNK_MOLS[k] * n}\t{cfg['run_name']}\t{cfg['opt_seed']}\t"
                    f"{cfg['mol_path']}")
        cfgs.append(cfg)
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    text = SBATCH.read_text(encoding='utf-8')
    text, hits = re.subn(r'(?m)^#SBATCH --array=\S+$', f'#SBATCH --array=0-{len(TASKS) - 1}%{CONCURRENT}', text)
    assert hits == 1, f'{SBATCH.name}: expected one #SBATCH --array line, found {hits}'
    SBATCH.write_text(text, encoding='utf-8', newline='\n')

    # every task must write its own run and draw its own starts, or tasks overwrite or duplicate each other
    assert len({c['run_name'] for c in cfgs}) == len(cfgs), 'run_name collision'
    assert len({c['opt_seed'] for c in cfgs}) == len(cfgs), 'opt_seed collision'
    assert all(s < 100 for _, s, _ in TASKS), 'a seed >= 100 would take the next chunk\'s opt_seed'
    # batch seeds step 1e4 inside a task's 1e6 slot: fewer than 100 batches even if OOM halving took the batch to 200
    assert max(CHUNK_MOLS[k] * n for k, _, n in TASKS) <= 100 * 200, 'a task could run into the next opt_seed slot'
    for c in cfgs:
        for trail, v in _paths(c):
            # a drive letter, a backslash or a relative data path would resolve on the dev box and nowhere else
            assert ':' not in v[:3] and '\\' not in v, f"local path at {c['run_name']}{trail}: {v!r}"
        for key in ('mol_path', 'out_dir'):
            assert c[key].startswith('/scratch/'), (c['run_name'], key, c[key])
    relax = sum(CHUNK_MOLS[k] * n for k, _, n in TASKS)
    print(f"wrote {len(cfgs)} task configs to {tasks_dir}, INDEX.tsv, and --array=0-{len(TASKS) - 1}%{CONCURRENT} "
          f"in {SBATCH.name}")
    print(f"  {len(CHUNK_MOLS)} chunks, {sum(CHUNK_MOLS):,} molecules; starts per molecule {STARTS} on chunks 0-49, "
          f"{REST_STARTS} on chunks 50-{len(CHUNK_MOLS) - 1}, {STARTS + sum(n for _, n in SATURATION)} on chunk 0")
    print(f"  {relax:,} relaxations; {relax / RELAX_PER_S / 3600:,.0f} GPU-hours at {RELAX_PER_S:g} per s; longest task "
          f"{max(CHUNK_MOLS[k] * n for k, _, n in TASKS):,} relaxations")


if __name__ == '__main__':
    main()
