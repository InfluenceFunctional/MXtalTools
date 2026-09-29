"""
qm9_dense_sep29: a pilot of the full-QM9 prior search. Dense random-start eLJ searches, SG2 (P-1), Z'=1, on the
first four chunks of the standardized QM9 molecule set the current conditional prior was built from
(qm9_cluster_mols_chunk{0..3}.pt, 195 molecules each), with the current search schedule.

What each array task runs (run_search.py, sampling_mode 'all'): 50 random starts for EVERY molecule of one chunk file.
  chunks 0-3, seed 0       -> 50 starts per molecule on 780 molecules
  chunk 0,    seeds 1-3    -> 200 starts per molecule on chunk 0's 195 molecules
Seeds are independent draws (distinct opt_seed), so a molecule's starts are the union over its seeds. The same
molecules carry 10 starts each in the current prior (qm9_anchors / qm9c100k), which is the comparison the pilot is for:
the per-molecule minimum as a function of the number of starts.

Schedule: acr_finish_sep27/0.yaml (the campaign no-wall wrap schedule) with optim_target elj; centroid_boundary wrap,
no reduction wall (outputs are NOT in the reduced cell: standardize_cells before use); convergence_eps x10, as in
zp1_sep28, because the base thresholds predate MXtalTools fe283818. NO early_stop cascade: its reference is the lowest
energy of the whole run, and one run here holds 195 molecules, so the deepest molecule would retire every other
molecule's walkers. Energy: raw eLJ (the searcher does not stamp lj_coeff), init_target_cp 'std', init_reduced.

    python make_battery.py     # writes tasks/<i>.yaml, INDEX.tsv; asserts the checks below
Outputs on the cluster: <OUT>/qm9dense_c<k>_<seed>.pt (+ progress/owner side files); load_search_chunks
('qm9dense_c<k>') gathers one chunk's seeds.
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'
DATA = '/scratch/mk8347/data/crystal_datasets/conditional'
MOL_PATH = DATA + '/priors/qm9_cluster_mols_chunk{k}.pt'
OUT_DIR = DATA + '/anchors/qm9_dense_sep29'
STARTS_PER_TASK = 50            # random starts per molecule per task
SEED_BASE = 2_000_000_000       # a task's batches use opt_seed + 1e4 * batch_idx; tasks are 1e6 apart
TASKS = [(k, 0) for k in range(4)] + [(0, s) for s in range(1, 4)]


def task_cfg(k, s):
    cfg = yaml.safe_load(BASE.read_text())
    cfg.pop('mace_predictor_path', None)
    cfg.update(device='cuda', mol_path=MOL_PATH.format(k=k), dataset_path=None, target_path=None,
               target_identifier=None, out_dir=OUT_DIR, run_name=f'qm9dense_c{k}_{s}', save_trajs=False,
               uma_predictor_path=None, init_sample_method='random', init_reduced=True, init_target_cp='std',
               force_restart_run=False, mol_seed=0, opt_seed=SEED_BASE + (100 * k + s) * 1_000_000,
               sampling_mode='all', mols_to_sample=0, num_samples=STARTS_PER_TASK, sgs_to_search=[2],
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
    (HERE / 'tasks').mkdir(exist_ok=True)
    rows = ['task\tchunk\tseed\trun_name\topt_seed\tmol_path']
    cfgs = []
    for i, (k, s) in enumerate(TASKS):
        cfg = task_cfg(k, s)
        (HERE / 'tasks' / f'{i}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        rows.append(f"{i}\t{k}\t{s}\t{cfg['run_name']}\t{cfg['opt_seed']}\t{cfg['mol_path']}")
        cfgs.append(cfg)
    (HERE / 'INDEX.tsv').write_text('\n'.join(rows) + '\n', encoding='utf-8')

    # every task must write its own run and draw its own starts, or tasks overwrite or duplicate each other
    assert len({c['run_name'] for c in cfgs}) == len(cfgs), 'run_name collision'
    assert len({c['opt_seed'] for c in cfgs}) == len(cfgs), 'opt_seed collision'
    for c in cfgs:
        for trail, v in _paths(c):
            # a drive letter, a backslash or a relative data path would resolve on the dev box and nowhere else
            assert ':' not in v[:3] and '\\' not in v, f"local path at {c['run_name']}{trail}: {v!r}"
        for key in ('mol_path', 'out_dir'):
            assert c[key].startswith('/scratch/'), (c['run_name'], key, c[key])
    per_mol = {}
    for k, s in TASKS:
        per_mol[k] = per_mol.get(k, 0) + STARTS_PER_TASK
    print(f"wrote {len(cfgs)} task configs to {HERE / 'tasks'} and INDEX.tsv")
    print('  starts per molecule by chunk:', per_mol, '(195 molecules per chunk)')
    print(f"  relaxations per task: 195 x {STARTS_PER_TASK} = {195 * STARTS_PER_TASK}; "
          f"total {195 * STARTS_PER_TASK * len(cfgs):,}")
    print(f"  submit: sbatch --array=0-{len(cfgs) - 1} submit_qm9_dense.sbatch")


if __name__ == '__main__':
    main()
