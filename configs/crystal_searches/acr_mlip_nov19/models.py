"""acr_mlip_nov19: acridine P2_1/c Z'=1 streaming searches in the endpoint mode, one per MLIP, recording the descents'
trajectories for band sets (training priors). What the searches share: the models, the molecule per model, the stages
and the paths. make_configs.py and run.py import it; `python3 models.py` (standard library only, for the job script)
prints one line per model: name, kind, extra PYTHONPATH ('-' for none).

The endpoint recipe is pool8_nov09's (minima_stream/mipcas_elj_balls.yaml): Rprop, then per-row BFGS that stops on its
force (fmax), each endpoint its own minimum, minima identified by FD-metric balls, no polish; the stream records every
descent's states. acr_mlip_nov17 (minima mode, no trajectories) supplied the seed pool's first rows.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = HERE.name
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
OUT = f'{DATA}/campaigns/{NAME}'  # every run's outputs and seed files
SEED_POOL = HERE / 'seed_pool.pt'  # build_seed_pool.py: acr_mlip_nov17's minima, then acr_fm_oct27's 21,882 states
KNOWN = HERE.parent / 'acr_zp1_sep30' / 'known_forms.pt'  # ACRDIN04, ACRDIN12 (cell parameters)
SG = 14
KT = 2.494339  # RT at 300 K, kJ/mol: every route reports the lattice energy in kJ/mol per molecule

# kind: the stages' optim_target; path / local: the checkpoint on the cluster / the dev box; mol: the model's gas-phase
# conformer in DATA (local copy in D:/crystal_datasets/acridine), built by acr_fm_oct27/build_conformer.py; mol_sha1 its
# SHA1, None for a conformer run.py builds on the cluster (UMA-m: the dev box lacks the host memory); pyextra: put first
# on PYTHONPATH (the container's fairchem 2.13 cannot build UMA 1.2; it has no torch-dftd, pylibs_dftd holds torch-dftd
# 0.5.3, pure Python, copied from the dev-box venv), fairchem: the version needed; slots: starts in
# flight (UMA-s peaked at 70.7 GB on an A100 at 96 rows, acr_fm_oct27 calibration; UMA-m is larger: 32, not measured);
# dispersion: the MACE route's D3 (MACE-CSP-25.2 is PBE without dispersion).
MODELS = {
    'maceoff24m': dict(kind='mace', path='/scratch/mk8347/models/MACE-OFF24_medium.model',
                       local='D:/crystal_datasets/MACE-OFF24_medium.model', mol='acr_maceoff24m_conformer.pt',
                       mol_sha1='3197928f1207a77378e2ae4528c4bc1af5aac5f1', pyextra='', fairchem='', slots=96),
    'umas1p1': dict(kind='uma', path='/scratch/mk8347/models/uma/uma-s-1p1.pt',
                    local='D:/crystal_datasets/uma-s-1p1.pt', mol='acr_umas1p1_conformer.pt',
                    mol_sha1='de020984ccf1d8b5f702d402e0dde2b6e1dcf854', pyextra='', fairchem='', slots=96),
    'umas1p2': dict(kind='uma', path='/scratch/mk8347/models/uma/uma-s-1p2p1.pt',
                    local='D:/crystal_datasets/uma-s-1p2p1.pt', mol='acr_umas1p2_conformer.pt',
                    mol_sha1='af06c3e76b677bffe58ae8ae64e2a4c7397e1fe5', pyextra='/scratch/mk8347/pylibs_fc216',
                    fairchem='2.16.0', slots=96),
    'umam1p1': dict(kind='uma', path='/scratch/mk8347/models/uma/uma-m-1p1.pt',
                    local=None, mol='acr_umam1p1_conformer.pt', mol_sha1=None, pyextra='', fairchem='', slots=32),
    'esens': dict(kind='uma', path='/scratch/mk8347/models/uma/esen_s.pt',
                  local='D:/crystal_datasets/esen_s.pt', mol='acr_esens_conformer.pt',
                  mol_sha1='3b7294e0f716daeff36a2ad082d80d8dbe0b2e50', pyextra='', fairchem='', slots=96),
    'macecspl': dict(kind='mace', path='/scratch/mk8347/models/day_models/mace_csp_25.2_large.model',
                     local='D:/crystal_datasets/day_models/mace_csp_25.2_large.model', mol='acr_macecspl_conformer.pt',
                     mol_sha1='395b91fcf6bf2ad085e30acf24fa0e9618488cea', pyextra='/scratch/mk8347/pylibs_dftd', fairchem='', slots=96,
                     dispersion=dict(xc='pbe', damping='bj')),
}

N_RANDOM = 2000  # random starts at most per search (the stop rule usually ends it sooner); raise and resubmit to extend
OPT_SEED = 1_000_000_000  # one seed for every model: the same random starts and the same seed order
STOP_P = 0.01  # Good-Turing bound on a new minimum within STOP_BAND_KT of the lowest, for the seeds and the starts
STOP_BAND_KT = 4.0
MIN_CONTACT = 1.5  # A: cfg:min_intermolecular_contact (MACE-OFF24's interpenetrated hole, acr_fm_oct27 calibration)
BALL_A = 0.2  # A: cfg:minima capture_A and merge_A (pool8)
TRAJ = dict(every=1, band_kT=15.0, net_A=0.1)  # cfg:stream.trajectories (pool8)
FMAX = 0.02  # kT / A: the BFGS stage's force test (pool8)
ENERGY_NOISE = 1e-3  # kT: a step within this energy change passes the slope test. Not measured for these MLIPs: MACE's
# float32 node sums jitter by ~0.003 kJ/mol (1.2e-3 kT) under a rigid translation (acr_foundation_oct26)


def stages(kind):
    """pool8's endpoint stages (minima_stream/mipcas_elj_balls.yaml) under the model: Rprop, then per-row BFGS to the
    force test with the energy cascade, both at a 10 A cutoff."""
    s1 = dict(optim_target=kind, enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.05,
              convergence_eps=0.01, optimizer_func='rprop', anneal_lr=False, grad_norm_clip=0.1, show_tqdm=False,
              max_num_steps=350, target_packing_coeff=None, centroid_boundary='wrap')
    s2 = dict(optim_target=kind, enforce_reduced=False, compression_factor=0.0, cutoff=10, fmax=FMAX,
              energy_noise=ENERGY_NOISE, optimizer_func='bfgs', show_tqdm=False, max_num_steps=600,
              target_packing_coeff=None, centroid_boundary='wrap', early_stop=[[25, 10.0], [50, 8.0], [100, 5.0]],
              early_stop_kT=KT)
    return [s1, s2]


def config(model, run, seeds, num_samples, stop_p, out_dir=OUT, mol_dir=DATA, model_path=None):
    """One search config (a dict): run `run` under `model`, seeds from the file `seeds` (None: none), `num_samples`
    random starts at most, `stop_p` (None: every seed and start is relaxed)."""
    m = MODELS[model]
    c = dict(device='cuda', mol_path=f"{mol_dir}/{m['mol']}", dataset_path=None, target_path=None, out_dir=out_dir,
             run_name=run, target_identifier=None, save_trajs=False, init_sample_method='random', init_reduced=True,
             force_restart_run=False, mol_seed=0, opt_seed=OPT_SEED, sampling_mode='all', mols_to_sample=1,
             num_samples=int(num_samples), sgs_to_search=[SG], zp_to_search=[1], batch_size=m['slots'],
             grow_batch_size=False, init_target_cp='std', oom_ceiling=False, min_intermolecular_contact=MIN_CONTACT,
             minima=dict(kT=KT, polish='none', capture_A=BALL_A, merge_A=BALL_A),
             # basin_set off: band sets are written after the search from <run>_traj.pt, so the band's settings are
             # chosen when the prior is built, not here
             stream=dict(slots=m['slots'], check_every=5, seeds=seeds, stop_p=stop_p, stop_band_kT=STOP_BAND_KT,
                         basin_set=False, trajectories=dict(TRAJ)),
             opt=stages(m['kind']))
    path = model_path or m['path']
    c['mace_predictor_path'] = path if m['kind'] == 'mace' else None
    c['uma_predictor_path'] = path if m['kind'] == 'uma' else None
    if m.get('dispersion'):
        c['mace_dispersion'] = dict(m['dispersion'])
    if m['kind'] == 'uma':
        # on, the relaxation's backward raised CheckpointError (oct27). No parity keys: acridine is planar (its
        # mirror image is itself; UMA-s gas vs mirror 0.003-0.004 kJ/mol, 2026-11-19) and P2_1/c has an improper
        # element, so a crystal is its own mirror image
        c['uma_activation_checkpointing'] = False
    return c


if __name__ == '__main__':
    for k, m in MODELS.items():
        print(k, m['kind'], m['pyextra'] or '-')
