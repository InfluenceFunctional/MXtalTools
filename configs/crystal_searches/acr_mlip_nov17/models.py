"""acr_mlip_nov17: acridine P2_1/c Z'=1 streaming minima searches, one per MLIP. What the searches share: the models,
the molecule per model, the stages and the paths. make_configs.py and run.py import it; `python3 models.py` (standard
library only, for the job script) prints one line per model: name, kind, extra PYTHONPATH ('-' for none).

The models, checkpoints and gas-phase conformers are acr_fm_oct27's (campaign.py there, which built and uploaded them on
2026-10-27); this folder replaces that campaign's coordinator, hop and curator machinery with the streaming search.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = HERE.name
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
OUT = f'{DATA}/campaigns/{NAME}'  # every run's outputs and seed files
SEED_POOL = HERE.parent / 'acr_fm_oct27' / 'seed_pool.pt'  # 21,882 distinct states of the earlier acridine landscapes
KNOWN = HERE.parent / 'acr_zp1_sep30' / 'known_forms.pt'  # ACRDIN04, ACRDIN12 (cell parameters)
SG = 14
KT = 2.494339  # RT at 300 K, kJ/mol: both routes report the lattice energy in kJ/mol per molecule

# kind: the stages' optim_target; path / local: the checkpoint on the cluster / the dev box; mol: the model's gas-phase
# conformer in DATA (local copy in D:/crystal_datasets/acridine), mol_sha1 as acr_fm_oct27's build_conformer.py wrote it;
# pyextra: put first on PYTHONPATH (the container's fairchem 2.13 cannot build UMA 1.2), fairchem: the version needed.
MODELS = {
    'maceoff24m': dict(kind='mace', path='/scratch/mk8347/models/MACE-OFF24_medium.model',
                       local='D:/crystal_datasets/MACE-OFF24_medium.model', mol='acr_maceoff24m_conformer.pt',
                       mol_sha1='3197928f1207a77378e2ae4528c4bc1af5aac5f1', pyextra='', fairchem=''),
    'umas1p1': dict(kind='uma', path='/scratch/mk8347/models/uma/uma-s-1p1.pt',
                    local='D:/crystal_datasets/uma-s-1p1.pt', mol='acr_umas1p1_conformer.pt',
                    mol_sha1='de020984ccf1d8b5f702d402e0dde2b6e1dcf854', pyextra='', fairchem=''),
    'umas1p2': dict(kind='uma', path='/scratch/mk8347/models/uma/uma-s-1p2p1.pt',
                    local='D:/crystal_datasets/uma-s-1p2p1.pt', mol='acr_umas1p2_conformer.pt',
                    mol_sha1='af06c3e76b677bffe58ae8ae64e2a4c7397e1fe5', pyextra='/scratch/mk8347/pylibs_fc216',
                    fairchem='2.16.0'),
}

N_RANDOM = 2000  # random starts at most per search (the stop rule usually ends it sooner); raise and resubmit to extend
SLOTS = 96  # starts in flight: UMA-s peaked at 70.7 GB on an A100 at 96 rows (acr_fm_oct27 calibration)
OPT_SEED = 1_000_000_000  # one seed for every model: the same random starts and the same seed order
STOP_P = 0.01  # Good-Turing bound on a new minimum within STOP_BAND_KT of the lowest, for the seeds and the starts
STOP_BAND_KT = 4.0
MIN_CONTACT = 1.5  # A: cfg:min_intermolecular_contact (MACE-OFF24's interpenetrated hole, acr_fm_oct27 calibration)


def stages(kind):
    """The exp6_recall_nov10 stream stages (its make_configs.py in gfn_diffusion eval/exp6_recall): Rprop, then
    per-row BFGS with the energy cascade, both at a 10 A cutoff under the model."""
    s1 = dict(optim_target=kind, enforce_reduced=False, compression_factor=0.0, cutoff=10, init_lr=0.05,
              convergence_eps=0.01, optimizer_func='rprop', anneal_lr=False, grad_norm_clip=0.1, show_tqdm=False,
              max_num_steps=350, target_packing_coeff=None, centroid_boundary='wrap')
    s2 = dict(optim_target=kind, enforce_reduced=False, compression_factor=0.0, cutoff=10, convergence_eps=1e-4,
              optimizer_func='bfgs', show_tqdm=False, max_num_steps=200, target_packing_coeff=None,
              centroid_boundary='wrap', early_stop=[[25, 20.0], [50, 15.0], [100, 12.0]], early_stop_kT=KT)
    return [s1, s2]


def config(model, run, seeds, num_samples, stop_p, out_dir=OUT, mol_dir=DATA, model_path=None):
    """One search config (a dict): run `run` under `model`, seeds from the file `seeds` (None: none), `num_samples`
    random starts at most, `stop_p` (None: every seed and start is relaxed)."""
    m = MODELS[model]
    c = dict(device='cuda', mol_path=f"{mol_dir}/{m['mol']}", dataset_path=None, target_path=None, out_dir=out_dir,
             run_name=run, target_identifier=None, save_trajs=False, init_sample_method='random', init_reduced=True,
             force_restart_run=False, mol_seed=0, opt_seed=OPT_SEED, sampling_mode='all', mols_to_sample=1,
             num_samples=int(num_samples), sgs_to_search=[SG], zp_to_search=[1], batch_size=SLOTS,
             grow_batch_size=False, init_target_cp='std', oom_ceiling=False, min_intermolecular_contact=MIN_CONTACT,
             minima=dict(kT=KT, capture_d2=9.0),
             # basin_set off: acridine's third moment about one principal axis is ~0 (3e-4), so crystal_fd's importer
             # re-derives a frame turned 180 deg for some rows and refuses them (pose_mismatch, 9.22 A; 4 of 8 minima
             # in the UMA-s 1.1 smoke, 2026-11-17); <run>_minima.pt holds every minimum
             stream=dict(slots=SLOTS, check_every=5, seeds=seeds, stop_p=stop_p, stop_band_kT=STOP_BAND_KT,
                         basin_set=False),
             opt=stages(m['kind']))
    path = model_path or m['path']
    c['mace_predictor_path'] = path if m['kind'] == 'mace' else None
    c['uma_predictor_path'] = path if m['kind'] == 'uma' else None
    if m['kind'] == 'uma':
        c['uma_activation_checkpointing'] = False  # on, the relaxation's backward raised CheckpointError (oct27)
        # UMA is not reflection-equivariant: a mirror-mean gas reference; P2_1/c has an improper element, so a crystal
        # is its own mirror image and the periodic energy needs no mirror average (exp6_recall_nov10's rule)
        c['mlip_gas_reference'] = 'mirror_mean'
        c['mlip_mirror_average'] = False
    return c


if __name__ == '__main__':
    for k, m in MODELS.items():
        print(k, m['kind'], m['pyextra'] or '-')
