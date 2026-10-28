"""acr_fm_oct27: what the acridine foundation-model campaigns share -- the models, the molecule per model, the relaxation
schedule and the paths. make_campaign.py, calibrate.py and preflight.py import it; `python3 campaign.py` (standard
library only, for the shell launchers) prints one line per model: name, kind, checkpoint, conformer, extra PYTHONPATH.

One campaign per model (P2_1/c, Z'=1), each under its own gas-phase conformer (the universal-conformer rule: the
model's gas-phase minimum, build_conformer.py), in <ROOT>/<model>. Every campaign re-relaxes the same seed list
(seed_pool.pt, build_seed_pool.py, plus the known forms of acr_zp1_sep30/known_forms.pt) and runs random starts and hops
under acr_zp1_sep30's coordinator settings.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = HERE.name
DATA = '/scratch/mk8347/data/crystal_datasets/acridine'
ROOT = f'{DATA}/campaigns/{NAME}'  # campaign of model m: ROOT/m; calibration of model m: ROOT/calibration/m
KNOWN = HERE.parent / 'acr_zp1_sep30' / 'known_forms.pt'  # ACRDIN04, ACRDIN12 (cell parameters)
BASE = HERE.parent / 'acr_finish_sep27' / '0.yaml'  # the no-wall wrap schedule of the acridine campaigns
KT = 2.494

# kind: the search's optim_target and the coordinator's energy_key; path: the cluster file; local: the same file on the
# dev box (make_campaign computes the energy model id from it; launch.sh and preflight.py compare the cluster file's);
# mol: the model's gas-phase conformer in DATA, mol_sha1 as written by build_conformer.py on 2026-10-27; pyextra: a
# directory put first on PYTHONPATH in every job (the container's fairchem 2.13 cannot build UMA 1.2: HydraModel rejects
# its model_id argument; 2.16, pure Python, pip install --no-deps --target), fairchem: the version the jobs must import.
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

# stage 3 polishes, to convergence_eps / 100 and up to 800 steps, every row that ends stage 2 within POLISH_KT of the
# job's lowest final energy (early_stop at its step 0 retires the rest; the first batch, with no reference yet, polishes
# all): acr_zp1_sep30's unconverged end states split one minimum into many basins (a tight re-relaxation put 27 of 36
# sampled basins on their structure's minimum and all 8 sampled one-state basins on known minima). POLISH_KT covers the
# registry window (3 kT) with 0.5 kT to spare.
POLISH_KT = 3.5
POLISH_STEPS = 800
CASCADE = [[25, 10.0], [50, 8.0], [100, 5.0]]  # stage 2 of random and hops: acr_zp1_sep30's
# cfg:min_intermolecular_contact (Angstrom) of every search: a relaxed row whose closest intermolecular atom pair is nearer
# is dropped before it is saved, written to a shard or sets the early_stop reference (run_search.drop_contact_clashes)
MIN_CONTACT = 1.5

# array tasks per stream and starting batch, per model (sized from the calibration's throughput; make_campaign.py and
# preflight.py read them, launch.sh through the generated <model>/model.env)
TASKS = {m: dict(random=12, hops=12, seeded=8) for m in MODELS}
BATCH = {m: 96 for m in MODELS}


def stages(kind, cascade=True, polish_cascade=True):
    """The relaxation schedule: acr_finish_sep27's two stages (convergence_eps x10, as acr_zp1_sep30), then the polish
    stage. cascade: stage 2's energy cascade (random and hops; a seed is relaxed to the end of stage 2);
    polish_cascade: polish only the rows within POLISH_KT (False polishes every row: calibration anchors)."""
    import yaml
    opt = yaml.safe_load(BASE.read_text())['opt']
    for st in opt:
        assert st['centroid_boundary'] == 'wrap' and st['enforce_reduced'] is False
        st.update(optim_target=kind, show_tqdm=False)
        # the base thresholds were tuned when ema_trajectory returned 0.1 x the smoothed trajectory (before
        # MXtalTools fe283818); x10 keeps relaxations stopping where the earlier acridine campaigns' did
        st['convergence_eps'] = 10 * float(st['convergence_eps'])
    polish = dict(opt[-1], convergence_eps=float(opt[-1]['convergence_eps']) / 100, max_num_steps=POLISH_STEPS)
    if cascade:
        opt[-1].update(early_stop=CASCADE, early_stop_kT=KT)
    if polish_cascade:
        polish.update(early_stop=[[0, POLISH_KT]], early_stop_kT=KT)
    return opt + [polish]


def predictor_paths(model):
    """The search config's model keys. uma_activation_checkpointing False: with it on, the relaxation's backward through
    UMA raised torch's CheckpointError (UMA-s 1.1 and 1.2, fairchem 2.13 and 2.16, CPU, 2026-10-27); off, the batch
    size controller (grow_batch_size, oom_ceiling) absorbs the extra activation memory."""
    m = MODELS[model]
    return dict(mace_predictor_path=m['path'] if m['kind'] == 'mace' else None,
                uma_predictor_path=m['path'] if m['kind'] == 'uma' else None, uma_activation_checkpointing=False)


def mol_path(model):
    return f"{DATA}/{MODELS[model]['mol']}"


if __name__ == '__main__':
    for k, m in MODELS.items():
        print(k, m['kind'], m['path'], mol_path(k), m['pyextra'] or '-')
