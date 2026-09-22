"""
MIPCAS sg2 Z'=1 ELJ arms under the triclinic Niggli penalty (local). Same pattern as ../mipcas/make_configs.py: one arm
per opt_seed, run_name <stem>_<arm> so load_search_chunks collates on the all-digit tail.

The triclinic Niggli penalty is always on in mxtaltools (it was an env switch when these arms ran).

    python make_configs.py
    for i in 0 1 2 3 4; do python ../../../mxtaltools/crystal_search/run_search.py --config mipcas_niggli_elj_$i.yaml; done
"""
from copy import deepcopy
from pathlib import Path

import yaml

BASE_PATH = 'mipcas_niggli.yaml'
N_ARMS = 5
ARM_SAMPLES = 1000
BATCH_SIZE = 500  # smoke: batch 200 peaks at 2.6 GB on the 16.3 GB card with ~2.7 GB taken by the desktop


def main():
    base = yaml.safe_load(Path(BASE_PATH).read_text())
    for q in Path('.').glob('mipcas_niggli_elj_*.yaml'):
        q.unlink()
    for arm in range(N_ARMS):
        cfg = deepcopy(base)
        cfg['opt_seed'] = arm
        cfg['num_samples'] = ARM_SAMPLES
        cfg['batch_size'] = BATCH_SIZE
        cfg['run_name'] = f"{base['run_name']}_{arm}"
        Path(f"{cfg['run_name']}.yaml").write_text(yaml.dump(cfg, default_flow_style=False))
    print(f"wrote {N_ARMS} arms x {ARM_SAMPLES} = {N_ARMS * ARM_SAMPLES} starts")


if __name__ == '__main__':
    main()
