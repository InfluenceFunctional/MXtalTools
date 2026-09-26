#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sep27_fin
#SBATCH --output=/scratch/mk8347/logs/acr_finish_sep27-%A_%a.out

# acr_finish_sep27: 4 jobs, acridine sg14 Z'=2 (MACE): two no-reduction random-start arms under wrap (0, 1), and a
# finish pass over acr_proposals_sep27 wrap end states: stage 2 continued with and without the reduction penalty (2, 3).
# Array index -> <index>.yaml. A finish job first snapshots its source runs into DATA_ROOT/sep27_finish/ with
# build_finish_seeds.py (the first 1000 rows of each random run, all rows of each seeded run); it stops with a message
# if a random source has fewer rows yet. Needs run_search's run_files fix.
# See README.md / MANIFEST.md in this directory.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine
CONFDIR=${MXT_ROOT}/configs/crystal_searches/acr_finish_sep27

CONFIG=${CONFDIR}/${SLURM_ARRAY_TASK_ID}.yaml
if [ ! -f "${CONFIG}" ]; then
    echo "no config for array index ${SLURM_ARRAY_TASK_ID} in ${CONFDIR}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi
if ! grep -q "def run_files" ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py; then
    echo "run_search.py lacks the stale-intermediates/resume fix (run_files): git pull MXtalTools first" >&2
    exit 1
fi

START=$(date +%s)
TASKDIR=$(mktemp -d)  # scratch working directory (run_search's side files are named per run)
echo "JOB ${SLURM_ARRAY_TASK_ID} config=$(basename ${CONFIG}) start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH

        # (1) finish jobs: snapshot the source end states into their seed file (once, shared by both);
        # (2) skip a complete config (exit 3), resume a partial one. Any other failure stops the job loudly.
        python - <<'PY'
import sys, os, json, yaml, torch
cfg = yaml.safe_load(open('${CONFIG}'))
dp = cfg.get('dataset_path')
if cfg.get('init_sample_method') == 'data':
    sys.path.insert(0, '${CONFDIR}')
    from build_finish_seeds import ensure
    ensure('wrap', cfg['out_dir'], cfg['mol_path'], dp)
want = cfg['num_samples']
if cfg.get('init_sample_method') == 'data':
    want = min(want, len(torch.load(dp, weights_only=False, map_location='cpu')))
stem = os.path.join(cfg['out_dir'], cfg['run_name'])
if os.path.exists(stem + '_progress.json'):
    cur = json.load(open(stem + '_progress.json'))['cursor']
    if cur >= want:
        print(f'SKIP {cfg[\"run_name\"]}: cursor {cur} >= {want}')
        sys.exit(3)
    print(f'RESUME {cfg[\"run_name\"]}: cursor {cur} of {want}')
else:
    print(f'FRESH {cfg[\"run_name\"]} (target {want})')
sys.exit(0)
PY
        rc=\$?
        if [ \$rc -eq 3 ]; then exit 0; fi
        if [ \$rc -ne 0 ]; then echo 'finish-seed snapshot or pre-check failed' >&2; exit \$rc; fi

        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
RC=$?
rm -rf "${TASKDIR}"
END=$(date +%s)
echo "JOB ${SLURM_ARRAY_TASK_ID} config=$(basename ${CONFIG}) status=${RC} seconds=$((END-START))"
exit ${RC}
