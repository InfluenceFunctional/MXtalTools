#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=56-111
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=aug21_sg14zp2
#SBATCH --output=/scratch/mk8347/logs/acr_rerun_aug21/slurm-%A_%a.out

# sg14 Z'=2 slice of the acr_rerun_aug21 battery, on GPU.
# Array indices 56-111 are exactly the sg14/Z'=2 arms:
#     56-75   aug21base        (20 arms, 2000 samples each)
#     76-85   aug21stdnocomp   (10)
#     86-95   aug21lightcomp   (10)
#     96-105  aug21lowlr       (10)
#     106-111 aug21seed        (6, ~281 seeds each -- much shorter)
#
# --time=06:00:00, NOT the 4h used for the benchmark. Measured there: a 2000-sample
# arm takes ~3.8-4.0h on an A100, and 6 of 10 benchmark arms were killed at the 4h
# wall with 88-96% done. Six hours leaves headroom for the slow tail (one benchmark
# arm was still in stage 1 after 5.5 minutes of wall per 150 steps, hitting the
# "cell size to cutoff is very large" path).
#
# --output is pinned here rather than left to default, so 56 arms cannot scatter
# their logs across whatever directory sbatch happened to be run from.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine
CONFIG=${MXT_ROOT}/configs/crystal_searches/acr_rerun_aug21/${SLURM_ARRAY_TASK_ID}.yaml

if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi

TASKDIR=$(mktemp -d)
trap 'rm -rf "${TASKDIR}"' EXIT

START=$(date +%s)
echo "ARM ${SLURM_ARRAY_TASK_ID} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH

        # SKIP ALREADY-COMPLETE ARMS.
        # run_search resumes a PARTIAL arm on its own (force_restart_run is false:
        # it loads the existing output and sets cursor = len(opt_outs)), so a
        # timed-out arm just continues. But a FULLY complete arm would enter the
        # loop with cursor == num_samples and slice an empty batch, which is not
        # one of the exceptions its handler catches. So check first and exit 0.
        python - <<'PY' || exit 0
import sys, os, yaml, torch
cfg = yaml.safe_load(open('${CONFIG}'))

# The target is NOT simply num_samples. For a seeded arm init_samples_to_optim
# clips it to the size of the seed shard -- the configs carry num_samples 2000
# while a shard holds ~282 -- so comparing against 2000 would never skip a
# finished seeded arm. It would resume with cursor already at the end and slice
# an empty batch, which run_search's handler does not catch.
want = cfg['num_samples']
if cfg.get('init_sample_method') == 'data':
    seeds = torch.load(cfg['dataset_path'], weights_only=False, map_location='cpu')
    want = min(want, len(seeds))

p = os.path.join(cfg['out_dir'], cfg['run_name'] + '.pt')
if os.path.exists(p):
    n = len(torch.load(p, weights_only=False, map_location='cpu'))
    if n >= want:
        print(f'SKIP {cfg[\"run_name\"]}: already has {n} >= {want} samples')
        sys.exit(1)          # non-zero -> the '|| exit 0' above skips the arm
    print(f'RESUME {cfg[\"run_name\"]}: {n} of {want} samples on disk')
else:
    print(f'FRESH {cfg[\"run_name\"]} (target {want})')
sys.exit(0)
PY

        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
STATUS=$?

END=$(date +%s)
echo "ARM ${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
