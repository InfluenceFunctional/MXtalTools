#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-19
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=mipcas_nig
#SBATCH --output=/scratch/mk8347/logs/mipcas_nig_sep17/slurm-%A_%a.out

# mipcas_nig_sep17: array 0-9 ELJ, 10-19 UMA (MANIFEST.md), each arm 5000 starts, triclinic Niggli penalty.
# A timed-out arm resumes on resubmission (force_restart_run false); a complete arm is skipped.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
OUT_DIR=/scratch/mk8347/data/crystal_datasets/mipcas/nig_sep17
CONFIG=${MXT_ROOT}/configs/crystal_searches/mipcas_nig_sep17/${SLURM_ARRAY_TASK_ID}.yaml

if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi
mkdir -p "${OUT_DIR}"

# run_search writes opt_intermediates.pt into the CWD; an ephemeral CWD per task keeps arms apart.
TASKDIR=$(mktemp -d)
trap 'rm -rf "${TASKDIR}"' EXIT

START=$(date +%s)
echo "ARM ${SLURM_ARRAY_TASK_ID} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH

        # skip a complete arm (a resumed complete arm would slice an empty batch, which run_search does not catch)
        python - <<'PY' || exit 0
import sys, os, yaml, torch
cfg = yaml.safe_load(open('${CONFIG}'))
p = os.path.join(cfg['out_dir'], cfg['run_name'] + '.pt')
if os.path.exists(p):
    n = len(torch.load(p, weights_only=False, map_location='cpu'))
    if n >= cfg['num_samples']:
        print(f'SKIP {cfg[\"run_name\"]}: already has {n} >= {cfg[\"num_samples\"]} samples')
        sys.exit(1)
    print(f'RESUME {cfg[\"run_name\"]}: {n} of {cfg[\"num_samples\"]} samples on disk')
else:
    print(f'FRESH {cfg[\"run_name\"]} (target {cfg[\"num_samples\"]})')
sys.exit(0)
PY

        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
STATUS=$?
END=$(date +%s)
echo "ARM ${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
