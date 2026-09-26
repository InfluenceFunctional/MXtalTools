#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sep26_wrap
#SBATCH --output=/scratch/mk8347/logs/acr_wrap_sep26-%A_%a.out

# acr_wrap_sep26: 8 test jobs, acridine sg14 Z'=2 (MACE), clamp vs opt-in centroid_boundary 'wrap'.
# Array index -> <index>.yaml, or <index>a.yaml then <index>b.yaml (the seeded jobs run both shards in one job).
# See README.md / MANIFEST.md in this directory. Timing basis: a 2000-sample aug21 arm took ~3.8-4.0 h on an
# A100, so 6000 samples ~12 h; --time 20 h leaves headroom, and run_search resumes a partial arm if requeued.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine
CONFDIR=${MXT_ROOT}/configs/crystal_searches/acr_wrap_sep26

CONFIGS=$(ls ${CONFDIR}/${SLURM_ARRAY_TASK_ID}.yaml ${CONFDIR}/${SLURM_ARRAY_TASK_ID}[ab].yaml 2>/dev/null)
if [ -z "${CONFIGS}" ]; then
    echo "no config for array index ${SLURM_ARRAY_TASK_ID} in ${CONFDIR}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi

START=$(date +%s)
STATUS=0
for CONFIG in ${CONFIGS}; do
    # one scratch working directory per config: run_search writes opt_intermediates.pt to the cwd on OOM
    TASKDIR=$(mktemp -d)
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

            # Skip an already-complete config (run_search resumes a PARTIAL one itself; a complete one would slice
            # an empty batch). Seeded configs are clipped to their shard size. Same check as acr_rerun_aug21.
            python - <<'PY' || exit 0
import sys, os, yaml, torch
cfg = yaml.safe_load(open('${CONFIG}'))
want = cfg['num_samples']
if cfg.get('init_sample_method') == 'data':
    seeds = torch.load(cfg['dataset_path'], weights_only=False, map_location='cpu')
    want = min(want, len(seeds))
p = os.path.join(cfg['out_dir'], cfg['run_name'] + '.pt')
if os.path.exists(p):
    n = len(torch.load(p, weights_only=False, map_location='cpu'))
    if n >= want:
        print(f'SKIP {cfg[\"run_name\"]}: already has {n} >= {want} samples')
        sys.exit(1)
    print(f'RESUME {cfg[\"run_name\"]}: {n} of {want} samples on disk')
else:
    print(f'FRESH {cfg[\"run_name\"]} (target {want})')
sys.exit(0)
PY

            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
        "
    RC=$?
    rm -rf "${TASKDIR}"
    echo "JOB ${SLURM_ARRAY_TASK_ID} config=$(basename ${CONFIG}) status=${RC}"
    [ ${RC} -ne 0 ] && STATUS=${RC}
done

END=$(date +%s)
echo "JOB ${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
