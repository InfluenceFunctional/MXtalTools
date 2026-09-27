#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --signal=USR1@900
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acr_camp
#SBATCH --output=/scratch/mk8347/logs/acr_campaign_sep28-%A_%a.out

# acr_campaign_sep28: one array task = one search job of one stream (STREAM=random|eljstart|hops, set by launch.sh).
# Run name <stream>_<task>, opt_seed = the stream's base + task * 1.5e7 (streams/<stream>.yaml). The job writes shards to
# the campaign directory, may run the curate pass between batches, and stops between batches on STOP / STOP.<stream>,
# at 30000 relaxations, or on SIGUSR1 15 min before the walltime. Resubmitting the same array resumes every task.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
CONFDIR=${MXT_ROOT}/configs/crystal_searches/acr_campaign_sep28
CAMP=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_campaign_sep28
STREAM=${STREAM:?set STREAM to random, eljstart or hops}
TASK=${SLURM_ARRAY_TASK_ID:?array task id}
RUN=${STREAM}_${TASK}

if [ ! -f "${CONFDIR}/streams/${STREAM}.yaml" ]; then echo "no stream config ${STREAM}" >&2; exit 1; fi
if [ ! -f "${CAMP}/coord.yaml" ]; then echo "campaign not set up: run launch.sh" >&2; exit 1; fi
if [ -f "${CAMP}/STOP" ] || [ -f "${CAMP}/STOP.${STREAM}" ]; then echo "campaign or stream ${STREAM} stopped"; exit 0; fi
# a lease left by a job that no longer exists (SIGKILL, node failure) would block this run for up to 30 min
OWNER=${CAMP}/runs/${RUN}_owner.json
if [ -f "${OWNER}" ]; then
    OLD=$(python3 -c "import json; print(json.load(open('${OWNER}')).get('job') or '')" 2>/dev/null || true)
    if [ -n "${OLD}" ] && [ "${OLD}" != "${SLURM_JOB_ID}" ] && [ -z "$(squeue -h -j "${OLD}" 2>/dev/null)" ]; then
        echo "removing the lease of finished job ${OLD}"
        rm -f "${OWNER}"
    fi
fi

echo "JOB ${SLURM_ARRAY_JOB_ID}_${TASK} stream=${STREAM} run=${RUN} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${CAMP}/runs \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH
        python - <<'PY'
import yaml
cfg = yaml.safe_load(open('${CONFDIR}/streams/${STREAM}.yaml'))
cfg['run_name'] = '${RUN}'
cfg['opt_seed'] = int(cfg['opt_seed']) + int('${TASK}') * 15_000_000
yaml.safe_dump(cfg, open('${CAMP}/runs/${RUN}.yaml', 'w'))
PY
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CAMP}/runs/${RUN}.yaml
    "
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${TASK} status=${RC} end=$(date -Is)"
exit ${RC}
