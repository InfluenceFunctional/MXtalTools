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

# acr_campaign_sep28: one array task = one search job of one stream (STREAM=random|eljstart|hops, set by launch.sh,
# which also names the job acr_camp_<stream>). Run name <stream>_<task>, opt_seed = the stream's base + task * 1.5e7.
# The run config is rendered from the campaign's FROZEN copy of the stream config (launch.sh freezes it on the first
# launch), so a later git pull cannot change a campaign's stages. The job writes shards to the campaign directory, may
# run the curate pass between batches, and stops between batches on STOP / STOP.<stream>, at 30000 relaxations, or on
# SIGUSR1 15 min before the walltime: python is exec'd as the container's process, so the signal reaches it.
# Resubmitting the same array (launch.sh) resumes every task.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
CAMP=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_campaign_sep28
STREAM=${STREAM:?set STREAM to random, eljstart or hops}
TASK=${SLURM_ARRAY_TASK_ID:?array task id}
RUN=${STREAM}_${TASK}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)

if [ ! -f "${CAMP}/coord.yaml" ]; then echo "campaign not set up: run launch.sh" >&2; exit 1; fi
if [ ! -f "${CAMP}/streams/${STREAM}.yaml" ]; then echo "no frozen stream config ${CAMP}/streams/${STREAM}.yaml" >&2; exit 1; fi
if [ -f "${CAMP}/STOP" ] || [ -f "${CAMP}/STOP.${STREAM}" ]; then echo "campaign or stream ${STREAM} stopped"; exit 0; fi

# A lease left by a job that no longer exists (node failure, SIGKILL) would block its run (or every curate pass) for up
# to 30-60 min. Remove it only on positive evidence that the job is gone: squeue answers and lists nothing, or says the
# id is invalid. Any other squeue failure keeps the lease, which then goes stale by itself.
clear_dead_lease() {
    local f=$1 old q rc
    [ -f "${f}" ] || return 0
    old=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('job') or '')" "${f}" 2>/dev/null || true)
    if [ -z "${old}" ] || [ "${old}" = "${SLURM_JOB_ID}" ]; then return 0; fi
    q=$(squeue -h -j "${old}" -o %T 2>&1); rc=$?
    if { [ ${rc} -eq 0 ] && [ -z "${q}" ]; } || [[ "${q}" == *"Invalid job id"* ]]; then
        echo "removing ${f}: its job ${old} is gone"
        rm -f "${f}"
    fi
}
clear_dead_lease "${CAMP}/runs/${RUN}_owner.json"
clear_dead_lease "${CAMP}/curator_owner.json"

echo "JOB ${SLURM_ARRAY_JOB_ID}_${TASK} stream=${STREAM} run=${RUN} code=${COMMIT} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${CAMP}/runs \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH
        python - <<'PY' || exit 1
import yaml
cfg = yaml.safe_load(open('${CAMP}/streams/${STREAM}.yaml'))
cfg['run_name'] = '${RUN}'
cfg['opt_seed'] = int(cfg['opt_seed']) + int('${TASK}') * 15_000_000
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open('${CAMP}/runs/${RUN}.yaml', 'w'))
PY
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CAMP}/runs/${RUN}.yaml"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${TASK} status=${RC} end=$(date -Is)"
exit ${RC}
