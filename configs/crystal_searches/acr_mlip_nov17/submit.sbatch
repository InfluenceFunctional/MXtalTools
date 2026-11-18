#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@900
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acrmlip
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-2

# acr_mlip_nov17 (README.md): array task -> model, in models.py order
#   0 maceoff24m   1 umas1p1   2 umas1p2
# One GPU per model: run.py builds the seed files, then <model>_known and <model>_search (configs/). Resubmitting
# resumes each search from its committed outputs; a finished search is not redone. A single model: --array=K.
# SIGUSR1 15 min before the walltime: python is exec'd as the container's process, so the signal reaches it and the
# stream drains (no new starts) before SLURM's SIGTERM.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib)
HERE=${MXT_ROOT}/configs/crystal_searches/acr_mlip_nov17
OUT=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_mlip_nov17
LINE=$(python3 ${HERE}/models.py | sed -n "$(( ${SLURM_ARRAY_TASK_ID:?array task id} + 1 ))p")
read -r MODEL KIND PYEXTRA <<< "${LINE}"
[ -n "${MODEL}" ] || { echo "no model for array task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
[ "${PYEXTRA}" = "-" ] && PYEXTRA=""
[ -z "${PYEXTRA}" ] || [ -d "${PYEXTRA}" ] || { echo "${PYEXTRA} missing (models.py pyextra)" >&2; exit 1; }
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "${OUT}"

echo "ACRMLIP ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} model=${MODEL} code=${COMMIT} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    ${PYEXTRA:+--bind ${PYEXTRA}:${PYEXTRA}} \
    --pwd ${OUT} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PYEXTRA:+${PYEXTRA}:}${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH MXT_COMMIT=${COMMIT} OMP_NUM_THREADS=8
        python -c \"import torch, spglib, importlib.metadata as m, mxtaltools.crystal_search.stream_search as ss; assert hasattr(ss, 'drop_contact_clashes'), 'MXtalTools too old: pull'; print('torch', torch.__version__, 'fairchem-core', m.version('fairchem-core'))\" || exit 1
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${HERE}/run.py ${MODEL}"
RC=$?
echo "ACRMLIP ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} model=${MODEL} status=${RC} end=$(date -Is)"
exit ${RC}
