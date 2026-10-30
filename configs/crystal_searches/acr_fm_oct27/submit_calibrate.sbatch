#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=fm_calib
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# acr_fm_oct27 calibration of one model (calibrate.sh submits one job per model, with MODEL, OUT and PYEXTRA):
# calibrate.py MODEL OUT -> OUT/result.json (identity cut, relaxed known forms, throughput) and OUT/log.txt.
# PYEXTRA (a directory put first on PYTHONPATH; UMA 1.2 needs fairchem 2.16, the container has 2.13) comes from
# campaign.MODELS; the job prints the versions it imported before anything else.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib); pip install --target
MODEL=${MODEL:?set MODEL (calibrate.sh does)}
OUT=${OUT:?set OUT (calibrate.sh does)}
PYEXTRA=${PYEXTRA:-}
BIND_EXTRA=""
[ -n "${PYEXTRA}" ] && BIND_EXTRA="--bind ${PYEXTRA}:${PYEXTRA}"
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "${OUT}"

echo "CALIBRATE ${SLURM_JOB_ID} model=${MODEL} pyextra=${PYEXTRA:-none} code=${COMMIT} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    ${BIND_EXTRA} \
    --pwd ${OUT} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${PYEXTRA:+${PYEXTRA}:}${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH MXT_COMMIT=${COMMIT} OMP_NUM_THREADS=8
        python -c \"import torch, fairchem.core, importlib.metadata as m; print('torch', torch.__version__, 'fairchem-core', m.version('fairchem-core'), fairchem.core.__file__, 'mace-torch', m.version('mace-torch'))\" || exit 1
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${MXT_ROOT}/configs/crystal_searches/acr_fm_oct27/calibrate.py ${MODEL} ${OUT}"
RC=$?
echo "CALIBRATE ${SLURM_JOB_ID} model=${MODEL} status=${RC} end=$(date -Is)"
exit ${RC}
