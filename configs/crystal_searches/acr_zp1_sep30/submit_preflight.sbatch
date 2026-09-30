#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=a1_preflight
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# acr_zp1_sep30 preflight (CPU only; launch.sh submits it with CAMP = the campaign directory, and the stream arrays with
# afterok on it): preflight.py checks the container, molecule and model, builds the seeded stream's files and writes
# <CAMP>/preflight_ok.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib); pip install --target
CAMP=${CAMP:?set CAMP to the campaign directory (launch.sh does)}

echo "PREFLIGHT ${SLURM_JOB_ID} code=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown) start=$(date -Is)"
srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${CAMP} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
        exec python -u ${MXT_ROOT}/configs/crystal_searches/acr_zp1_sep30/preflight.py ${CAMP}"
RC=$?
echo "PREFLIGHT ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
