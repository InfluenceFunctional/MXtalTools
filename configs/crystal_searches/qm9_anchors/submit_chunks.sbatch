#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-49
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=qm9_anchors
#SBATCH --output=/scratch/mk8347/projects/gfn_cond/MXtalTools/slurm_logs/qm9_anchors_%A_%a.out

# qm9_anchors :: one array task per molecule chunk.
# Config names are index-keyed (qm9c100k_chunk<k>.yaml), so the array id selects the config
# directly -- no INDEX mapping. Chunks are disjoint molecule sets and write distinct outputs.
module purge

# ---- paths ----
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/conditional

# ---- config selection ----
CONFIG=${MXT_ROOT}/configs/crystal_searches/qm9_anchors/qm9c100k_chunk${SLURM_ARRAY_TASK_ID}.yaml
if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi

# run_search writes opt_intermediates.pt into the CWD under an unnamed, run-agnostic path;
# on a shared CWD one task's OOM salvage can be read back by another. One CWD per task.
TASKDIR=/scratch/mk8347/tmp/qm9_anchors/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${TASKDIR}" "${DATA_ROOT}/anchors"

# ---- run ----
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH
        python ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
STATUS=$?

if [ ${STATUS} -eq 0 ]; then
    rm -rf "${TASKDIR}"
fi
exit ${STATUS}
