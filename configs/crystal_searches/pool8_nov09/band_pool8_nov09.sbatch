#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8band
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-45

# pool8_nov09 band sets, CPU only: band_from_slim.py on each search task's <run>_slim.pt (the prior constructor's
# first step; README.md "Band sets"). One array job runs TASKS_PER_JOB tasks on LANES cores, each lane its tasks one
# after another: job j, lane l runs tasks j * TASKS_PER_JOB + l, + LANES, ... A task with OUT/done/<run>.json is
# skipped and a task stopped mid-way redoes only its unwritten splits, so resubmitting the same array finishes what is
# left. Each task logs to OUT/logs/<run>.log; this job's log prints one line per task with its exit status.
module purge

TASKS_PER_JOB=64
LANES=8
N_TASKS=2899
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib), as the search job
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
HERE=${MXT_ROOT}/configs/crystal_searches/pool8_nov09
BASE=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09
SEARCH=${BASE}/search
SPLIT=${BASE}/mols/pool8_mols.tsv
OUT=${BASE}/band32
JOB=${SLURM_ARRAY_TASK_ID:?array task id}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "${OUT}/logs"
echo "JOB ${SLURM_ARRAY_JOB_ID}_${JOB} tasks $(( JOB * TASKS_PER_JOB ))..$(( JOB * TASKS_PER_JOB + TASKS_PER_JOB - 1 )) code=${COMMIT} start=$(date -Is)"
[ -f "${SPLIT}" ] || { echo "FATAL: split file missing ${SPLIT}"; exit 1; }

srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${OUT} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH
        export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        python -c 'import torch, spglib, mxtaltools.crystal_search.stream_search' \
            || { echo 'FATAL: import check failed in the container'; exit 1; }
        run_task() {
            local T=\$1 RUN=pool8_c\$1
            [ -f ${OUT}/done/\$RUN.json ] && { echo \"task \$T \$RUN: done before\"; return 0; }
            [ -f ${SEARCH}/\$RUN'_slim.pt' ] || { echo \"task \$T: no ${SEARCH}/\$RUN'_slim.pt'\"; return 1; }
            python -u ${HERE}/band_from_slim.py ${SEARCH}/\$RUN'_slim.pt' ${OUT} --split ${SPLIT} \
                >> ${OUT}/logs/\$RUN.log 2>&1
            local RC=\$?
            echo \"task \$T \$RUN: status \$RC \$(date -Is)\"
            return \$RC
        }
        lane() {
            local L=\$1 FAIL=0 T
            for (( T = ${JOB} * ${TASKS_PER_JOB} + L; T < (${JOB} + 1) * ${TASKS_PER_JOB} && T < ${N_TASKS}; T += ${LANES} )); do
                run_task \$T || FAIL=1
            done
            return \$FAIL
        }
        PIDS=()
        for L in \$(seq 0 \$(( ${LANES} - 1 ))); do
            lane \$L & PIDS+=(\$!)
        done
        FAIL=0
        for P in \"\${PIDS[@]}\"; do wait \$P || FAIL=1; done
        exit \$FAIL"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${JOB} status=${RC} end=$(date -Is)"
exit ${RC}
