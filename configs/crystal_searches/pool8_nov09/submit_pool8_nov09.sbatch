#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=pool8nov09
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-362%100

# pool8_nov09, CPU only: one array job runs TASKS_PER_JOB tasks side by side, one core each (make_battery.py writes
# tasks/<task>.yaml, INDEX.tsv, and the array range, TASKS_PER_JOB and N_TASKS below). Job j runs tasks
# j * TASKS_PER_JOB .. + TASKS_PER_JOB - 1. A task runs run_search (resuming from its progress file if it was stopped),
# then slim_outputs.py (<run>_slim.pt, the bulky outputs deleted). A task whose <run>_slim.pt exists is skipped, so
# resubmitting the same array finishes only what is left. Each task logs to <out_dir>/<run>.log; this job's log
# prints one line per task with its exit status.
module purge

TASKS_PER_JOB=8
N_TASKS=2899
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib), as rc_oct22
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
HERE=${MXT_ROOT}/configs/crystal_searches/pool8_nov09
JOB=${SLURM_ARRAY_TASK_ID:?array task id}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
OUT=/scratch/mk8347/data/crystal_datasets/conditional/pool8_nov09/search
mkdir -p "${OUT}"
echo "JOB ${SLURM_ARRAY_JOB_ID}_${JOB} tasks $(( JOB * TASKS_PER_JOB ))..$(( JOB * TASKS_PER_JOB + TASKS_PER_JOB - 1 )) code=${COMMIT} start=$(date -Is)"

# A lease left by a job that no longer exists (OOM kill, node failure) would block its run for up to 30 min. Remove it
# only on positive evidence that the job is gone: squeue answers and lists nothing, or says the id is invalid. (As
# rc_oct22; run here, outside the container, where squeue exists.)
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
for J in $(seq 0 $(( TASKS_PER_JOB - 1 ))); do
    T=$(( JOB * TASKS_PER_JOB + J ))
    [ ${T} -lt ${N_TASKS} ] || break
    R=$(awk '/^run_name:/{print $2}' "${HERE}/tasks/${T}.yaml" 2>/dev/null)
    [ -n "${R}" ] && clear_dead_lease "${OUT}/${R}_owner.json"
done

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
        python -c 'import torch, spglib, yaml, mxtaltools.crystal_search.stream_search' \
            || { echo 'FATAL: import check failed in the container'; exit 1; }
        echo \"DEVICE cpu: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\"
        run_task() {
            local T=\$1 CFG=${HERE}/tasks/\$1.yaml RUN MOL
            [ -f \"\$CFG\" ] || { echo \"task \$T: no config\"; return 1; }
            RUN=\$(awk '/^run_name:/{print \$2}' \"\$CFG\"); MOL=\$(awk '/^mol_path:/{print \$2}' \"\$CFG\")
            [ -f ${OUT}/\$RUN'_slim.pt' ] && { echo \"task \$T \$RUN: done before\"; return 0; }
            [ -f \"\$MOL\" ] || { echo \"task \$T: molecule file missing \$MOL\"; return 1; }
            python - \"\$CFG\" ${OUT}/\$RUN.yaml <<'PY' || return 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open(sys.argv[2], 'w'))
PY
            python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${OUT}/\$RUN.yaml \
                > ${OUT}/\$RUN.log 2>&1 \
              && python -u ${HERE}/slim_outputs.py ${OUT}/\$RUN.yaml >> ${OUT}/\$RUN.log 2>&1
            local RC=\$?
            echo \"task \$T \$RUN: status \$RC \$(date -Is)\"
            return \$RC
        }
        PIDS=()
        for J in \$(seq 0 \$(( ${TASKS_PER_JOB} - 1 ))); do
            T=\$(( ${JOB} * ${TASKS_PER_JOB} + J ))
            [ \$T -lt ${N_TASKS} ] || break
            run_task \$T & PIDS+=(\$!)
        done
        FAIL=0
        for P in \"\${PIDS[@]}\"; do wait \$P || FAIL=1; done
        exit \$FAIL"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${JOB} status=${RC} end=$(date -Is)"
exit ${RC}
