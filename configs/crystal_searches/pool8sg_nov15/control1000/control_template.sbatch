#!/bin/bash
#SBATCH --time=@TIME@
#SBATCH --mem=@MEM@
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=p8ctl
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=@ARRAY@

# pool8sg_nov15 control1000: one array job per task (tasks/<i>.yaml, INDEX.tsv), one core: run_search (resuming from
# its progress file if stopped), then pool8_nov09's slim_outputs.py. No band step: the slims are the result. Rendered
# by make_control.py from control_template.sbatch.
module purge

N_TASKS=@N_TASKS@
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib), as pool8_nov09
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
HERE=${MXT_ROOT}/configs/crystal_searches/pool8sg_nov15/control1000
SLIM=${MXT_ROOT}/configs/crystal_searches/pool8_nov09/slim_outputs.py
T=${SLURM_ARRAY_TASK_ID:?array task id}
[ ${T} -lt ${N_TASKS} ] || { echo "no task ${T}"; exit 1; }
CFG=${HERE}/tasks/${T}.yaml
RUN=$(awk '/^run_name:/{print $2}' "${CFG}")
OUT=$(awk '/^out_dir:/{print $2}' "${CFG}")
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
# The same frame rule and clean library as the battery (pool8sg_nov15/submit_template.sbatch).
FRAME_BLOB=ff4e836e446452b74dadf3c83ce01a7b8a000513
[ "$(git -C "${MXT_ROOT}" hash-object mxtaltools/common/geometry_utils.py)" = "${FRAME_BLOB}" ] \
    || { echo "FATAL: mxtaltools/common/geometry_utils.py is not the battery's frame rule (blob ${FRAME_BLOB})"; exit 1; }
git -C "${MXT_ROOT}" diff --quiet HEAD -- mxtaltools \
    || { echo "FATAL: local changes under ${MXT_ROOT}/mxtaltools; the search must run committed code"; exit 1; }
mkdir -p "${OUT}"
if [ -f "${OUT}/${RUN}_slim.pt" ]; then echo "task ${T} ${RUN}: done before"; exit 0; fi
echo "JOB ${SLURM_ARRAY_JOB_ID}_${T} ${RUN} code=${COMMIT} start=$(date -Is)"

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
        python - ${CFG} ${OUT}/${RUN}.yaml <<'PY' || exit 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open(sys.argv[2], 'w'))
PY
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${OUT}/${RUN}.yaml > ${OUT}/${RUN}.log 2>&1 \
          && python -u ${SLIM} ${OUT}/${RUN}.yaml >> ${OUT}/${RUN}.log 2>&1"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${T} ${RUN} status=${RC} end=$(date -Is)"
exit ${RC}
