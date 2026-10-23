#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --signal=USR1@600
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=rcoct22
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-999%250

# rc_oct22, CPU only: one array task = one row of INDEX.tsv = tasks/<task>.yaml (make_battery.py writes both, and the
# array range above from the task count). A task runs its random starts (2) for every molecule of one chunk file on
# one core, in batches of 50, and writes <out_dir>/<run_name>.pt. The run config is rendered into the output directory
# with the MXtalTools commit stamped in. It stops between batches on SIGUSR1, 10 min before the walltime (python is
# exec'd as the container's process, so the signal reaches it); resubmitting the same array resumes every task from
# its progress file, and a finished task exits at once. A task whose molecule file is not on the cluster exits at once
# with an error. Task = array index + TASK_OFFSET (default 0), so tasks above the cluster's MaxArraySize run as
#   sbatch --export=ALL,TASK_OFFSET=1000 --array=0-<n>%250 submit_rc_oct22.sbatch     (tasks 1000..1000+n)
# check_battery.py matches logs to tasks by the task each log prints, not by the array index.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib), as acr_zp1_sep30; pip install --target
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
HERE=${MXT_ROOT}/configs/crystal_searches/rc_oct22
IDX=${SLURM_ARRAY_TASK_ID:?array task id}
TASK=$(( IDX + ${TASK_OFFSET:-0} ))
CFG=${HERE}/tasks/${TASK}.yaml
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)

if [ ! -f "${CFG}" ]; then echo "no task config ${CFG}" >&2; exit 1; fi
RUN=$(awk '/^run_name:/{print $2}' "${CFG}")
OUT=$(awk '/^out_dir:/{print $2}' "${CFG}")
MOL=$(awk '/^mol_path:/{print $2}' "${CFG}")
if [ -z "${RUN}" ] || [ -z "${OUT}" ] || [ -z "${MOL}" ]; then echo "could not read run_name/out_dir/mol_path from ${CFG}" >&2; exit 1; fi
if [ ! -f "${MOL}" ]; then echo "molecule file missing: ${MOL}" >&2; exit 1; fi
mkdir -p "${OUT}"

# A lease left by a job that no longer exists (node failure, SIGKILL) would block its run for up to 30 min. Remove it
# only on positive evidence that the job is gone: squeue answers and lists nothing, or says the id is invalid.
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
clear_dead_lease "${OUT}/${RUN}_owner.json"

echo "JOB ${SLURM_ARRAY_JOB_ID}_${IDX} task=${TASK} run=${RUN} mol=${MOL} code=${COMMIT} start=$(date -Is)"
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
        python - <<'PY' || exit 1
import yaml
cfg = yaml.safe_load(open('${CFG}'))
cfg['mxt_commit'] = '${COMMIT}'
yaml.safe_dump(cfg, open('${OUT}/${RUN}.yaml', 'w'))
PY
        echo \"DEVICE cpu: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)\"
        exec python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${OUT}/${RUN}.yaml"
RC=$?
echo "JOB ${SLURM_ARRAY_JOB_ID}_${IDX} task=${TASK} status=${RC} end=$(date -Is)"
exit ${RC}
