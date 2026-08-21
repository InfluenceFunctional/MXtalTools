#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-9
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acr_bench_gpu

# GPU half of the CPU-vs-GPU throughput benchmark. These are FULL production arms
# (2000 samples, batch 1000) -- the point we actually run at. The CPU half runs a
# batch-size sweep at its own operating point instead, so the arms are
# deliberately NOT identical.
#
# Compare THROUGHPUT (samples/second), not wall time. summarize_bench.py does it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine

CONFIG=${MXT_ROOT}/configs/crystal_searches/acr_bench_aug21/bench_gpu_${SLURM_ARRAY_TASK_ID}.yaml
if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi
# fail now rather than after the optimisation, at the first torch.save
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi

# run_search writes opt_intermediates.pt into the CWD under a run-agnostic name;
# on a shared CWD one task's OOM salvage can be read back by another.
TASKDIR=$(mktemp -d)
trap 'rm -rf "${TASKDIR}"' EXIT

START=$(date +%s)
echo "BENCH device=gpu arm=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"

srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        # unbuffered: a killed job otherwise loses the whole buffer, which is how
        # a previous chunk died with no message at all
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
STATUS=$?

END=$(date +%s)
echo "BENCH device=gpu arm=${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
