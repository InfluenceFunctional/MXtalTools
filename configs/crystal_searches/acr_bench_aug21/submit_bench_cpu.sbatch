#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-9
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=acr_bench_cpu

# CPU half of the CPU-vs-GPU throughput benchmark. These arms are NOT paired with
# the GPU ones: 1000 samples each with batch size swept over {10,25,50,100,200} x
# 2 seeds, because the CPU's best batch size is unknown and is certainly not the
# GPU's 1000. The sweep measures throughput and finds the operating point at once.
#
# What makes the two halves comparable is the per-sample work -- same SG/Z', same
# 500-step schedule, same energy function -- so compare samples/second, NOT wall
# time. summarize_bench.py does it.
#
# Differences from the GPU script, all deliberate:
#   * no --gres and no --nv, or the job would sit in the GPU queue and quietly
#     use one anyway
#   * --cpus-per-task 16 and the thread env vars below. torch defaults to all
#     visible cores, which on a shared node is not what SLURM allocated -- the
#     job would oversubscribe and the timing would be meaningless
#   * 24h rather than 1h, and a TIMEOUT HERE IS FINE. run_search saves after
#     every batch, so an arm killed at the wall leaves everything it finished and
#     summarize_bench.py counts what is on disk. An arm that did 600 of 1000
#     samples is a valid throughput measurement; only zero output tells us
#     nothing. Do not read wall time directly -- read samples/second.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine

CONFIG=${MXT_ROOT}/configs/crystal_searches/acr_bench_aug21/bench_cpu_${SLURM_ARRAY_TASK_ID}.yaml
if [ ! -f "${CONFIG}" ]; then
    echo "missing config: ${CONFIG}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi

TASKDIR=$(mktemp -d)
trap 'rm -rf "${TASKDIR}"' EXIT

START=$(date +%s)
echo "BENCH device=cpu arm=${SLURM_ARRAY_TASK_ID} threads=${SLURM_CPUS_PER_TASK} start=$(date -Is)"

# NOTE: no --nv. Passing it would expose a GPU and torch could use it despite
# device: cpu in the config, silently invalidating the comparison.
srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${TASKDIR} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH
        # pin torch to what SLURM actually gave us
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
        export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
        # belt and braces: make a stray .cuda() fail loudly instead of silently
        # borrowing a GPU that happens to be visible on the node
        export CUDA_VISIBLE_DEVICES=''
        python -c 'import torch; print(f\"torch threads={torch.get_num_threads()} cuda_available={torch.cuda.is_available()}\")'
        python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
    "
STATUS=$?

END=$(date +%s)
echo "BENCH device=cpu arm=${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
