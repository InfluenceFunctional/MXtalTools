#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=e6pilot
#SBATCH --output=/scratch/mk8347/logs/%x-%A_%a.out
#SBATCH --array=0-3

# exp-6 recall test, stage B pilot (README.md): array task -> (molecule, kind)
#   0 imidazole reference   1 acetic_acid reference   (UMA-s 1.1 random-start search, <= 2000 starts)
#   2 imidazole cheap       3 acetic_acid cheap       (exp-6 and eLJ searches, 300 picks each, UMA-s on the picks)
# Outputs: /scratch/mk8347/data/crystal_datasets/exp6_recall_nov10/pilot/. Resubmitting resumes (run_search
# force_restart_run false; existing picks kept).
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
PYLIBS=/scratch/mk8347/pylibs  # packages the container lacks (spglib)
HERE=${MXT_ROOT}/configs/crystal_searches/exp6_recall_nov10
OUT=/scratch/mk8347/data/crystal_datasets/exp6_recall_nov10/pilot
TASKS=("imidazole reference" "acetic_acid reference" "imidazole cheap" "acetic_acid cheap")
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:?array task id}]}
COMMIT=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "${OUT}"

echo "E6PILOT ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} task='${TASK}' code=${COMMIT} start=$(date -Is)"
srun singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --bind /scratch/mk8347/models:/scratch/mk8347/models \
    --pwd ${OUT} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:${PYLIBS}:\$PYTHONPATH MXT_COMMIT=${COMMIT} OMP_NUM_THREADS=8
        python -c \"import torch, spglib, fairchem.core, importlib.metadata as m, mxtaltools.crystal_search.run_search as rs; assert hasattr(rs, 'attach_mlip_parity'), 'MXtalTools too old: pull'; print('torch', torch.__version__, 'fairchem-core', m.version('fairchem-core'))\" || exit 1
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        exec python -u ${HERE}/run_pilot.py ${TASK}"
RC=$?
echo "E6PILOT ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} task='${TASK}' status=${RC} end=$(date -Is)"
exit ${RC}
