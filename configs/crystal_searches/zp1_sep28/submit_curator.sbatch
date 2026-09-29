#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=z1_curator
#SBATCH --output=/scratch/mk8347/logs/%x-%j.out

# zp1_sep28 curator of one campaign (CPU only; launch.sh submits it with CAMP = the campaign directory): the campaign's curate pass every 15 min
# (coordinator run_loop under the curator lease), so the GPU jobs spend no GPU time on it -- they curate themselves only
# once the last pass is more than an hour old. A pass while a GPU job holds the curator lease is skipped. Exits after two
# passes that both report the campaign stop, or after 23.5 h; relaunching the campaign resubmits it.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
CAMP=${CAMP:?set CAMP to the campaign directory (launch.sh does)}

if [ ! -f "${CAMP}/coord.yaml" ]; then echo "campaign not set up: run launch.sh" >&2; exit 1; fi

# a curator lease left by a job that no longer exists: removed only on positive evidence (as in submit_stream.sbatch)
F=${CAMP}/curator_owner.json
if [ -f "${F}" ]; then
    OLD=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('job') or '')" "${F}" 2>/dev/null || true)
    if [ -n "${OLD}" ] && [ "${OLD}" != "${SLURM_JOB_ID}" ]; then
        Q=$(squeue -h -j "${OLD}" -o %T 2>&1); RC=$?
        if { [ ${RC} -eq 0 ] && [ -z "${Q}" ]; } || [[ "${Q}" == *"Invalid job id"* ]]; then
            echo "removing ${F}: its job ${OLD} is gone"
            rm -f "${F}"
        fi
    fi
fi

echo "CURATOR ${SLURM_JOB_ID} code=$(git -C "${MXT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown) start=$(date -Is)"
srun singularity exec \
    --overlay ${OVERLAY}:ro \
    --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
    --bind /scratch/mk8347/data:/scratch/mk8347/data \
    --pwd ${CAMP} \
    ${IMAGE} \
    /bin/bash -c "
        source /ext3/env.sh
        export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
        exec python -u -m mxtaltools.crystal_search.coordinator ${CAMP} --loop --interval 900 --max_hours 23.5"
RC=$?
echo "CURATOR ${SLURM_JOB_ID} status=${RC} end=$(date -Is)"
exit ${RC}
