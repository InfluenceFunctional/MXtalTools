#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8
#SBATCH --account=torch_pr_226_chemistry
#SBATCH --job-name=sep27_prop
#SBATCH --output=/scratch/mk8347/logs/acr_proposals_sep27-%A_%a.out

# acr_proposals_sep27: 9 jobs, acridine sg14 Z'=2 (MACE): clean clamp vs wrap, and three proposal arms under wrap.
# Array index -> <index>.yaml, or <index>a.yaml then <index>b.yaml (the seeded jobs run both shards in one job).
# Proposal arms read compact seed files (seeds/*.pt here); they are expanded to DATA_ROOT/sep27_seeds/ with
# build_seeds.py before run_search (rebuilt if the committed file changed). Needs run_search's run_files fix.
# See README.md / MANIFEST.md in this directory. Timing basis: a 2000-sample aug21 arm took ~3.8-4.0 h on an
# A100, so 6000 samples ~12 h; --time 20 h leaves headroom, and run_search resumes a partial arm if requeued.
module purge

IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
OVERLAY=/scratch/mk8347/venvs/mxt_container/overlay-50G-10M-copy.ext3
PROJECT_ROOT=/scratch/mk8347/projects/gfn_cond
MXT_ROOT=${PROJECT_ROOT}/MXtalTools
DATA_ROOT=/scratch/mk8347/data/crystal_datasets/acridine
CONFDIR=${MXT_ROOT}/configs/crystal_searches/acr_proposals_sep27

CONFIGS=$(ls ${CONFDIR}/${SLURM_ARRAY_TASK_ID}.yaml ${CONFDIR}/${SLURM_ARRAY_TASK_ID}[ab].yaml 2>/dev/null)
if [ -z "${CONFIGS}" ]; then
    echo "no config for array index ${SLURM_ARRAY_TASK_ID} in ${CONFDIR}" >&2
    exit 1
fi
if [ ! -d "${DATA_ROOT}/opt_outs" ]; then
    echo "missing output dir: ${DATA_ROOT}/opt_outs" >&2
    exit 1
fi
# this battery needs run_search's stale-intermediates / resume fix: without it ~half the relaxations repeat
if ! grep -q "def run_files" ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py; then
    echo "run_search.py lacks the stale-intermediates/resume fix (run_files): git pull MXtalTools first" >&2
    exit 1
fi

START=$(date +%s)
STATUS=0
for CONFIG in ${CONFIGS}; do
    # one scratch working directory per config (run_search's side files are also named per run)
    TASKDIR=$(mktemp -d)
    echo "JOB ${SLURM_ARRAY_TASK_ID} config=$(basename ${CONFIG}) start=$(date -Is)"
    srun singularity exec --nv \
        --overlay ${OVERLAY}:ro \
        --bind ${PROJECT_ROOT}:${PROJECT_ROOT} \
        --bind /scratch/mk8347/data:/scratch/mk8347/data \
        --pwd ${TASKDIR} \
        ${IMAGE} \
        /bin/bash -c "
            source /ext3/env.sh
            export PYTHONPATH=${MXT_ROOT}:\$PYTHONPATH

            # (1) expand the compact proposal seeds (again if the committed file changed);
            # (2) skip a complete config (exit 3), resume a partial one. Any other failure stops the job loudly.
            python - <<'PY'
import sys, os, json, yaml, torch
cfg = yaml.safe_load(open('${CONFIG}'))
dp = cfg.get('dataset_path')
compact = os.path.join('${CONFDIR}', 'seeds', os.path.splitext(os.path.basename(dp))[0] + '.pth') if dp else None
if compact and os.path.exists(compact):
    n_compact = len(torch.load(compact, weights_only=False)['params'])
    if not os.path.exists(dp) or len(torch.load(dp, weights_only=False, map_location='cpu')) != n_compact:
        sys.path.insert(0, '${CONFDIR}')
        from build_seeds import build
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        build(compact, cfg['mol_path'], dp + '.tmp')
        os.replace(dp + '.tmp', dp)
        print(f'built {n_compact} seeds {dp} from {compact}')
want = cfg['num_samples']
if cfg.get('init_sample_method') == 'data':
    want = min(want, len(torch.load(dp, weights_only=False, map_location='cpu')))
stem = os.path.join(cfg['out_dir'], cfg['run_name'])
if os.path.exists(stem + '_progress.json'):
    cur = json.load(open(stem + '_progress.json'))['cursor']
    if cur >= want:
        print(f'SKIP {cfg[\"run_name\"]}: cursor {cur} >= {want}')
        sys.exit(3)
    print(f'RESUME {cfg[\"run_name\"]}: cursor {cur} of {want}')
elif os.path.exists(stem + '.pt'):
    print(f'RESUME {cfg[\"run_name\"]}: output without a progress file (older run_search)')
else:
    print(f'FRESH {cfg[\"run_name\"]} (target {want})')
sys.exit(0)
PY
            rc=\$?
            if [ \$rc -eq 3 ]; then exit 0; fi
            if [ \$rc -ne 0 ]; then echo 'seed build or pre-check failed' >&2; exit \$rc; fi

            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
            python -u ${MXT_ROOT}/mxtaltools/crystal_search/run_search.py --config ${CONFIG}
        "
    RC=$?
    rm -rf "${TASKDIR}"
    echo "JOB ${SLURM_ARRAY_TASK_ID} config=$(basename ${CONFIG}) status=${RC}"
    [ ${RC} -ne 0 ] && STATUS=${RC}
done

END=$(date +%s)
echo "JOB ${SLURM_ARRAY_TASK_ID} status=${STATUS} seconds=$((END-START))"
exit ${STATUS}
