#!/bin/bash
# acr_campaign_sep28: set up the campaign directory, check the energy model, submit the three streams.
#   bash launch.sh [n_tasks_per_stream]      (default 4; run from anywhere)
# Re-running is safe: an existing campaign directory is reused if its coord.yaml matches this one, and resubmitted
# arrays resume their runs. hops starts 60 min later, after the first curate pass has written hop_parents.pt (a hop job
# that finds no parent yet stops cleanly; resubmit it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
N=${1:-4}
CAMP=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_campaign_sep28
MODEL=/scratch/mk8347/data/acr_112025_mh1_stagetwo.model

# the energy model on the cluster must be the file the priors were scored with (coordinator.energy_model_id)
MID=$(python3 - "${MODEL}" <<'PY'
import hashlib, os, sys
p = sys.argv[1]
size = os.path.getsize(p)
h = hashlib.sha1()
with open(p, 'rb') as fh:
    h.update(fh.read(1 << 20))
    if size > 2 << 20:
        fh.seek(-(1 << 20), os.SEEK_END)
        h.update(fh.read(1 << 20))
print(f'mace:{os.path.basename(p)}:{size}:{h.hexdigest()[:16]}')
PY
)
WANT=$(grep '^energy_model_id:' "${HERE}/coord.yaml" | awk '{print $2}')
if [ "${MID}" != "${WANT}" ]; then
    echo "the cluster's energy model is ${MID}; the campaign and its priors were built for ${WANT}" >&2
    exit 1
fi

mkdir -p "${CAMP}/runs"
if [ -f "${CAMP}/coord.yaml" ]; then
    if ! cmp -s "${HERE}/coord.yaml" "${CAMP}/coord.yaml"; then
        echo "${CAMP}/coord.yaml differs from ${HERE}/coord.yaml: start a new campaign directory instead" >&2
        exit 1
    fi
else
    cp "${HERE}/coord.yaml" "${CAMP}/coord.yaml"
fi

LAST=$((N - 1))
sbatch --export=ALL,STREAM=random --array=0-${LAST} "${HERE}/submit_stream.sbatch"
sbatch --export=ALL,STREAM=eljstart --array=0-${LAST} "${HERE}/submit_stream.sbatch"
sbatch --export=ALL,STREAM=hops --array=0-${LAST} --begin=now+60minutes "${HERE}/submit_stream.sbatch"
echo "campaign ${CAMP}: 3 streams x ${N} tasks submitted; progress in ${CAMP}/stats.md"
