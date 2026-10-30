#!/bin/bash
# acr_fm_oct27: set up and submit one model's campaign (make_campaign.py MODEL wrote <MODEL>/).
#   bash launch.sh MODEL
# The first launch freezes <MODEL>/coord.yaml and <MODEL>/streams/*.yaml into the campaign directory (the jobs read the
# frozen copies) and records the MXtalTools commit. A relaunch -- after the 6 h walltime -- resumes every task, and
# refuses when:
#   - coord.yaml or a stream config differs from the frozen copy (start a new campaign directory instead);
#   - this checkout is at another commit than the first launch (ALLOW_CODE_CHANGE=1 overrides);
#   - tasks of this campaign are still queued or running (FORCE=1 overrides), or squeue cannot say;
#   - the checkpoint on the cluster is not the one coord.yaml was built for.
# Submits the preflight job (CPU; skipped once <campaign>/preflight_ok exists), the curator (CPU; CURATOR=0 skips it),
# then the seeded and random arrays (afterok on the preflight) and the hops array (after the first random task starts: a
# hop job waits, up to a bound, for the first basin). Jobs are named <MODEL>_<stream>; task counts come from
# <MODEL>/model.env (campaign.TASKS).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MXT=$(cd "${HERE}/../../.." && pwd)
MODEL=${1:?usage: bash launch.sh MODEL}
[ -f "${HERE}/${MODEL}/model.env" ] || { echo "no ${HERE}/${MODEL}/model.env: run make_campaign.py ${MODEL}" >&2; exit 1; }
# shellcheck disable=SC1090
source "${HERE}/${MODEL}/model.env"
COMMIT=$(git -C "${MXT}" rev-parse HEAD)

model_id() {  # coordinator.energy_model_id for a checkpoint of kind $1
    python3 - "$1" "$2" <<'PY'
import hashlib, os, sys
kind, p = sys.argv[1], sys.argv[2]
size = os.path.getsize(p)
h = hashlib.sha1()
with open(p, 'rb') as fh:
    h.update(fh.read(1 << 20))
    if size > 2 << 20:
        fh.seek(-(1 << 20), os.SEEK_END)
        h.update(fh.read(1 << 20))
print(f'{kind}:{os.path.basename(p)}:{size}:{h.hexdigest()[:16]}')
PY
}

freeze() {  # copy on the first launch; afterwards the copy must not change under the running campaign
    local src=$1 dst=$2
    if [ -f "${dst}" ]; then
        if ! cmp -s "${src}" "${dst}"; then
            echo "${dst} differs from ${src}: start a new campaign directory rather than change a running one" >&2
            exit 1
        fi
    else
        cp "${src}" "${dst}"
    fi
}

want=$(grep '^energy_model_id:' "${HERE}/${MODEL}/coord.yaml" | awk '{print $2}')
[ "${want}" = "${MODEL_ID}" ] || { echo "coord.yaml and model.env disagree: rerun make_campaign.py ${MODEL}" >&2; exit 1; }
mid=$(model_id "${KIND}" "${MODEL_PATH}") || { echo "cannot read ${MODEL_PATH}" >&2; exit 1; }
[ "${mid}" = "${want}" ] || { echo "the cluster's checkpoint is ${mid}; the campaign was built for ${want}" >&2; exit 1; }
[ -z "${PYEXTRA}" ] || [ -d "${PYEXTRA}" ] || { echo "${PYEXTRA} missing (campaign.MODELS pyextra)" >&2; exit 1; }
mkdir -p "${CDIR}/runs" "${CDIR}/streams" || exit 1
freeze "${HERE}/${MODEL}/coord.yaml" "${CDIR}/coord.yaml"
for s in random hops seeded; do freeze "${HERE}/${MODEL}/streams/${s}.yaml" "${CDIR}/streams/${s}.yaml"; done
if [ -f "${CDIR}/launch_commit" ]; then
    if [ "$(cat "${CDIR}/launch_commit")" != "${COMMIT}" ] && [ "${ALLOW_CODE_CHANGE:-0}" != "1" ]; then
        echo "launched at MXtalTools $(cat "${CDIR}/launch_commit"); this checkout is ${COMMIT}. Check out that commit," \
             "or set ALLOW_CODE_CHANGE=1 to continue under the new code" >&2
        exit 1
    fi
else
    echo "${COMMIT}" > "${CDIR}/launch_commit"
fi
live=$(squeue -h -u "${USER}" -n "${MODEL}_random,${MODEL}_hops,${MODEL}_seeded,${MODEL}_preflight" -o %i) || {
    echo "squeue failed, cannot check for live tasks" >&2; exit 1; }
if [ -n "${live}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "tasks still queued or running ($(echo ${live} | tr '\n' ' ')); not resubmitting (FORCE=1)" >&2
    exit 1
fi

EXP="ALL,CAMP=${CDIR},MODEL=${MODEL},PYEXTRA=${PYEXTRA}"
dep=""
if [ ! -f "${CDIR}/preflight_ok" ]; then
    pj=$(sbatch --parsable --job-name="${MODEL}_preflight" --export="${EXP}" "${HERE}/submit_preflight.sbatch") || {
        echo "preflight not submitted" >&2; exit 1; }
    pj=${pj%%;*}
    dep="--dependency=afterok:${pj}"
    echo "preflight job ${pj} (the arrays start only if it succeeds; log /scratch/mk8347/logs/${MODEL}_preflight-${pj}.out)"
fi
if [ "${CURATOR:-1}" = "1" ]; then
    cur=$(squeue -h -u "${USER}" -n "${MODEL}_curator" -o %i) || { echo "squeue failed" >&2; exit 1; }
    if [ -z "${cur}" ]; then
        jid=$(sbatch --parsable --job-name="${MODEL}_curator" --export="${EXP}" "${HERE}/submit_curator.sbatch") && \
            echo "curator job ${jid%%;*}" || echo "the curator was NOT accepted; the GPU jobs curate themselves" >&2
    else
        echo "curator already queued or running (${cur})"
    fi
fi
sj=$(sbatch --parsable --job-name="${MODEL}_seeded" --export="${EXP},STREAM=seeded" --array=0-$((N_SEEDED - 1)) ${dep} \
     "${HERE}/submit_stream.sbatch") || { echo "seeded not submitted" >&2; exit 1; }
rj=$(sbatch --parsable --job-name="${MODEL}_random" --export="${EXP},STREAM=random" --array=0-$((N_RANDOM - 1)) ${dep} \
     "${HERE}/submit_stream.sbatch") || { echo "random not submitted" >&2; exit 1; }
rj=${rj%%;*}
# hops wait only for the FIRST random task to start, and only while the registry does not exist yet: `after:<array>`
# holds until every task of the array has started (acr_zp1_sep30, 2026-10-01: hops out of the queue for 18 h)
hdep="${dep}"
[ -f "${CDIR}/registry.pt" ] || hdep="--dependency=after:${rj}_0"
hj=$(sbatch --parsable --job-name="${MODEL}_hops" --export="${EXP},STREAM=hops" --array=0-$((N_HOPS - 1)) ${hdep} \
     "${HERE}/submit_stream.sbatch") || { echo "hops not submitted" >&2; exit 1; }
echo "${MODEL}: seeded ${sj%%;*} (0-$((N_SEEDED - 1))), random ${rj} (0-$((N_RANDOM - 1))), hops ${hj%%;*}" \
     "(0-$((N_HOPS - 1)); ${hdep:-no dependency}) -> ${CDIR} (MXtalTools $(cut -c1-12 "${CDIR}/launch_commit"));" \
     "progress in ${CDIR}/stats.md"
