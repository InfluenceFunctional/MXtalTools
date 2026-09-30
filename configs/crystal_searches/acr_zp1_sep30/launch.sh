#!/bin/bash
# acr_zp1_sep30: set up and submit the campaign of make_campaign.py.
#   bash launch.sh
# The first launch freezes coord.yaml and streams/*.yaml into the campaign directory (the jobs read the frozen copies)
# and records the MXtalTools commit. A relaunch -- after the 6 h walltime -- resumes every task, and refuses when:
#   - coord.yaml or a stream config differs from the frozen copy (start a new campaign directory instead);
#   - this checkout is at another commit than the first launch (ALLOW_CODE_CHANGE=1 overrides);
#   - tasks of the campaign are still queued or running (FORCE=1 overrides), or squeue cannot say;
#   - the MACE file on the cluster is not the one coord.yaml was built for.
# Submits the preflight job (CPU; skipped once <campaign>/preflight_ok exists), the curator (CPU; CURATOR=0 skips it),
# then the seeded and random arrays (afterok on the preflight) and the hops array (after the random array starts: a hop
# job waits, up to a bound, for the first basin). Tasks: random 12, hops 12, seeded 8 (N_RANDOM / N_HOPS / N_SEEDED
# override; the seeded count must match the seed files, i.e. make_campaign.TASKS['seeded']).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MXT=$(cd "${HERE}/../../.." && pwd)
CDIR=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_zp1_sep30
MODEL=/scratch/mk8347/data/acr_newmodel.model
COMMIT=$(git -C "${MXT}" rev-parse HEAD)
N_R=${N_RANDOM:-12}; N_H=${N_HOPS:-12}; N_S=${N_SEEDED:-8}

model_id() {  # coordinator.energy_model_id for a mace file
    python3 - "$1" <<'PY'
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

want=$(grep '^energy_model_id:' "${HERE}/coord.yaml" | awk '{print $2}')
mid=$(model_id "${MODEL}") || { echo "cannot read ${MODEL}" >&2; exit 1; }
[ "${mid}" = "${want}" ] || { echo "the cluster's MACE model is ${mid}; the campaign was built for ${want}" >&2; exit 1; }
[ "${N_S}" = "8" ] || { echo "N_SEEDED must equal the number of seed files (8)" >&2; exit 1; }
mkdir -p "${CDIR}/runs" "${CDIR}/streams" || exit 1
freeze "${HERE}/coord.yaml" "${CDIR}/coord.yaml"
for s in random hops seeded; do freeze "${HERE}/streams/${s}.yaml" "${CDIR}/streams/${s}.yaml"; done
if [ -f "${CDIR}/launch_commit" ]; then
    if [ "$(cat "${CDIR}/launch_commit")" != "${COMMIT}" ] && [ "${ALLOW_CODE_CHANGE:-0}" != "1" ]; then
        echo "launched at MXtalTools $(cat "${CDIR}/launch_commit"); this checkout is ${COMMIT}. Check out that commit," \
             "or set ALLOW_CODE_CHANGE=1 to continue under the new code" >&2
        exit 1
    fi
else
    echo "${COMMIT}" > "${CDIR}/launch_commit"
fi
live=$(squeue -h -u "${USER}" -n a1_random,a1_hops,a1_seeded,a1_preflight -o %i) || {
    echo "squeue failed, cannot check for live tasks" >&2; exit 1; }
if [ -n "${live}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "tasks still queued or running ($(echo ${live} | tr '\n' ' ')); not resubmitting (FORCE=1)" >&2
    exit 1
fi

dep=""
if [ ! -f "${CDIR}/preflight_ok" ]; then
    pj=$(sbatch --parsable --export=ALL,CAMP="${CDIR}" "${HERE}/submit_preflight.sbatch") || {
        echo "preflight not submitted" >&2; exit 1; }
    pj=${pj%%;*}
    dep="--dependency=afterok:${pj}"
    echo "preflight job ${pj} (the arrays start only if it succeeds; log /scratch/mk8347/logs/a1_preflight-${pj}.out)"
fi
if [ "${CURATOR:-1}" = "1" ]; then
    cur=$(squeue -h -u "${USER}" -n a1_curator -o %i) || { echo "squeue failed" >&2; exit 1; }
    if [ -z "${cur}" ]; then
        jid=$(sbatch --parsable --export=ALL,CAMP="${CDIR}" "${HERE}/submit_curator.sbatch") && \
            echo "curator job ${jid%%;*}" || echo "the curator was NOT accepted; the GPU jobs curate themselves" >&2
    else
        echo "curator already queued or running (${cur})"
    fi
fi
sj=$(sbatch --parsable --job-name=a1_seeded --export=ALL,CAMP="${CDIR}",STREAM=seeded --array=0-$((N_S - 1)) ${dep} \
     "${HERE}/submit_stream.sbatch") || { echo "seeded not submitted" >&2; exit 1; }
rj=$(sbatch --parsable --job-name=a1_random --export=ALL,CAMP="${CDIR}",STREAM=random --array=0-$((N_R - 1)) ${dep} \
     "${HERE}/submit_stream.sbatch") || { echo "random not submitted" >&2; exit 1; }
rj=${rj%%;*}
hj=$(sbatch --parsable --job-name=a1_hops --export=ALL,CAMP="${CDIR}",STREAM=hops --array=0-$((N_H - 1)) \
     --dependency=after:${rj} "${HERE}/submit_stream.sbatch") || { echo "hops not submitted" >&2; exit 1; }
echo "seeded ${sj%%;*} (0-$((N_S - 1))), random ${rj} (0-$((N_R - 1))), hops ${hj%%;*} (0-$((N_H - 1)), after ${rj})" \
     "-> ${CDIR} (MXtalTools $(cut -c1-12 "${CDIR}/launch_commit")); progress in ${CDIR}/stats.md"
