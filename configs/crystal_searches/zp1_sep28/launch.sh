#!/bin/bash
# zp1_sep28: set up and submit campaigns of make_campaigns.py (mipcas_elj, mipcas_uma, nehzor_elj, nehzor_uma).
#   bash launch.sh all                      every campaign
#   bash launch.sh nehzor_uma mipcas_elj    the named ones
# Per campaign, the first launch freezes coord.yaml and streams/*.yaml into the campaign directory (the jobs read the
# frozen copies) and records the MXtalTools commit. A relaunch -- after a walltime -- resumes every task's run, and
# refuses (that campaign only; the others still launch) when:
#   - coord.yaml or a stream config differs from the frozen copy (start a new campaign directory instead);
#   - this checkout is at another commit than the first launch (ALLOW_CODE_CHANGE=1 overrides);
#   - tasks of the campaign are still queued or running (FORCE=1 overrides), or squeue cannot say;
#   - a UMA campaign's model file on the cluster is not the one its coord.yaml was built for.
# Submits the curator (CPU; CURATOR=0 skips it), the random array and the hops array (hops start after the random array
# does: a hop job waits, up to a bound, for the first curate pass to write hop_parents.pt). Tasks per stream: 4 + 4 for
# eLJ, 3 + 3 for UMA (N_RANDOM / N_HOPS override for every campaign of this call).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MXT=$(cd "${HERE}/../../.." && pwd)
DATA=/scratch/mk8347/data/crystal_datasets
UMA=/scratch/mk8347/models/uma/esen_s.pt
ALL=(mipcas_elj mipcas_uma nehzor_elj nehzor_uma)
if [ $# -eq 0 ]; then echo "usage: bash launch.sh all | <campaign> ...  (${ALL[*]})" >&2; exit 1; fi
if [ "$1" = "all" ]; then NAMES=("${ALL[@]}"); else NAMES=("$@"); fi
COMMIT=$(git -C "${MXT}" rev-parse HEAD)

model_id() {  # coordinator.energy_model_id for a uma file
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
print(f'uma:{os.path.basename(p)}:{size}:{h.hexdigest()[:16]}')
PY
}

freeze() {  # copy on the first launch; afterwards the copy must not change under the running campaign
    local src=$1 dst=$2
    if [ -f "${dst}" ]; then
        if ! cmp -s "${src}" "${dst}"; then
            echo "${dst} differs from ${src}: start a new campaign directory rather than change a running one" >&2
            return 1
        fi
    else
        cp "${src}" "${dst}"
    fi
}

launch_one() {
    local name=$1 mol energy cdir n_r n_h live cur rj jid dep
    case "${name}" in
        mipcas_elj|mipcas_uma|nehzor_elj|nehzor_uma) ;;
        *) echo "unknown campaign ${name} (${ALL[*]})" >&2; return 1 ;;
    esac
    mol=${name%_*}
    energy=${name#*_}
    cdir=${DATA}/${mol}/campaigns/${name}_sep28
    if [ "${energy}" = "uma" ]; then
        local want mid
        want=$(grep '^energy_model_id:' "${HERE}/${name}/coord.yaml" | awk '{print $2}')
        mid=$(model_id "${UMA}") || { echo "${name}: cannot read ${UMA}" >&2; return 1; }
        if [ "${mid}" != "${want}" ]; then
            echo "${name}: the cluster's UMA model is ${mid}; the campaign was built for ${want}" >&2
            return 1
        fi
        n_r=${N_RANDOM:-3}; n_h=${N_HOPS:-3}
    else
        n_r=${N_RANDOM:-4}; n_h=${N_HOPS:-4}
    fi
    mkdir -p "${cdir}/runs" "${cdir}/streams" || return 1
    freeze "${HERE}/${name}/coord.yaml" "${cdir}/coord.yaml" || return 1
    for s in random hops; do freeze "${HERE}/${name}/streams/${s}.yaml" "${cdir}/streams/${s}.yaml" || return 1; done
    if [ -f "${cdir}/launch_commit" ]; then
        if [ "$(cat "${cdir}/launch_commit")" != "${COMMIT}" ] && [ "${ALLOW_CODE_CHANGE:-0}" != "1" ]; then
            echo "${name}: launched at MXtalTools $(cat "${cdir}/launch_commit"); this checkout is ${COMMIT}. Check out" \
                 "that commit, or set ALLOW_CODE_CHANGE=1 to continue under the new code" >&2
            return 1
        fi
    else
        echo "${COMMIT}" > "${cdir}/launch_commit"
    fi
    live=$(squeue -h -u "${USER}" -n "z1_${name}_random,z1_${name}_hops" -o %i) || {
        echo "${name}: squeue failed, cannot check for live tasks" >&2; return 1; }
    if [ -n "${live}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "${name}: tasks still queued or running ($(echo ${live} | tr '\n' ' ')); not resubmitting (FORCE=1)" >&2
        return 1
    fi
    if [ "${CURATOR:-1}" = "1" ]; then
        cur=$(squeue -h -u "${USER}" -n "z1_${name}_curator" -o %i) || { echo "${name}: squeue failed" >&2; return 1; }
        if [ -z "${cur}" ]; then
            if jid=$(sbatch --parsable --job-name="z1_${name}_curator" --export=ALL,CAMP="${cdir}" \
                     "${HERE}/submit_curator.sbatch"); then
                echo "${name}: curator job ${jid%%;*}"
            else
                echo "${name}: the curator was NOT accepted; the GPU jobs curate themselves (1 h fallback)" >&2
            fi
        else
            echo "${name}: curator already queued or running (${cur})"
        fi
    fi
    rj=$(sbatch --parsable --job-name="z1_${name}_random" --export=ALL,CAMP="${cdir}",STREAM=random \
         --array=0-$((n_r - 1)) "${HERE}/submit_stream.sbatch") || { echo "${name}: random not submitted" >&2; return 1; }
    rj=${rj%%;*}
    jid=$(sbatch --parsable --job-name="z1_${name}_hops" --export=ALL,CAMP="${cdir}",STREAM=hops \
          --array=0-$((n_h - 1)) --dependency=after:${rj} "${HERE}/submit_stream.sbatch") || {
        echo "${name}: hops not submitted" >&2; return 1; }
    echo "${name}: random array ${rj} (tasks 0-$((n_r - 1))), hops array ${jid%%;*} (tasks 0-$((n_h - 1)), after ${rj})" \
         "-> ${cdir} (MXtalTools $(cut -c1-12 "${cdir}/launch_commit")); progress in ${cdir}/stats.md"
}

FAILED=0
for name in "${NAMES[@]}"; do
    launch_one "${name}" || FAILED=$((FAILED + 1))
done
if [ ${FAILED} -gt 0 ]; then echo "${FAILED} campaign(s) not launched: see the messages above" >&2; exit 1; fi
