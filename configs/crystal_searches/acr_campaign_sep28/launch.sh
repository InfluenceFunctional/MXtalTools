#!/bin/bash
# acr_campaign_sep28: set up the campaign directory, check the energy model and the code, submit the streams.
#   bash launch.sh [n_tasks_per_stream] [stream ...]      (default 4 tasks each of random, eljstart and hops)
# First launch: freezes coord.yaml and streams/*.yaml into the campaign directory (the jobs read the frozen copies) and
# records the MXtalTools commit. A relaunch -- after a walltime, or to add one stream's tasks (bash launch.sh 4 hops) --
# resumes every task's run, and refuses when:
#   - coord.yaml or a stream config differs from the frozen copy (start a new campaign directory instead);
#   - this checkout is at another commit than the first launch (ALLOW_CODE_CHANGE=1 overrides; each shard records the
#     commit of the job that wrote it);
#   - tasks of a stream it would submit are still queued or running (FORCE=1 overrides), or squeue cannot say.
# hops waits for the random array to start (the first curate pass, which writes hop_parents.pt, runs in whichever job
# starts first); a hop job that finds no hop_parents.pt waits for it, up to a bound, before failing.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MXT=$(cd "${HERE}/../../.." && pwd)
N=${1:-4}
shift || true
STREAMS=("$@")
if [ ${#STREAMS[@]} -eq 0 ]; then STREAMS=(random eljstart hops); fi
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
for s in "${STREAMS[@]}"; do
    if [ ! -f "${HERE}/streams/${s}.yaml" ]; then echo "no stream config streams/${s}.yaml" >&2; exit 1; fi
done

# freeze the campaign's definition on the first launch; afterwards it must not change under the running campaign
freeze() {
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
mkdir -p "${CAMP}/runs" "${CAMP}/streams"
# the first run of this launcher on a campaign an earlier launcher started (it left no launch_commit): stop files there
# were written under the earlier stop rule, which the relaunched code replaces; they are moved aside, not honoured
if [ ! -f "${CAMP}/launch_commit" ] && [ -f "${CAMP}/registry.pt" ]; then
    OLDSTOP=$(ls "${CAMP}"/STOP "${CAMP}"/STOP.* 2>/dev/null || true)
    if [ -n "${OLDSTOP}" ]; then
        mkdir -p "${CAMP}/superseded_stop_files"
        mv ${OLDSTOP} "${CAMP}/superseded_stop_files/"
        echo "moved stop files written under the earlier stop rule to ${CAMP}/superseded_stop_files/:" ${OLDSTOP}
    fi
fi
freeze "${HERE}/coord.yaml" "${CAMP}/coord.yaml"
for f in "${HERE}"/streams/*.yaml; do freeze "${f}" "${CAMP}/streams/$(basename "${f}")"; done
COMMIT=$(git -C "${MXT}" rev-parse HEAD)
if [ -f "${CAMP}/launch_commit" ]; then
    FIRST=$(cat "${CAMP}/launch_commit")
    if [ "${FIRST}" != "${COMMIT}" ] && [ "${ALLOW_CODE_CHANGE:-0}" != "1" ]; then
        echo "the campaign was launched at MXtalTools ${FIRST}; this checkout is ${COMMIT}. Check out ${FIRST}, or set" \
             "ALLOW_CODE_CHANGE=1 to continue under the new code (shards record each job's commit)." >&2
        exit 1
    fi
else
    echo "${COMMIT}" > "${CAMP}/launch_commit"
fi

# never submit a second copy of a live task: it would fight the running one for its run lease
for s in "${STREAMS[@]}"; do  # acr_camp: every stream's tasks of a launch made before streams had their own names
    LIVE=$(squeue -h -u "${USER}" -n "acr_camp_${s},acr_camp" -o %i) || { echo "squeue failed: cannot check for live tasks" >&2; exit 1; }
    if [ -n "${LIVE}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "stream ${s} (or an earlier launch's acr_camp array) still has queued or running tasks" \
             "($(echo ${LIVE} | tr '\n' ' ')): not resubmitting (scancel them first, or FORCE=1)" >&2
        exit 1
    fi
done

# the curator: the campaign's curate pass on CPU every 15 min, so no GPU job spends GPU time on it (they curate only
# if it is gone: their fallback interval is 1 h). CURATOR=0 skips it.
if [ "${CURATOR:-1}" = "1" ]; then
    CUR=$(squeue -h -u "${USER}" -n acr_camp_curator -o %i) || { echo "squeue failed: cannot check for a curator" >&2; exit 1; }
    if [ -z "${CUR}" ]; then
        if CJ=$(sbatch --parsable "${HERE}/submit_curator.sbatch"); then  # a refused curator must not stop the launch
            echo "submitted curator: job ${CJ%%;*}"
        else
            echo "the curator job was NOT accepted by sbatch; the GPU jobs will curate themselves (1 h fallback)" >&2
        fi
    else
        echo "curator already queued or running ($(echo ${CUR} | tr '\n' ' '))"
    fi
fi

LAST=$((N - 1))
RJ=""
for s in "${STREAMS[@]}"; do
    DEP=""
    if [ "${s}" = "hops" ] && [ -n "${RJ}" ]; then DEP="--dependency=after:${RJ}"; fi
    JID=$(sbatch --parsable --job-name="acr_camp_${s}" --export=ALL,STREAM="${s}" --array=0-${LAST} ${DEP} \
          "${HERE}/submit_stream.sbatch")
    JID=${JID%%;*}
    echo "submitted ${s}: array ${JID} (tasks 0-${LAST})${DEP:+, ${DEP}}"
    if [ "${s}" = "random" ]; then RJ=${JID}; fi
done
echo "campaign ${CAMP} (MXtalTools $(cat "${CAMP}/launch_commit" | cut -c1-12)): progress in ${CAMP}/stats.md"
