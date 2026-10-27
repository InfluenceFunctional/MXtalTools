#!/bin/bash
# acr_fm_oct27: submit the calibration job of each model (default: every model in campaign.MODELS).
#   bash calibrate.sh [MODEL ...]
# Each job writes /scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_fm_oct27/calibration/<MODEL>/result.json;
# make_campaign.py reads a copy of it (calibration/<MODEL>.json beside this script) for the campaign's identity cut.
# Refuses a model whose checkpoint, conformer or extra PYTHONPATH directory is missing on the cluster, or whose result
# already exists (delete the directory to recalibrate).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=/scratch/mk8347/data/crystal_datasets/acridine/campaigns/acr_fm_oct27
rows=$(cd "${HERE}" && python3 campaign.py) || { echo "cannot read campaign.MODELS" >&2; exit 1; }
want="$*"
echo "${rows}" | while read -r M K P MOL PX; do
    if [ -n "${want}" ] && [[ " ${want} " != *" ${M} "* ]]; then continue; fi
    OUT=${ROOT}/calibration/${M}
    for f in "${P}" "${MOL}"; do [ -f "${f}" ] || { echo "${M}: ${f} missing, not submitted" >&2; continue 2; }; done
    [ "${PX}" = "-" ] || [ -d "${PX}" ] || { echo "${M}: ${PX} missing, not submitted" >&2; continue; }
    [ -f "${OUT}/result.json" ] && { echo "${M}: ${OUT}/result.json exists, not submitted" >&2; continue; }
    PY=""; [ "${PX}" = "-" ] || PY=${PX}
    j=$(sbatch --parsable --job-name=fm_calib_${M} --export=ALL,MODEL=${M},OUT=${OUT},PYEXTRA=${PY} \
        "${HERE}/submit_calibrate.sbatch") && echo "${M}: job ${j%%;*} -> ${OUT} (log /scratch/mk8347/logs/fm_calib_${M}-${j%%;*}.out)"
done
