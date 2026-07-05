"""
Dump Euclidean-normalizer data for general Wyckoff positions, standard (ITA
tabulated) setting, for all 230 space groups. Run locally where cctbx is
installed.

Fixes vs. the previous version:
  - vectors_and_moduli() returns ss_vec_mod objects with .v / .m attributes,
    not (vector, modulus) tuples -- fixed.
  - crystal_system pulled via space_group.crystal_system(), not
    space_group_info.crystal_system() (which doesn't exist).

Leaned down per request: only the (k2l=True, l2n=True) "full" normalizer
variant is kept for the point-operation generators, since that's the one
DIALS itself uses to build the full metric supergroup (i.e. it already is
"everything"). The other three flag combinations are dropped -- they were
only useful for provenance while debugging, not for building
NORMALIZER_TABLE itself.

sgtbx.space_group_info(number=N) already returns the ITA standard/tabulated
setting, so no extra change-of-basis step is needed for "standard setting".

Output: normalizer_dump.json, one entry per space group number (1-230):
  {
    number, hermann_mauguin, hall_symbol, crystal_system,
    order_z, is_chiral, is_centrosymmetric,
    normalizer_index,              # expanded_order_z / order_z
    normalizer_extra_ops: [        # point-op generators beyond G
      {xyz, rotation, translation}, ...
    ],
    seminvariants: {
      vectors_and_moduli: [{vector, modulus}, ...],  # modulus 0 => continuous ("inf")
      continuous_shifts_are_principal: bool,
      principal_continuous_shift_flags: [bool,bool,bool]  # only if principal
    }
  }
"""

import json
import traceback

from cctbx import sgtbx


def op_to_dict(rt_mx):
    r = rt_mx.r().as_double()
    t = rt_mx.t().as_double()
    return {
        "xyz": rt_mx.as_xyz(),
        "rotation": [list(r[0:3]), list(r[3:6]), list(r[6:9])],
        "translation": list(t),
    }


def group_op_xyz_set(group):
    return {group(i).as_xyz() for i in range(group.order_z())}


def extra_ops(base_group, expanded_group):
    base_xyz = group_op_xyz_set(base_group)
    return [
        op_to_dict(expanded_group(i))
        for i in range(expanded_group.order_z())
        if expanded_group(i).as_xyz() not in base_xyz
    ]


def dump_seminvariants(group):
    ss = sgtbx.structure_seminvariants(group)
    result = {}
    try:
        result["vectors_and_moduli"] = [
            {"vector": list(entry.v), "modulus": entry.m}
            for entry in ss.vectors_and_moduli()
        ]
    except Exception as e:
        result["vectors_and_moduli_error"] = repr(e)
        result["vectors_and_moduli_trace"] = traceback.format_exc()

    try:
        principal = ss.continuous_shifts_are_principal()
        result["continuous_shifts_are_principal"] = principal
        if principal:
            result["principal_continuous_shift_flags"] = list(
                ss.principal_continuous_shift_flags()
            )
    except Exception as e:
        result["continuous_shifts_error"] = repr(e)

    return result


def dump_one(number):
    entry = {"number": number}
    try:
        sg_info = sgtbx.space_group_info(number=number)
        group = sg_info.group()
        sg_type = sg_info.type()

        entry["hall_symbol"] = sg_type.hall_symbol()
        entry["hermann_mauguin"] = sgtbx.space_group_symbols(number).hermann_mauguin()
        entry["order_z"] = group.order_z()
        entry["is_chiral"] = group.is_chiral()

        try:
            entry["is_centrosymmetric"] = group.is_centric()
        except Exception as e:
            entry["is_centrosymmetric_error"] = repr(e)

        try:
            entry["crystal_system"] = str(group.crystal_system())
        except Exception as e:
            entry["crystal_system_error"] = repr(e)

        try:
            expanded = sg_type.expand_addl_generators_of_euclidean_normalizer(True, True)
            entry["normalizer_index"] = expanded.order_z() / group.order_z()
            entry["normalizer_extra_ops"] = extra_ops(group, expanded)
        except Exception as e:
            entry["normalizer_error"] = repr(e)
            entry["normalizer_trace"] = traceback.format_exc()

        entry["seminvariants"] = dump_seminvariants(group)

    except Exception as e:
        entry["fatal_error"] = repr(e)
        entry["trace"] = traceback.format_exc()

    return entry


def main():
    results = {}
    for number in range(1, 231):
        print(f"Processing space group {number}...")
        results[str(number)] = dump_one(number)

    with open("normalizer_dump.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done. Wrote normalizer_dump.json")
    n_errors = sum(1 for v in results.values() if "fatal_error" in v)
    print(f"{n_errors} space groups hit a fatal error.")


if __name__ == "__main__":
    main()


"""
Reduce the raw normalizer dump into an actual NORMALIZER_TABLE of
(rotation, translation) coset representatives, matching the SG14 format.

Two independent pieces are combined per space group:
  1. Point-operation coset reps: deduplicated from `normalizer_extra_ops`
     (which is a raw set-difference, not coset reps -- e.g. 12 raw ops for
     SG75 collapse to 3 true coset reps + identity = index 4).
  2. Translation coset reps: built directly from the finite (non-continuous)
     seminvariant vectors/moduli -- all combinations of k_i/m_i * v_i.

Final table entries = Cartesian product of the two, minus the identity/zero
entry -- giving exactly `multiplicity - 1` entries per group, same shape as
your existing SG14 entry.

NOTE ON COMPOSITION ORDER: point reps have zero translation and translation
reps have identity rotation, so combining them as (R_i, t_j) directly
(rather than R_i @ t_j) is one valid choice of representative for that
coset -- not the only one, but correctness of *coverage* doesn't depend on
which valid representative is picked, since downstream folding re-canonicalizes
via transform_aunit_params anyway. Flagging this so you can sanity check it
matches your composition convention if results look off.

Run this locally where cctbx is installed, using rt_mx objects directly
(not the JSON dump). Note: rt_mx composition is via the .multiply(rhs)
method, NOT the * operator (the latter raises TypeError in this binding).
"""

import json
import itertools

from cctbx import sgtbx


def dedup_point_cosets(base_group, expanded_group):
    """Return list of rt_mx reps, one per coset of base_group in expanded_group,
    INCLUDING the identity coset (so len == point_op_index)."""
    base_ops = [base_group(i) for i in range(base_group.order_z())]
    reps = []
    covered = set()

    for i in range(expanded_group.order_z()):
        op = expanded_group(i)
        xyz = op.as_xyz()
        if xyz in covered:
            continue
        reps.append(op)
        for b in base_ops:
            covered.add(op.multiply(b).as_xyz())

    return reps


def translation_coset_reps(seminvariant_entries):
    """seminvariant_entries: list of {'vector':[...], 'modulus': m} with m > 0 only.
    Returns list of fractional translation vectors (tuples), INCLUDING (0,0,0)."""
    finite = [e for e in seminvariant_entries if e["modulus"] > 0]
    if not finite:
        return [(0.0, 0.0, 0.0)]

    per_vector_steps = []
    for e in finite:
        v = e["vector"]
        m = e["modulus"]
        steps = [tuple((k / m) * c for c in v) for k in range(m)]
        per_vector_steps.append(steps)

    reps = []
    for combo in itertools.product(*per_vector_steps):
        t = [0.0, 0.0, 0.0]
        for step in combo:
            t = [t[i] + step[i] for i in range(3)]
        t = tuple(x % 1.0 for x in t)
        reps.append(t)

    # dedup in case of overlap (shouldn't happen if vectors are independent,
    # but cheap to guard)
    return list(dict.fromkeys(reps))


def build_table_entry(sg_info, base_group):
    sg_type = sg_info.type()
    expanded = sg_type.expand_addl_generators_of_euclidean_normalizer(True, True)

    point_reps = dedup_point_cosets(base_group, expanded)

    ss = sgtbx.structure_seminvariants(base_group)
    seminvariant_entries = [
        {"vector": list(e.v), "modulus": e.m} for e in ss.vectors_and_moduli()
    ]
    continuous_dims = [e["vector"] for e in seminvariant_entries if e["modulus"] == 0]
    t_reps = translation_coset_reps(seminvariant_entries)

    entries = []
    for op in point_reps:
        r = op.r().as_double()
        R = [list(r[0:3]), list(r[3:6]), list(r[6:9])]
        t_point = op.t().as_double()  # actual translation baked into this coset rep,
                                       # NOT assumed zero -- e.g. SG4's mirror carries
                                       # translation [0, -0.5, 0], not [0, 0, 0]
        for t_semi in t_reps:
            t_total = tuple(
                (t_point[i] + t_semi[i]) % 1.0 for i in range(3)
            )
            if op.as_xyz() == "x,y,z" and t_total == (0.0, 0.0, 0.0):
                continue  # skip identity
            entries.append((R, list(t_total)))

    return entries, continuous_dims


def main():
    table = {}
    continuous_flags = {}

    for number in range(1, 231):
        print(f"Processing space group {number}...")
        try:
            sg_info = sgtbx.space_group_info(number=number)
            group = sg_info.group()
            entries, continuous_dims = build_table_entry(sg_info, group)
            table[number] = entries
            if continuous_dims:
                continuous_flags[number] = continuous_dims
        except Exception as e:
            table[number] = {"error": repr(e)}

    with open("normalizer_table.json", "w") as f:
        json.dump({"table": table, "continuous_dims": continuous_flags}, f, indent=2)

    with open("normalizer_table.json") as f:
        data = json.load(f)

    table = data["table"]
    continuous_dims = data["continuous_dims"]
    padded = {}
    for sg in range(1, 231):
        dims = continuous_dims.get(str(sg), [])
        n_pad = 3 - len(dims)
        padded[str(sg)] = dims + [[0, 0, 0]] * n_pad
    print("Wrote normalizer_table.json")

if __name__ == "__main__":
    main()
