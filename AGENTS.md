# MXtalTools repository instructions

## Purpose and priorities

Make the toolkit trustworthy to reuse while preserving its independence. Prefer simple, composable mechanisms and introduce only as much structure as current work requires.

MXtalTools is a standalone molecular/crystal toolkit and the scientific/data substrate used by GFN. The immediate priority is correctness and performance on interfaces exercised by current GFN crystal workflows. Conformer integration should wait for the active GFN conformer refactor. General trainer modernization, migration of old dataset indexing, and broad user-friendliness are second-tier work unless explicitly selected.

The production dependency is one-way: GFN depends on MXtalTools. MXtalTools must not acquire a runtime dependency on GFN.

## Authority by knowledge type

- Invariants and interfaces: current implementation, validation, and focused tests.
- Defaults: task-specific explicit defaults; there is no universal canonical MXtalTools training config.
- Workflows: named entry points and their smoke tests.
- Decisions: concise accepted choices not expressed in code or tests.
- Working assumptions: explicitly labelled and scoped.
- Observations: measurements and findings; evidence, not policy.
- History: old configs, trainers, WIP files, examples, and downstream citations; no current authority unless explicitly promoted.

Agent memories, prior-chat summaries, handoffs, and tool-specific state are navigation leads, not repository authority; recheck their material claims against the sources above before acting. Do not include `.claude/worktrees/` or similar generated/cache directories in repository-wide searches unless the task explicitly targets that worktree; a nested copy is never evidence about the main working tree.

If sources disagree, report the conflict rather than inferring policy from repetition or age.

## Proof model

Documentation should carry or point to the strongest practical proof for each material claim:

- Current behavior, invariants, and interfaces: implementation plus focused validation or tests where practical. Code proves what currently happens; it does not by itself prove that the behavior is correct or intended.
- Operational defaults and workflows: task-specific config, named entry point, and a smoke or contract check.
- Decisions, rationale, priorities, and intended direction: an explicit owner decision with clear scope. These claims may have no code proof.
- Working assumptions: an explicit label, scope, and condition for revisiting them.
- Observations and performance claims: reproducible inputs, measurements, or a named benchmark.

Prefer docs that point to proof over prose that duplicates detailed executable state. If proof and prose disagree, surface the conflict rather than forcing code to match the document or treating accidental behavior as policy. Label material claims that do not yet have adequate proof.

Do not use commit recency, commit messages, branch names, or apparent Git history to infer authority or intent. If historical context matters, preserve it explicitly as dated evidence; otherwise remove obsolete claims from active context.

## Owner adjudication and knowledge capture

Ask the owner only when missing intent would materially change the result. Present one direct question, lead with the recommended answer, give two or three mutually exclusive choices with their consequences, and state the default when proceeding without an answer would be safe. Keep it short. Do not turn status updates, facts discoverable from current sources, reversible implementation details, or low-consequence preferences into owner adjudications. Once answered, treat the decision as settled within its stated scope unless new evidence creates a real conflict.

Before proposing to store a claim as repository knowledge, name its class—**invariant, interface, default, workflow, decision, working assumption, observation, or history**—and its proposed authoritative home. Explain why persistence is warranted. Do not ask the owner to classify trivia or log routine implementation details. If code, config, or a focused test can express the claim self-evidently, encode it or point to it there instead of creating duplicative prose.

## Current GFN-facing boundary

Treat these groups as contracted where current GFN imports them:

- molecular/crystal data objects and collation;
- crystal latent/cell transforms, symmetry, canonicalization, construction, periodic indexing, and analysis;
- ELJ and cell-reduction calculations;
- model primitives directly used by GFN;
- conformer topology, perception, coordinate construction/Jacobians, energies, and priors after the conformer refactor settles.

MACE/UMA adapters are selectable MLIP integrations and active optimization targets; their dependencies remain optional for workflows that do not select them. Plotting, reporting, clustering, encoder-loading, and similar conveniences are internal or best-effort unless explicitly promoted.

This is a contract with the current jointly evolving GFN source, not a compatibility promise for historical versions. Do not treat downstream GFN comments or findings as the authoritative home of MXtalTools rationale; restate durable toolkit invariants locally.

## Indexing and crystal operations

`MolCrystalData.analyze(...)` is the shared dispatch surface through which GFN may select ELJ, MACE, UMA, or other `crystal_analysis.py` computations. ELJ is the cheap backend selected by GFN's current canonical config; MACE is used specifically for acridine; UMA is the more expensive general MLIP backend. Both MLIP interfaces are active optimization targets and need backend-specific correctness and performance evidence.

For canonical GFN ELJ training, the current contracted path is `MolCrystalData.analyze(['reduction_en', 'elj'])` -> `mol2cluster` -> `construct_radial_graph`/`build_radial_graph` -> eLJ analysis. The focused consumer proof lives in GFN's `test_mxtaltools_crystal_boundary.py` and must remain CPU/synthetic.

The rewritten on-device PBC neighbour list in `mlip_interfaces/pbc_neighbours.py` currently belongs to the MACE adapter; it is not the ELJ path and is not yet a universal crystal-indexing replacement. UMA has its own live interface and must not inherit MACE's indexing contract by inference. Older dataset-preparation indexing remains separate migration debt and must not be used to infer any live interface. Changes to periodic indexing, crystal construction, or energy behavior require focused equivalence/correctness tests at the relevant backend boundary. Performance claims require a named benchmark, not a comment.

## Change discipline and verification

- Keep the package independently importable and testable without GFN.
- Do not expand optional test-time imports from GFN; move genuinely shared policy to the appropriate lower layer or keep it independent.
- Do not modernize unrelated historical trainers/configs during a GFN-boundary change.
- Run focused CPU contracts and owning unit tests first, small synthetic integrations for cross-module changes, and GPU/real-data/MLIP/benchmark tests only when relevant.
- Old, `old_bad`, WIP, diagnostic, and task-specific tests are opt-in unless a current contract explicitly names them.

Before widening a task, check whether it advances a current GFN-facing contract or a specifically selected standalone MXtalTools goal, whether the knowledge belongs in code, tests, config, a decision, or evidence, and whether a new abstraction or document solves a demonstrated recurring problem. Stop and surface work that drifts into broad modernization, historical reconciliation, or the conformer refactor without explicit authorization.

## Tables

A table is a claim, not a grid of numbers. Every table an agent presents — in a report, in chat, or printed to stdout by a test, diagnostic, or analysis script — carries labels and a caption:

- **Labels.** Each column names the quantity and its unit or scale (`lattice energy (kJ/mol)`, `RDF distance`, `wall time (s/call)`); each row names its case — the structure, dataset, space group, backend, or code path, not an index. Never present a bare number whose meaning lives only in the surrounding conversation.
- **Caption.** One or two sentences above or below the table: what was measured, on what (dataset, structure set, backend, revision), how many samples each cell rests on, and what the reader is meant to compare. Where a number is only meaningful against a reference, give the reference — the baseline backend, the prior value, or the resolution below which differences are noise.

If a column cannot be given a quantity and a unit, it does not belong in the table. If the caption cannot say what the comparison shows, the table is not yet a result.

## Keeping context current

Stale prose is harmful context, not harmless history. Maintain active knowledge in two bounded ways:

1. **Event-driven:** when a change invalidates directly relevant prose, update it or explicitly demote it in the same change. Do not sweep unrelated documentation.
2. **Milestone-triggered:** at an owner-declared project milestone or during an explicitly requested dedicated audit, inspect only high-exposure context: `AGENTS.md`, README/routing material, active workflow documents, canonical-config comments, and accepted decisions.

Classify reviewed material as current policy, working assumption, observation, or history. A stale or unresolved document must not continue to sound current: correct it, mark its status prominently, or remove it from active context. Do not assume Git can recover missing intent; preserve important history explicitly and discard obsolete claims from places agents are expected to consult.

Freshness dates may help route attention but do not establish correctness. Verification against current code, config, and tests does.

When available, use `$audit-active-context` at an owner-declared project milestone or when active guidance may be stale or conflicting; do not run calendar audits or invoke it for every ordinary change by default. When unavailable, apply this file's bounded context-maintenance rules directly. Directly invalidated active guidance is still corrected event-by-event.

When available, use `$orchestrate-repository-work` only after the user authorizes delegation and the project has genuinely separable workstreams. When unavailable, keep one agent as the default; give any authorized agents narrow, non-overlapping deliverables and require evidence-backed handoffs to one integration owner. Collapse the organization when only sequential work remains. Skill names are optional environment capabilities, not repository dependencies; this file and scoped repository sources retain authority.

Comments and module documents are explanatory. Verify them against implementation and tests before relying on them. Add institutional documentation only when it prevents a concrete ambiguity that code, tests, or a task-specific config cannot express.
