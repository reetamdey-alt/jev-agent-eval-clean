#!/usr/bin/env python
"""Build the release-v2.0 canonical case store (spec sections 6, 50, 78).

Composes base cases from:
- legacy v1 core (240 cases, migrated per spec 77)
- public source adapters (SWE-bench, SWE-bench-Live, BFCL V4, Terminal-Bench,
  TUA-Bench, tau2, AgentBench, IFEval, IFBench, OSWorld, WebArena-Verified,
  WorkArena, Online-Mind2Web) — execution/static transformations per spec 31
- deterministic synthetic generators (12 families per spec 32)
- derived metamorphic variants (never primary samples)

Emits:
- datasets/release-v2/cases.jsonl        (all primary base cases)
- datasets/release-v2/variants.jsonl     (metamorphic/counterfactual variants)
- datasets/release-v2/manifest.yaml      (composition, checksums, provenance)
- datasets/holdout/holdout.jsonl         (private holdout split)

The 6.1 capability allocation table is the target; realized counts are
recorded in the manifest. Splits follow spec 7.3 dev-split scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_agent_eval.schemas.case import CanonicalCase, CaseSplit  # noqa: E402

SEED = 20260922

# Spec 6.1 capability allocation table (base cases) — the 16 spec families.
# tool_relevance/tool_sufficiency fold into tool_necessity; permission_gating
# folds into tool_risk (spec 6.1 allows cross-capability counting via
# secondary capabilities; primary counting stays one-capability-per-case).
CAPABILITY_TARGETS: dict[str, int] = {
    "task_routing": 900,
    "tool_necessity": 900,
    "tool_risk": 1500,
    "user_intent_alignment": 1000,
    "goal_completion": 900,
    "stop_continue": 900,
    "progress_stuck": 800,
    "claim_evidence": 1000,
    "model_escalation": 700,
    "context_relevance": 800,
    "instruction_constraints": 700,
    "prompt_injection": 1000,
    "recovery": 700,
    "state_transition": 600,
    "uncertainty_abstention": 500,
    "trajectory_control": 900,
}

# Spec 6.3 difficulty mix per capability.
DIFFICULTY_MIX = {"easy": 0.15, "medium": 0.45, "hard": 0.30, "expert": 0.10}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def load_jsonl(path: Path) -> list[CanonicalCase]:
    cases = []
    for line in path.read_text().splitlines():
        if line.strip():
            cases.append(CanonicalCase.model_validate(json.loads(line)))
    return cases


def dump_jsonl(cases: list[CanonicalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")


def assign_difficulty(c: CanonicalCase, rng: random.Random) -> CanonicalCase:
    """Assign difficulty from the spec-6.3 mix if not already set."""
    if c.difficulty:
        return c
    r = rng.random()
    if r < DIFFICULTY_MIX["easy"]:
        c.difficulty = "easy"
    elif r < DIFFICULTY_MIX["easy"] + DIFFICULTY_MIX["medium"]:
        c.difficulty = "medium"
    elif r < DIFFICULTY_MIX["easy"] + DIFFICULTY_MIX["medium"] + DIFFICULTY_MIX["hard"]:
        c.difficulty = "hard"
    else:
        c.difficulty = "expert"
    return c


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", default=str(ROOT / "datasets" / "public"))
    parser.add_argument("--internal-dir", default=str(ROOT / "datasets" / "internal"))
    parser.add_argument("--out-dir", default=str(ROOT / "datasets" / "release-v2"))
    parser.add_argument("--holdout-dir", default=str(ROOT / "datasets" / "holdout"))
    parser.add_argument("--target-total", type=int, default=13800)
    parser.add_argument("--holdout-target", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(SEED)
    public_dir = Path(args.public_dir)
    internal_dir = Path(args.internal_dir)
    out_dir = Path(args.out_dir)
    holdout_dir = Path(args.holdout_dir)

    base_cases: list[CanonicalCase] = []
    composition: Counter = Counter()
    source_mix: Counter = Counter()

    # ------------------------------------------------------------------
    # 1. Legacy v1 core migration (spec 77)
    # ------------------------------------------------------------------
    legacy_count = 0
    for f in sorted(internal_dir.glob("*.jsonl")):
        for c in load_jsonl(f):
            migrated = c.model_copy(
                update={
                    "dataset": "legacy-v1-core",
                    "dev_split": CaseSplit.LEGACY,
                    "source": "legacy_v1",
                    "source_ref": c.case_id,
                    "cluster_key": c.cluster_key or f"legacy-{f.stem}",
                }
            )
            base_cases.append(migrated)
            legacy_count += 1
            composition[c.capability] += 1
            source_mix["legacy_v1"] += 1
    print(f"legacy-v1-core: {legacy_count} cases")

    # ------------------------------------------------------------------
    # 1b. Real coding-agent trace ingestion (spec 29; source class A)
    # ------------------------------------------------------------------
    from jev_agent_eval.ingestion.claude_transcripts import ingest_transcripts

    trace_roots = [Path.home() / ".claude" / "projects"]
    trace_cases, trace_stats = ingest_transcripts(trace_roots, max_sessions=500, seed=SEED)
    for c in trace_cases:
        base_cases.append(c)
        composition[c.capability] += 1
        source_mix["real_trace"] += 1
    print(f"real traces: {trace_stats}")

    # ------------------------------------------------------------------
    # 2. Public source adapters (spec 31 transformations)
    # ------------------------------------------------------------------
    from jev_agent_eval.datasets.adapters_v2 import build_public_cases

    pub_cases, pub_meta = build_public_cases(public_dir)
    for c in pub_cases:
        c = assign_difficulty(c, rng)
        base_cases.append(c)
        composition[c.capability] += 1
        source_mix[c.source or "public"] += 1
    for name, n in sorted(pub_meta.items()):
        print(f"public:{name}: {n} cases")

    # ------------------------------------------------------------------
    # 2b. Cap over-filled capabilities: subsample public cases per capability
    # toward the 6.1 targets (keeps source diversity, equalizes families).
    # ------------------------------------------------------------------
    capped: list[CanonicalCase] = []
    cap_counts: Counter = Counter()

    # Legacy + weakest-provenance first out: keep benchmark_exact over weak_inference.
    def _provenance_rank(c: CanonicalCase) -> int:
        g = next(iter(c.gold.values()), None)
        prov = str(getattr(g, "provenance", "") or "")
        if "benchmark" in prov or "execution" in prov:
            return 0
        if "rule" in prov:
            return 1
        return 2

    for c in sorted(base_cases, key=lambda c: (c.capability, _provenance_rank(c), c.case_id)):
        target = CAPABILITY_TARGETS.get(c.capability)
        if target is not None and cap_counts[c.capability] >= target:
            continue
        capped.append(c)
        cap_counts[c.capability] += 1
    # Keep capabilities without a 6.1 target (legacy extras) as-is.
    base_cases = capped
    print(f"after capability caps: {len(base_cases)} cases")
    from jev_agent_eval.generators.families import (
        GENERATOR_REGISTRY,
        MinimalPairGenerator,
    )

    # Capability coverage gap: fill toward 6.1 targets with generators.
    per_gen = max(60, (args.target_total - len(base_cases)) // (len(GENERATOR_REGISTRY) * 3))
    syn_parents: list[CanonicalCase] = []
    for gname, gcls in sorted(GENERATOR_REGISTRY.items()):
        gen = gcls()
        for c in gen.generate(seed=SEED + hash(gname) % 1000, n=per_gen):
            c = assign_difficulty(c, rng)
            base_cases.append(c)
            syn_parents.append(c)
            composition[c.capability] += 1
            source_mix["synthetic"] += 1
    print(f"synthetic generators: {len(syn_parents)} base cases")

    # ------------------------------------------------------------------
    # 4. Fill remaining capability gaps with balanced generator output.
    # Targets are inflated by the projected holdout take so the PUBLIC
    # release retains the 6.1 allocation after holdout split-off.
    # ------------------------------------------------------------------
    holdout_per_cap_estimate = max(1, args.holdout_target // len(CAPABILITY_TARGETS))
    gaps = {
        cap: tgt + holdout_per_cap_estimate - composition[cap]
        for cap, tgt in CAPABILITY_TARGETS.items()
    }
    gaps = {cap: n for cap, n in gaps.items() if n > 0}
    # Capability->generator routing for gap filling
    capability_generators: dict[str, list[str]] = {
        "tool_risk": ["minimal_pair", "composition", "adversarial"],
        "permission_gating": ["boundary", "adversarial"],
        "tool_relevance": ["near_neighbor"],
        "tool_sufficiency": ["near_neighbor"],
        "task_routing": ["near_neighbor", "counterfactual"],
        "goal_completion": ["counterfactual", "temporal_state"],
        "claim_evidence": ["contradiction"],
        "recovery": ["recovery"],
        "uncertainty_abstention": ["uncertainty"],
        "state_transition": ["temporal_state", "counterfactual"],
        "context_relevance": ["long_context", "position_bias"],
        "tool_necessity": ["near_neighbor", "counterfactual"],
        "stop_continue": ["stop_continue", "counterfactual", "temporal_state"],
        "progress_stuck": ["progress_stuck", "temporal_state", "recovery"],
        "model_escalation": ["model_escalation", "uncertainty", "recovery"],
        "prompt_injection": ["injection_resistance", "adversarial"],
        "trajectory_control": ["trajectory_control", "temporal_state", "counterfactual"],
        "user_intent_alignment": ["intent_alignment", "near_neighbor"],
        "instruction_constraints": ["near_neighbor"],
    }
    filled = 0
    for cap, need in sorted(gaps.items()):
        gnames = capability_generators.get(cap)
        if not gnames:
            continue
        # Cycle the routed generators until the gap is filled.
        i = 0
        added_for_cap = 0
        while added_for_cap < need:
            gname = gnames[i % len(gnames)]
            gen = GENERATOR_REGISTRY[gname]()
            batch = list(
                gen.generate(
                    seed=SEED + 7919 + i * 31 + added_for_cap, n=min(need - added_for_cap, 2000)
                )
            )
            if not batch:
                break
            for c in batch:
                if added_for_cap >= need:
                    break
                # Retarget capability to the gap family (construction rules
                # already determine the correct gold for that decision type).
                c2 = c.model_copy(update={"capability": cap})
                c2 = assign_difficulty(c2, rng)
                base_cases.append(c2)
                syn_parents.append(c2)
                composition[cap] += 1
                source_mix["synthetic"] += 1
                added_for_cap += 1
            i += 1
        filled += added_for_cap
    print(f"gap-fill: {filled} cases")

    # ------------------------------------------------------------------
    # 5. Metamorphic variants (never primary; spec 15, 14)
    # ------------------------------------------------------------------
    variants: list[CanonicalCase] = []
    mp = MinimalPairGenerator()
    variants.extend(
        mp.generate_pair_variants(SEED, [c for c in syn_parents if c.capability == "tool_risk"])
    )
    from jev_agent_eval.metamorphic import build_variants as build_mm_variants

    metamorphic_pool = [c for c in base_cases if not c.is_variant][:2000]
    mm = build_mm_variants(metamorphic_pool, seed=SEED)
    for v in mm:
        c = v.case
        # Mark as variant with relation per spec 15 invariance table.
        c = c.model_copy(
            update={
                "is_variant": True,
                "relation": "invariant" if v.expect_same else "flip_expected",
            }
        )
        variants.append(c)
    print(f"variants: {len(variants)} (diagnostic only)")

    # ------------------------------------------------------------------
    # 6. Assign dev splits (spec 7.3) and build holdout
    # ------------------------------------------------------------------
    for c in base_cases:
        if c.dev_split is None:
            c.dev_split = CaseSplit.RELEASE_PUBLIC

    # Holdout: stratified 3000+ from the public release pool (spec 7.1).
    holdout: list[CanonicalCase] = []
    by_cap: dict[str, list[CanonicalCase]] = {}
    for c in base_cases:
        if c.dev_split in (CaseSplit.RELEASE_PUBLIC, None) and c.source != "legacy_v1":
            by_cap.setdefault(c.capability, []).append(c)
    per_cap_holdout = max(1, args.holdout_target // max(1, len(by_cap)))
    for _cap, members in sorted(by_cap.items()):
        rng.shuffle(members)
        take = min(per_cap_holdout, max(0, len(members) - 50))  # keep >=50 per cap public
        for c in members[:take]:
            c.dev_split = CaseSplit.PRIVATE_HOLDOUT
            holdout.append(c)
    # Top up to the 3000+ target from the largest remaining pools.
    if len(holdout) < args.holdout_target:
        deficit = args.holdout_target - len(holdout)
        pool = [
            c
            for cap, members in by_cap.items()
            for c in members
            if c.dev_split == CaseSplit.RELEASE_PUBLIC
        ]
        rng.shuffle(pool)
        for c in pool[:deficit]:
            c.dev_split = CaseSplit.PRIVATE_HOLDOUT
            holdout.append(c)
    print(f"holdout: {len(holdout)} cases")

    # ------------------------------------------------------------------
    # 7. Write artifacts + manifest
    # ------------------------------------------------------------------
    dump_jsonl(base_cases, out_dir / "cases.jsonl")
    dump_jsonl(variants, out_dir / "variants.jsonl")
    dump_jsonl(holdout, holdout_dir / "holdout.jsonl")

    release_cases = [c for c in base_cases if c.dev_split != CaseSplit.PRIVATE_HOLDOUT]
    composition_final = Counter(c.capability for c in release_cases)
    source_final = Counter(c.source for c in release_cases)
    difficulty_final = Counter(c.difficulty or "unassigned" for c in release_cases)

    manifest = {
        "dataset": "release-v2",
        "dataset_version": "2.0.0",
        "seed": SEED,
        "total_base_cases": len(release_cases),
        "holdout_cases": len(holdout),
        "variant_cases": len(variants),
        "capability_counts": dict(sorted(composition_final.items())),
        "source_mix": dict(sorted(source_final.items())),
        "difficulty_mix": dict(sorted(difficulty_final.items())),
        "case_store_checksum": sha256_file(out_dir / "cases.jsonl"),
        "variants_checksum": sha256_file(out_dir / "variants.jsonl"),
        "holdout_checksum": sha256_file(holdout_dir / "holdout.jsonl"),
        "public_source_revisions": _public_revisions(public_dir),
        "question_pack_version": "2.0",
    }
    (out_dir / "manifest.yaml").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",  # yaml-compatible JSON
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "public_source_revisions"}, indent=2)
    )
    return 0


def _public_revisions(public_dir: Path) -> dict:
    revs = {}
    for prov in sorted(public_dir.glob("*/PROVENANCE.json")):
        d = json.loads(prov.read_text())
        name = prov.parent.name
        revs[name] = {
            k: d.get(k)
            for k in (
                "upstream_revision",
                "hf_revision",
                "hf_revisions",
                "downloaded_at",
                "license",
            )
            if d.get(k) is not None
        }
    return revs


if __name__ == "__main__":
    raise SystemExit(main())
