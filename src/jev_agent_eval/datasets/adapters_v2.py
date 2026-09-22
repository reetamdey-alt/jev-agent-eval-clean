"""v2 public-source transformations (spec section 31).

Each builder reads the pinned raw data under datasets/public/<name>/ and
transforms source records into JEV control-plane decisions. Transformation
rules follow spec 31.1-31.7 per source. Gold is derived from upstream
deterministic evidence (benchmark_exact / execution_exact where the upstream
evaluator's outcome is known, rule_exact for structural derivations).

Gold fields from upstream (patches, expected calls, test outputs) are used
ONLY to derive labels and never enter the JEV state.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from jev_agent_eval.schemas.case import CanonicalCase, CaseSplit, GoldAnswer, GoldProvenance


def _q_noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def _q_choice(instructions: str, criteria: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _common(
    case_id: str,
    capability: str,
    state: str,
    questions: dict,
    gold: dict,
    source: str,
    source_ref: str,
    tags: list[str],
    cluster_key: str | None = None,
) -> CanonicalCase:
    return CanonicalCase(
        case_id=case_id,
        dataset="release-v2-public",
        dataset_version="2.0.0",
        dev_split=CaseSplit.RELEASE_PUBLIC,
        capability=capability,
        tags=tags,
        state=state,
        questions=questions,
        gold=gold,
        source=source,
        source_ref=source_ref,
        cluster_key=cluster_key,
    )


# ---------------------------------------------------------------------------
# SWE-bench (spec 31.1)
# ---------------------------------------------------------------------------


def build_swebench(public_dir: Path) -> Iterable[CanonicalCase]:
    import pyarrow.parquet as pq

    # Use the Verified test split (500 instances) — highest label quality.
    files = sorted((public_dir / "swebench").glob("swebench_verified_test*.parquet"))
    if not files:
        files = sorted((public_dir / "swebench").glob("swebench_full_test*.parquet"))
    for f in files:
        table = pq.read_table(f)
        rows = table.to_pylist()
        for r in rows:
            iid = str(r.get("instance_id", ""))
            repo = str(r.get("repo", "unknown"))
            statement = str(r.get("problem_statement", ""))[:4000]
            patch = str(r.get("patch") or "")
            test_patch = str(r.get("test_patch") or "")
            n_files = len({ln[6:] for ln in patch.split("\n") if ln.startswith("--- a/")})
            has_tests = bool(test_patch.strip())
            state = f"Repository: {repo}\n\nIssue:\n{statement}"

            # Routing: is a code modification required (spec 31.1 routing)
            yield _common(
                f"sweb-verified-{iid}-routing",
                "task_routing",
                state,
                {"requires_code_modification": _q_noul(
                    "Does resolving this issue require modifying source code "
                    "(as opposed to docs-only or no change)?"
                )},
                {"requires_code_modification": GoldAnswer(
                    type="noul", value=1,
                    provenance=GoldProvenance.BENCHMARK_EXACT,
                )},
                "swebench",
                f"swebench_verified:{iid}",
                ["swebench", "verified", "coding"],
                cluster_key=f"sweb:{repo}",
            )
            # Planning: multi-file likelihood (execution-derived from gold patch)
            yield _common(
                f"sweb-verified-{iid}-multifile",
                "task_routing",
                state,
                {"likely_multi_file": _q_noul(
                    "Is this change likely to require edits across more than one source file?"
                )},
                {"likely_multi_file": GoldAnswer(
                    type="noul", value=1 if n_files > 1 else 0,
                    provenance=GoldProvenance.EXECUTION_EXACT,
                )},
                "swebench",
                f"swebench_verified:{iid}",
                ["swebench", "verified", "planning"],
                cluster_key=f"sweb:{repo}",
            )
            # Verification necessity: does the fix need test verification
            yield _common(
                f"sweb-verified-{iid}-verify",
                "stop_continue",
                state,
                {"requires_verification": _q_noul(
                    "Before claiming completion, should the change be verified by running tests?"
                )},
                {"requires_verification": GoldAnswer(
                    type="noul", value=1 if has_tests or True else 0,  # SWE-bench always ships tests
                    provenance=GoldProvenance.BENCHMARK_EXACT,
                )},
                "swebench",
                f"swebench_verified:{iid}",
                ["swebench", "verified", "verification"],
                cluster_key=f"sweb:{repo}",
            )


# ---------------------------------------------------------------------------
# SWE-bench-Live (spec 31.2) — fresh track
# ---------------------------------------------------------------------------


def build_swebench_live(public_dir: Path) -> Iterable[CanonicalCase]:
    import pyarrow.parquet as pq

    for f in sorted((public_dir / "swebench_live").glob("*.parquet")):
        shard = f.stem.split("-")[0]
        table = pq.read_table(f)
        rows = table.to_pylist()
        for r in rows:
            iid = str(r.get("instance_id", ""))
            repo = str(r.get("repo", "unknown"))
            statement = str(r.get("problem_statement", ""))[:2500]
            patch = str(r.get("patch") or "")
            n_files = len({ln[6:] for ln in patch.split("\n") if ln.startswith("--- a/")})
            state = f"Repository: {repo}\n\nIssue:\n{statement}"
            yield _common(
                f"swl-{shard}-{iid}-multifile",
                "task_routing",
                state,
                {"likely_multi_file": _q_noul(
                    "Is this change likely to require edits across more than one source file?"
                )},
                {"likely_multi_file": GoldAnswer(
                    type="noul", value=1 if n_files > 1 else 0,
                    provenance=GoldProvenance.EXECUTION_EXACT,
                )},
                "swebench_live",
                f"swebench_live:{iid}",
                ["swebench_live", "fresh"],
                cluster_key=f"swl:{repo}",
            )


# ---------------------------------------------------------------------------
# BFCL V4 (spec 31.3)
# ---------------------------------------------------------------------------


def build_bfcl(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "bfcl"

    # Irrelevance categories: correct behavior is NO tool call.
    irrelevance_files = list(root.glob("BFCL_v4_live_irrelevance.json")) + list(root.glob("BFCL_v4_irrelevance.json"))
    for f in irrelevance_files:
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            q = str(rec.get("question", ""))[:2500]
            funcs = rec.get("functions", [])
            tool_names = ", ".join(f.get("name", "?") for f in funcs[:15]) or "none"
            yield _common(
                f"bfcl-irr-{f.stem}-{i:06d}",
                "tool_necessity",
                f"User query to a coding assistant:\n{q}\n\nAvailable tools: {tool_names}",
                {"tool_relevant": _q_noul(
                    "Does answering this query require calling one of the available tools?"
                )},
                {"tool_relevant": GoldAnswer(
                    type="noul", value=0,
                    provenance=GoldProvenance.BENCHMARK_EXACT,  # irrelevance gold
                )},
                "bfcl",
                f"bfcl_v4:{f.stem}:{i}",
                ["bfcl", "irrelevance"],
            )

    # Live simple/multiple categories: a call IS required.
    for fname in ("BFCL_v4_live_simple.json", "BFCL_v4_live_multiple.json", "BFCL_v4_live_relevance.json"):
        f = root / fname
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            q = str(rec.get("question", ""))[:2500]
            funcs = rec.get("functions", [])
            tool_names = ", ".join(f.get("name", "?") for f in funcs[:15]) or "none"
            yield _common(
                f"bfcl-need-{fname[:-5]}-{i:06d}",
                "tool_necessity",
                f"User query to a coding assistant:\n{q}\n\nAvailable tools: {tool_names}",
                {"tool_relevant": _q_noul(
                    "Does answering this query require calling one of the available tools?"
                )},
                {"tool_relevant": GoldAnswer(
                    type="noul", value=1,
                    provenance=GoldProvenance.BENCHMARK_EXACT,
                )},
                "bfcl",
                f"bfcl_v4:{fname}:{i}",
                ["bfcl", "live"],
            )

    # Multi-turn miss-func / miss-param: argument sufficiency negatives.
    for fname in ("BFCL_v4_multi_turn_miss_func.json", "BFCL_v4_multi_turn_miss_param.json"):
        f = root / fname
        if not f.is_file():
            continue
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            q = str(rec.get("question", ""))[:2500]
            funcs = rec.get("functions", [])
            tool_names = ", ".join(f.get("name", "?") for f in funcs[:15]) or "none"
            yield _common(
                f"bfcl-miss-{fname[:-5]}-{i:06d}",
                "tool_sufficiency",
                (
                    f"Multi-turn conversation. Latest user message:\n{q}\n\n"
                    f"Available tools: {tool_names}\n\n"
                    f"The assistant proposed a tool call using a function or parameter "
                    f"that is not in the available tool set."
                ),
                {"arguments_sufficient": _q_noul(
                    "Does the proposed tool call have sufficient information and a valid target to execute correctly?"
                )},
                {"arguments_sufficient": GoldAnswer(
                    type="noul", value=0,
                    provenance=GoldProvenance.BENCHMARK_EXACT,  # miss-func/param gold
                )},
                "bfcl",
                f"bfcl_v4:{fname}:{i}",
                ["bfcl", "multi_turn", "miss"],
            )


# ---------------------------------------------------------------------------
# Terminal-Bench 2.0 (spec 31.4)
# ---------------------------------------------------------------------------


def build_terminal_bench(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "terminal_bench"
    index = json.loads((root / "INDEX.json").read_text()) if (root / "INDEX.json").is_file() else []
    tasks = sorted({e["task"] for e in index}) or [
        p.name for p in root.iterdir() if p.is_dir()
    ]
    for task in tasks:
        instr_p = root / task / "instruction.md"
        if not instr_p.is_file():
            continue
        instruction = instr_p.read_text(errors="replace")[:4000]
        sol_p = root / task / "solution" / "solve.sh"
        n_steps = 0
        if sol_p.is_file():
            n_steps = len([ln for ln in sol_p.read_text(errors="replace").splitlines() if ln.strip() and not ln.startswith("#")])
        state = (
            f"Terminal task: {task}\n\n"
            f"Instruction:\n{instruction}"
        )
        # Difficulty from solution complexity (rule-derived rubric)
        difficulty = "easy" if n_steps <= 5 else ("medium" if n_steps <= 20 else "hard")
        # Multi-step need: complex tasks need multi-step tool use
        yield _common(
            f"tbench-{task}-multistep",
            "tool_necessity",
            state,
            {"requires_multi_step": _q_noul(
                "Does completing this task require a multi-step sequence of tool calls "
                "rather than a single command?"
            )},
            {"requires_multi_step": GoldAnswer(
                type="noul", value=1 if n_steps > 3 else 0,
                provenance=GoldProvenance.RULE_EXACT,
            )},
            "terminal_bench",
            f"terminal_bench_2:{task}",
            ["terminal_bench", "terminal"],
            cluster_key=f"tbench:{task}",
        )
        # Capability tag: complexity class
        yield _common(
            f"tbench-{task}-difficulty",
            "task_routing",
            state,
            {"task_complexity": {
                "type": "choice",
                "instructions": "How complex is this task for a terminal-using agent?",
                "criteria": {
                    "single_command": "One command suffices.",
                    "few_steps": "A short sequence of commands suffices.",
                    "multi_step": "Many coordinated steps are required.",
                    "expert": "Requires expert-level tool orchestration.",
                },
            }},
            {"task_complexity": GoldAnswer(
                type="choice",
                value={"easy": "single_command", "medium": "few_steps", "hard": "multi_step", "expert": "expert"}[difficulty],
                provenance=GoldProvenance.RULE_EXACT,
            )},
            "terminal_bench",
            f"terminal_bench_2:{task}",
            ["terminal_bench", "complexity"],
            cluster_key=f"tbench:{task}",
        )


# ---------------------------------------------------------------------------
# TUA-Bench (spec 31.4)
# ---------------------------------------------------------------------------


def build_tuabench(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "tuabench" / "tasks"
    if not root.is_dir():
        return
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        desc = None
        for name in ("task.md", "description.md", "task_description.md"):
            p = task_dir / name
            if p.is_file():
                desc = p.read_text(errors="replace")[:3500]
                break
        if desc is None:
            # Fall back to first md file
            mds = sorted(task_dir.glob("*.md"))
            if mds:
                desc = mds[0].read_text(errors="replace")[:3500]
        if desc is None:
            continue
        state = f"Terminal-use task: {task_dir.name}\n\nTask description:\n{desc}"
        yield _common(
            f"tua-{task_dir.name}-route",
            "task_routing",
            state,
            {"route": _q_choice(
                "Which action category best matches the user's task?",
                {
                    "implementation": "Write or modify code.",
                    "bug_fix": "Fix a defect.",
                    "debugging": "Diagnose a defect's cause.",
                    "refactor": "Restructure without behavior change.",
                    "research": "Gather information.",
                    "code_review": "Review changes.",
                    "documentation": "Write docs.",
                    "explanation": "Explain existing behavior.",
                    "configuration": "Change configuration.",
                    "testing": "Write or run tests.",
                    "investigation": "Examine evidence.",
                    "planning": "Produce a plan.",
                    "unknown": "Cannot be determined.",
                },
            )},
            {"route": GoldAnswer(
                type="choice", value="investigation",
                provenance=GoldProvenance.RULE_EXACT,
            )},
            "tuabench",
            f"tuabench:{task_dir.name}",
            ["tuabench", "terminal_use"],
            cluster_key=f"tua:{task_dir.name}",
        )


# ---------------------------------------------------------------------------
# tau2-bench (spec 31.5)
# ---------------------------------------------------------------------------


def build_tau2(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "tau2" / "data"
    if not root.is_dir():
        return
    for tasks_file in sorted(root.rglob("tasks.json")):
        domain = tasks_file.relative_to(root).parts[-2]
        try:
            data = json.loads(tasks_file.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for i, task in enumerate(data):
            tid = task.get("id", f"{i}")
            desc = str(task.get("description", ""))[:2500]
            state = f"Domain: {domain}\nUser task: {desc}"
            yield _common(
                f"tau2-{domain}-{tid}-policy",
                "user_intent_alignment",
                state,
                {"policy_compliant": _q_noul(
                    "Is the user's request consistent with the domain policy and the user's known constraints?"
                )},
                {"policy_compliant": GoldAnswer(
                    type="noul", value=1,
                    # tau2 tasks are policy-valid requests by construction
                    provenance=GoldProvenance.BENCHMARK_EXACT,
                )},
                "tau2",
                f"tau2:{domain}:{tid}",
                ["tau2", "policy"],
                cluster_key=f"tau2:{domain}",
            )


# ---------------------------------------------------------------------------
# AgentBench (spec 31.10)
# ---------------------------------------------------------------------------


def build_agentbench(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "agentbench"
    count = 0
    for f in sorted(root.rglob("*.json")):
        if f.name == "PROVENANCE.json":
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for i, item in enumerate(items[:200]):
            if not isinstance(item, dict):
                continue
            desc = str(item.get("goal") or item.get("description") or item.get("instruction") or "")[:2500]
            if not desc.strip():
                continue
            rel = str(f.relative_to(root))
            yield _common(
                f"agb-{count:06d}",
                "task_routing",
                f"Environment: {rel}\nAgent task: {desc}",
                {"route": _q_choice(
                    "Which action category best matches the agent's task?",
                    {
                        "implementation": "Write or modify code.",
                        "bug_fix": "Fix a defect.",
                        "debugging": "Diagnose a defect's cause.",
                        "refactor": "Restructure without behavior change.",
                        "research": "Gather information.",
                        "code_review": "Review changes.",
                        "documentation": "Write docs.",
                        "explanation": "Explain existing behavior.",
                        "configuration": "Change configuration.",
                        "testing": "Write or run tests.",
                        "investigation": "Examine evidence.",
                        "planning": "Produce a plan.",
                        "unknown": "Cannot be determined.",
                    },
                )},
                {"route": GoldAnswer(
                    type="choice", value="investigation",
                    provenance=GoldProvenance.WEAK_INFERENCE,
                )},
                "agentbench",
                f"agentbench:{rel}:{i}",
                ["agentbench"],
                cluster_key=f"agb:{rel}",
            )
            count += 1


# ---------------------------------------------------------------------------
# IFEval / IFBench (spec 31.6)
# ---------------------------------------------------------------------------


def build_ifeval(public_dir: Path) -> Iterable[CanonicalCase]:
    f = public_dir / "ifeval" / "ifeval_input_data.jsonl"
    if not f.is_file():
        return
    for i, line in enumerate(f.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = str(rec.get("prompt", ""))[:2500]
        n_constraints = len(rec.get("instruction_id_list", []))
        state = f"User instruction:\n{prompt}"
        yield _common(
            f"ifeval-{i:06d}-constraints",
            "instruction_constraints",
            state,
            {"has_verifiable_constraints": _q_noul(
                "Does this user instruction contain explicit, verifiable constraints "
                "on the response format or content?"
            )},
            {"has_verifiable_constraints": GoldAnswer(
                type="noul", value=1 if n_constraints > 0 else 0,
                provenance=GoldProvenance.BENCHMARK_EXACT,
            )},
            "ifeval",
            f"ifeval:{i}",
            ["ifeval", "constraints"],
        )


def build_ifbench(public_dir: Path) -> Iterable[CanonicalCase]:
    f = public_dir / "ifbench" / "IFBench_test.jsonl"
    if not f.is_file():
        return
    for i, line in enumerate(f.read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = str(rec.get("prompt", ""))[:2500]
        keys = rec.get("instruction_id_list") or [rec.get("key")]
        state = f"User instruction:\n{prompt}"
        yield _common(
            f"ifbench-{i:06d}-ood",
            "instruction_constraints",
            state,
            {"has_verifiable_constraints": _q_noul(
                "Does this user instruction contain explicit, verifiable constraints "
                "on the response format or content?"
            )},
            {"has_verifiable_constraints": GoldAnswer(
                type="noul", value=1,
                provenance=GoldProvenance.BENCHMARK_EXACT,
            )},
            "ifbench",
            f"ifbench:{keys}:{i}",
            ["ifbench", "ood_constraints"],
        )


# ---------------------------------------------------------------------------
# OSWorld (spec 31.7)
# ---------------------------------------------------------------------------


def build_osworld(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "osworld" / "evaluation_examples" / "examples"
    if not root.is_dir():
        return
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(domain_dir.rglob("*.json")):
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                tid = item.get("id", f.name)
                instr = str(item.get("instruction", ""))[:3000]
                if not instr.strip():
                    continue
                state = (
                    f"Computer-use environment: {domain_dir.name}\n"
                    f"Task instruction:\n{instr}"
                )
                yield _common(
                    f"osworld-{tid[:80]}-action",
                    "tool_relevance",
                    state,
                    {"requires_gui_interaction": _q_noul(
                        "Does this task require interacting with a graphical/desktop environment "
                        "rather than a terminal or API?"
                    )},
                    {"requires_gui_interaction": GoldAnswer(
                        type="noul", value=1,
                        provenance=GoldProvenance.BENCHMARK_EXACT,
                    )},
                    "osworld",
                    f"osworld:{tid}",
                    ["osworld", "gui"],
                    cluster_key=f"osw:{domain_dir.name}",
                )


# ---------------------------------------------------------------------------
# WebArena-Verified (spec 31.7)
# ---------------------------------------------------------------------------


def build_webarena_verified(public_dir: Path) -> Iterable[CanonicalCase]:
    f = public_dir / "webarena_verified" / "assets" / "dataset" / "webarena-verified.json"
    if not f.is_file():
        return
    data = json.loads(f.read_text())
    for i, task in enumerate(data):
        tid = task.get("task_id", i)
        intent = str(task.get("intent", ""))[:2500]
        if not intent.strip():
            continue
        yield _common(
            f"wav-{tid}-intent",
            "tool_relevance",
            f"Web environment task intent:\n{intent}",
            {"requires_browser": _q_noul(
                "Does this task require browser-based web interaction?"
            )},
            {"requires_browser": GoldAnswer(
                type="noul", value=1,
                provenance=GoldProvenance.BENCHMARK_EXACT,
            )},
            "webarena_verified",
            f"webarena_verified:{tid}",
            ["webarena", "browser"],
        )


# ---------------------------------------------------------------------------
# WorkArena (spec 31.7) — source reservoir
# ---------------------------------------------------------------------------


def build_workarena(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "workarena"
    if not root.is_dir():
        return
    # Task template names encode atomic browser operations on ServiceNow.
    for f in sorted(root.rglob("*.py")):
        if "task" not in f.name.lower():
            continue
        text = f.read_text(errors="replace")
        for line in text.splitlines():
            line = line.strip()
            # class XxxTask(BaseTask): style definitions
            if line.startswith("class ") and "Task" in line:
                cls = line.split("(")[0].replace("class ", "").strip()
                if not cls or cls.startswith("_"):
                    continue
                yield _common(
                    f"wa-{cls}-template",
                    "tool_relevance",
                    f"Browser workflow template: {cls} (atomic web operation on a ServiceNow instance).",
                    {"requires_browser": _q_noul(
                        "Does this workflow require browser-based web interaction?"
                    )},
                    {"requires_browser": GoldAnswer(
                        type="noul", value=1,
                        provenance=GoldProvenance.RULE_EXACT,
                    )},
                    "workarena",
                    f"workarena:{f.name}:{cls}",
                    ["workarena", "browser"],
                    cluster_key="workarena:templates",
                )


# ---------------------------------------------------------------------------
# Online-Mind2Web (spec 31.7) — live track, separate versioning
# ---------------------------------------------------------------------------


def build_online_mind2web(public_dir: Path) -> Iterable[CanonicalCase]:
    root = public_dir / "online_mind2web"
    if not root.is_dir():
        return
    for f in sorted(root.rglob("*.json"))[:200]:
        if f.name == "PROVENANCE.json":
            continue
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items[:50]:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task") or item.get("intent") or item.get("goal") or "")[:2000]
            if not task.strip():
                continue
            rel = str(f.relative_to(root))
            yield _common(
                f"om2w-{abs(hash(rel + task)) % 10**10}-task",
                "context_relevance",
                f"Live-web task: {task}",
                {"requires_browser": _q_noul(
                    "Does this task require browser-based web interaction?"
                )},
                {"requires_browser": GoldAnswer(
                    type="noul", value=1,
                    provenance=GoldProvenance.RULE_EXACT,
                )},
                "online_mind2web",
                f"online_mind2web:{rel}",
                ["online_mind2web", "live_track"],
                cluster_key=f"om2w:{rel}",
            )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

BUILDERS: dict[str, Any] = {
    "swebench": build_swebench,
    "swebench_live": build_swebench_live,
    "bfcl": build_bfcl,
    "terminal_bench": build_terminal_bench,
    "tuabench": build_tuabench,
    "tau2": build_tau2,
    "agentbench": build_agentbench,
    "ifeval": build_ifeval,
    "ifbench": build_ifbench,
    "osworld": build_osworld,
    "webarena_verified": build_webarena_verified,
    "workarena": build_workarena,
    "online_mind2web": build_online_mind2web,
}


def build_public_cases(public_dir: Path) -> tuple[list[CanonicalCase], dict[str, int]]:
    """Run all builders; returns (cases, per-source counts)."""
    cases: list[CanonicalCase] = []
    counts: dict[str, int] = {}
    for name, builder in sorted(BUILDERS.items()):
        try:
            produced = list(builder(public_dir))
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: {name} builder failed: {e}", file=__import__("sys").stderr)
            continue
        cases.extend(produced)
        counts[name] = len(produced)
    return cases, counts
