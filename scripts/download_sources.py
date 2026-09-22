#!/usr/bin/env python
"""Download and pin all public benchmark sources (v2 spec sections 5, 68, 89).

Fetches real data for every source in the v2 source list, records the exact
upstream revision, license, download date, and sha256 checksum into
datasets/public/<name>/PROVENANCE.json, and never executes downloaded content.

GitHub-hosted sources are shallow-cloned (pinned to the cloned HEAD commit),
then curated data files are extracted; HF-hosted datasets are fetched by file.
Downloaded trees are stored as data only (the .git directory is removed).

Security: data files only; no code execution; bounded sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "datasets" / "public"

MAX_BYTES = 600 * 1024 * 1024  # 600 MiB per file safety cap
MAX_MEMBER_BYTES = 50 * 1024 * 1024  # 50 MiB per extracted clone file


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fetch(url: str, max_bytes: int = MAX_BYTES) -> tuple[bytes, str]:
    """Fetch a URL with a hard size cap. Returns (body, sha256)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "jev-agent-eval/2.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"response exceeds size cap: {url}")
    return data, sha256_bytes(data)


def gh_api_hf(dataset_id: str) -> dict:
    import urllib.request

    req = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{dataset_id}",
        headers={"User-Agent": "jev-agent-eval/2.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def clone(repo: str, name: str) -> tuple[Path, str]:
    """Shallow-clone repo to a temp dir; returns (tree_path, HEAD commit sha)."""
    target = PUBLIC / "_clones" / name
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(target)],
        check=True,
        capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return target, head


def cleanup_clone(name: str) -> None:
    target = PUBLIC / "_clones" / name
    if target.exists():
        shutil.rmtree(target)


def tree_files(
    root: Path,
    subdir: str | None = None,
    suffixes: tuple[str, ...] = (".json", ".jsonl"),
    exclude: tuple[str, ...] = (),
) -> dict[str, bytes]:
    """Collect data files under root (optionally a subdir), keyed by relative path."""
    base = root / subdir if subdir else root
    files: dict[str, bytes] = {}
    if not base.exists():
        return files
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if ".git" in p.parts:
            continue
        if any(part in exclude for part in p.parts):
            continue
        if p.suffix not in suffixes:
            continue
        if p.stat().st_size > MAX_MEMBER_BYTES:
            continue
        if len(rel) > 240:
            continue
        files[rel] = p.read_bytes()
    return files


def save(name: str, files: dict[str, bytes], provenance: dict) -> None:
    """Write files + provenance under datasets/public/<name>/."""
    target = PUBLIC / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for rel, body in files.items():
        p = target / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        checksums[rel] = sha256_bytes(body)
    provenance = {
        **provenance,
        "downloaded_at": _now(),
        "checksums": checksums,
    }
    (target / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[{name}] saved {len(files)} files, {sum(len(b) for b in files.values())} bytes")


# --------------------------------------------------------------------------
# Hugging Face-hosted datasets
# --------------------------------------------------------------------------


def dl_swebench(name: str = "swebench") -> None:
    """SWE-bench full (2,294) + Verified (500) official parquet files."""
    out: dict[str, bytes] = {}
    datasets = {
        "swebench_full": "princeton-nlp/SWE-bench",
        "swebench_verified": "princeton-nlp/SWE-bench_Verified",
    }
    revisions = {}
    for stem, ds in datasets.items():
        hf = gh_api_hf(ds)
        revisions[ds] = hf["sha"]
        for s in hf["siblings"]:
            fname = s["rfilename"]
            if fname.endswith(".parquet"):
                url = f"https://huggingface.co/datasets/{ds}/resolve/main/{fname}"
                body, _ = fetch(url)
                out[f"{stem}_{fname.split('/')[-1]}"] = body
    save(
        name,
        out,
        {
            "source": "SWE-bench",
            "upstream_repo": "https://github.com/SWE-bench/SWE-bench",
            "upstream_url": "https://www.swebench.com/",
            "hf_revisions": revisions,
            "license": "MIT (dataset); see upstream for per-instance licenses",
            "expected_counts": {"full": 2294, "verified": 500},
        },
    )


def dl_swebench_live(name: str = "swebench_live") -> None:
    """SWE-bench-Live: continuously updated monthly parquet shards."""
    hf = gh_api_hf("SWE-bench-Live/SWE-bench-Live")
    files: dict[str, bytes] = {}
    for s in hf["siblings"]:
        fname = s["rfilename"]
        if fname.endswith(".parquet"):
            url = f"https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live/resolve/main/{fname}"
            body, _ = fetch(url)
            files[fname.replace("data/", "")] = body
    save(
        name,
        files,
        {
            "source": "SWE-bench-Live",
            "upstream_repo": "https://github.com/microsoft/SWE-bench-Live",
            "upstream_url": "https://swe-bench-live.github.io/",
            "hf_revision": hf["sha"],
            "license": "MIT",
        },
    )


def dl_ifeval(name: str = "ifeval") -> None:
    """Google IFEval: 541 prompts with programmatic verifiers."""
    hf = gh_api_hf("google/IFEval")
    files: dict[str, bytes] = {}
    for s in hf["siblings"]:
        if s["rfilename"].endswith((".jsonl", ".json")):
            url = f"https://huggingface.co/datasets/google/IFEval/resolve/main/{s['rfilename']}"
            body, _ = fetch(url)
            files[s["rfilename"]] = body
    save(
        name,
        files,
        {
            "source": "Google IFEval",
            "upstream_repo": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
            "hf_revision": hf["sha"],
            "license": "Apache-2.0",
            "expected_counts": {"prompts": 541},
        },
    )


# --------------------------------------------------------------------------
# GitHub-hosted sources (shallow clone + curated extraction)
# --------------------------------------------------------------------------


def dl_bfcl(name: str = "bfcl") -> None:
    """BFCL V4 agentic/multi-turn test data from the official repo."""
    tmp, commit = clone("EnlightenedAI/BFCL", "bfcl")
    data_root = tmp / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    files: dict[str, bytes] = {}
    # V4 test categories (agentic, multi-turn, live, memory, web search) and
    # their possible answers + multi-turn function doc definitions.
    for p in sorted(data_root.rglob("*")):
        if not p.is_file() or p.suffix != ".json":
            continue
        rel = str(p.relative_to(data_root))
        if not (
            rel.startswith("BFCL_v4_")
            or rel.startswith("possible_answer/BFCL_v4_")
            or rel.startswith("multi_turn_func_doc/")
        ):
            continue
        files[rel] = p.read_bytes()
    cleanup_clone("bfcl")
    save(
        name,
        files,
        {
            "source": "BFCL V4",
            "upstream_repo": "https://github.com/EnlightenedAI/BFCL",
            "upstream_revision": commit,
            "license": "Apache-2.0",
            "expected_counts": {"agentic_v4": 665},
        },
    )


def dl_ifbench(name: str = "ifbench") -> None:
    """AllenAI IFBench: test jsonl with 25 classic + 58 OOD constraint verifiers."""
    tmp, commit = clone("allenai/IFBench", "ifbench")
    body = (tmp / "data" / "IFBench_test.jsonl").read_bytes()
    cleanup_clone("ifbench")
    save(
        name,
        {"IFBench_test.jsonl": body},
        {
            "source": "AllenAI IFBench",
            "upstream_repo": "https://github.com/allenai/IFBench",
            "upstream_revision": commit,
            "license": "Apache-2.0",
            "expected_counts": {"constraints": "25 classic + 58 OOD keys"},
        },
    )


def dl_terminal_bench(name: str = "terminal_bench") -> None:
    """Terminal-Bench 2.0: 89 tasks with task.toml + instruction + solution + tests."""
    tmp, commit = clone("harbor-framework/terminal-bench-2", "terminal_bench")
    files: dict[str, bytes] = {}
    index = []
    for p in sorted(tmp.iterdir()):
        if not p.is_dir() or p.name.startswith((".", "_")):
            continue
        for rel in (
            "task.toml",
            "instruction.md",
            "solution/solve.sh",
            "solution/solve.py",
            "tests/test.sh",
            "tests/test_outputs.py",
        ):
            fp = p / rel
            if fp.is_file() and fp.stat().st_size <= MAX_MEMBER_BYTES:
                files[f"{p.name}/{rel}"] = fp.read_bytes()
                index.append({"task": p.name, "file": rel})
    files["INDEX.json"] = json.dumps(index).encode()
    cleanup_clone("terminal_bench")
    save(
        name,
        files,
        {
            "source": "Terminal-Bench 2.0",
            "upstream_repo": "https://github.com/harbor-framework/terminal-bench-2",
            "upstream_url": "https://www.tbench.ai/",
            "upstream_revision": commit,
            "license": "Apache-2.0",
            "expected_counts": {"tasks": 89},
        },
    )


def dl_tuabench(name: str = "tuabench") -> None:
    """TUA-Bench: 120 execution-based terminal-use tasks."""
    tmp, commit = clone("facebookresearch/TUA-Bench", "tuabench")
    files = tree_files(
        tmp,
        subdir="tasks",
        suffixes=(".md", ".yaml", ".yml", ".toml", ".txt", ".json"),
    )
    cleanup_clone("tuabench")
    save(
        name,
        files,
        {
            "source": "TUA-Bench",
            "upstream_repo": "https://github.com/facebookresearch/TUA-Bench",
            "upstream_revision": commit,
            "license": "See upstream LICENSE (NOASSERTION)",
            "expected_counts": {"tasks": 120},
        },
    )


def dl_tau2(name: str = "tau2") -> None:
    """tau2-bench: domain tasks/policies/evaluators under data/."""
    tmp, commit = clone("sierra-research/tau2-bench", "tau2")
    files = tree_files(tmp, subdir="data", suffixes=(".json", ".yaml", ".yml", ".md"))
    cleanup_clone("tau2")
    save(
        name,
        files,
        {
            "source": "tau2-bench",
            "upstream_repo": "https://github.com/sierra-research/tau2-bench",
            "upstream_revision": commit,
            "license": "MIT",
        },
    )


def dl_agentbench(name: str = "agentbench") -> None:
    """AgentBench FC: task definitions + data from the repo."""
    tmp, commit = clone("THUDM/AgentBench", "agentbench")
    files = tree_files(tmp, subdir="agent_eval", suffixes=(".json", ".jsonl", ".yaml"))
    files.update(tree_files(tmp, subdir="data", suffixes=(".json", ".jsonl", ".yaml")))
    readme = tmp / "README.md"
    if readme.is_file():
        files["README.md"] = readme.read_bytes()
    cleanup_clone("agentbench")
    save(
        name,
        files,
        {
            "source": "AgentBench / AgentBench FC",
            "upstream_repo": "https://github.com/THUDM/AgentBench",
            "upstream_revision": commit,
            "license": "MIT (code); data licenses per environment",
        },
    )


def dl_osworld(name: str = "osworld") -> None:
    """OSWorld: 369 real computer tasks (task definitions only)."""
    tmp, commit = clone("xlang-ai/OSWorld", "osworld")
    files = tree_files(tmp, subdir="evaluation_examples", suffixes=(".json", ".jsonl"))
    readme = tmp / "README.md"
    if readme.is_file():
        files["README.md"] = readme.read_bytes()
    cleanup_clone("osworld")
    save(
        name,
        files,
        {
            "source": "OSWorld",
            "upstream_repo": "https://github.com/xlang-ai/OSWorld",
            "upstream_url": "https://os-world.github.io/",
            "upstream_revision": commit,
            "license": "Apache-2.0 (task definitions)",
            "expected_counts": {"tasks": 369},
        },
    )


def dl_webarena_verified(name: str = "webarena_verified") -> None:
    """WebArena-Verified: manually verified versioned tasks."""
    tmp, commit = clone("ServiceNow/webarena-verified", "webarena_verified")
    files = tree_files(tmp, suffixes=(".json", ".jsonl", ".yaml", ".md"))
    cleanup_clone("webarena_verified")
    save(
        name,
        files,
        {
            "source": "WebArena-Verified",
            "upstream_repo": "https://github.com/ServiceNow/webarena-verified",
            "upstream_revision": commit,
            "license": "Apache-2.0",
        },
    )


def dl_workarena(name: str = "workarena") -> None:
    """WorkArena-L1/++: task template definitions (source reservoir)."""
    tmp, commit = clone("ServiceNow/WorkArena", "workarena")
    files = tree_files(
        tmp,
        suffixes=(".py", ".md"),
    )
    # Keep only task-definition modules to bound size.
    files = {k: v for k, v in files.items() if "task" in k.lower() or k == "README.md"}
    cleanup_clone("workarena")
    save(
        name,
        files,
        {
            "source": "WorkArena / WorkArena++",
            "upstream_repo": "https://github.com/ServiceNow/WorkArena",
            "upstream_revision": commit,
            "license": "MIT (code); ServiceNow data terms upstream",
            "expected_counts": {
                "workarena_l1": "19,912 instances over 33 atomic tasks",
                "workarena_pp": 682,
            },
        },
    )


def dl_online_mind2web(name: str = "online_mind2web") -> None:
    """Online-Mind2Web: task definitions (live-web; separately versioned track)."""
    tmp, commit = clone("OSU-NLP-Group/Online-Mind2Web", "online_mind2web")
    # Bound extraction: keep small json/yaml/md task-definition files only.
    files = tree_files(tmp, suffixes=(".json", ".jsonl", ".yaml", ".md"))
    files = {k: v for k, v in files.items() if len(v) <= 5 * 1024 * 1024}
    cleanup_clone("online_mind2web")
    save(
        name,
        files,
        {
            "source": "Online-Mind2Web",
            "upstream_repo": "https://github.com/OSU-NLP-Group/Online-Mind2Web",
            "upstream_revision": commit,
            "license": "MIT",
            "note": "live-web source; kept in separately versioned live track (spec 5.12)",
        },
    )


DOWNLOADERS = {
    "swebench": dl_swebench,
    "swebench_live": dl_swebench_live,
    "bfcl": dl_bfcl,
    "terminal_bench": dl_terminal_bench,
    "tuabench": dl_tuabench,
    "tau2": dl_tau2,
    "agentbench": dl_agentbench,
    "ifeval": dl_ifeval,
    "ifbench": dl_ifbench,
    "osworld": dl_osworld,
    "webarena_verified": dl_webarena_verified,
    "workarena": dl_workarena,
    "online_mind2web": dl_online_mind2web,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="*", default=None, help="Subset to download (default: all)"
    )
    parser.add_argument("--list", action="store_true", help="List sources without downloading")
    args = parser.parse_args()

    selected = args.sources or list(DOWNLOADERS)
    for s in selected:
        if s not in DOWNLOADERS:
            print(f"unknown source: {s}", file=sys.stderr)
            return 2

    if args.list:
        for s in selected:
            print(f"{s}")
        return 0

    failures: list[str] = []
    for s in selected:
        t0 = time.time()
        try:
            DOWNLOADERS[s]()
            print(f"[{s}] ok in {time.time() - t0:.1f}s", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(s)
            print(f"[{s}] FAILED: {e}", file=sys.stderr)
    if failures:
        print(f"failed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
