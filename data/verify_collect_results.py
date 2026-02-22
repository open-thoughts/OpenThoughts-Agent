#!/usr/bin/env python3
"""Collect results from parallel harbor verification jobs and report discrepancies."""

import json
from datetime import datetime
from pathlib import Path

DCAGENT_DIR = Path("/scratch/10000/eguha3/dc-agent")
SAMPLE_DIR = DCAGENT_DIR / "data" / "verify_samples"
RESULTS_JSON = DCAGENT_DIR / "data" / "hf_dataset_verification.json"
RESULTS_MD = DCAGENT_DIR / "data" / "hf_dataset_verification.md"
MODEL = "openai/gpt-5-nano-2025-08-07"

# Old pass rates for comparison
OLD_RATES = {
    "DCAgent/exp_rpt_stack-bash": 4.0,
    "DCAgent/exp_rpt_stack-bash-withtests": 4.0,
    "DCAgent/exp_rpt_stack-cpp": 36.0,
    "DCAgent/exp_rpt_stack-csharp": 12.0,
    "DCAgent/exp_rpt_stack-dockerfile": 0.0,
    "DCAgent/exp_rpt_stack-go": 0.0,
    "DCAgent/exp_rpt_stack-jest": 0.0,
    "DCAgent/exp_rpt_stack-php": 0.0,
    "DCAgent/exp_rpt_stack-pytest": 0.0,
    "DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano": 0.0,
    "DCAgent/exp_rpt_stack-pytest-withtests": 0.0,
    "DCAgent/exp_rpt_stack-ruby": 0.0,
    "DCAgent/exp_rpt_stack-rust": 0.0,
    "DCAgent/exp_rpt_stack-selfdoc": 0.0,
    "DCAgent/exp_rpt_bigcodebench": 60.0,
    "DCAgent/exp_rpt_bugsinpy": 40.0,
    "DCAgent/exp_rpt_bugsinpy-mf": 52.0,
    "DCAgent/exp_rpt_bugswarm": 72.0,
    "DCAgent/exp_rpt_codeelo": 84.0,
    "DCAgent/exp_rpt_codenet-python": 92.0,
    "DCAgent/exp_rpt_codereval-python": 52.0,
    "DCAgent/exp_rpt_crosscodeeval-python": 60.0,
    "DCAgent/exp_rpt_defects4j": 48.0,
    "DCAgent/exp_rpt_e2egit": 56.0,
    "DCAgent/exp_rpt_exercism-python": 80.0,
    "DCAgent/exp_rpt_ghactions": 84.0,
    "DCAgent/exp_rpt_manybugs": 60.0,
    "DCAgent/exp_rpt_methods2test": 92.0,
    "DCAgent/exp_rpt_pymethods2test": 76.0,
    "DCAgent/exp_rpt_quixbugs-python-10k": 84.0,
    "DCAgent/exp_rpt_softwareheritage": 72.0,
    "DCAgent/exp_rpt_swebench": 64.0,
    "DCAgent/exp_rpt_taco": 72.0,
    "DCAgent/exp_rpt_unitsyn-python": 80.0,
}


def parse_result_json(result_file: Path) -> tuple[float | None, int, int]:
    """Parse pass rate from harbor result.json."""
    try:
        with open(result_file) as f:
            data = json.load(f)
    except Exception:
        return None, 0, 0

    if "stats" in data:
        stats = data["stats"]
        evals = stats.get("evals", {})
        if evals:
            eval_key = list(evals.keys())[0]
            metrics = evals[eval_key].get("metrics", [])
            if metrics:
                mean = metrics[0].get("mean", 0)
                n = metrics[0].get("n", 0)
                return mean * 100, int(round(mean * n)), n

    if "results" in data:
        results = data["results"]
        total = len(results)
        passed = sum(1 for r in results if r.get("reward", 0) > 0)
        return (passed / total * 100) if total > 0 else 0, passed, total

    return None, 0, 0


def main():
    # Load manifest
    manifest_file = SAMPLE_DIR / "manifest.json"
    if not manifest_file.exists():
        print("ERROR: No manifest.json found. Run verify_all_hf_parallel.sh first.")
        return

    with open(manifest_file) as f:
        manifest = json.load(f)

    results = []
    for repo_id, info in manifest.items():
        safe = repo_id.replace("/", "_").replace("-", "_")
        job_name = f"verify_{safe}"
        old_rate = OLD_RATES.get(repo_id)

        result = {
            "repo_id": repo_id,
            "status": info["status"],
            "total_tasks": info.get("total", 0),
            "sampled_tasks": info.get("sampled", 0),
            "pass_rate": None,
            "old_pass_rate": old_rate,
            "passed": 0,
            "tested": 0,
            "discrepancy": False,
            "error": None if info["status"] == "OK" else info["status"],
            "timestamp": datetime.now().isoformat(),
        }

        if info["status"] != "OK":
            results.append(result)
            continue

        # Find result.json
        result_file = None
        jobs_dir = DCAGENT_DIR / "jobs"
        for d in sorted(jobs_dir.glob(f"{job_name}*"), reverse=True):
            rf = d / "result.json"
            if rf.exists():
                result_file = rf
                break

        if result_file is None:
            result["status"] = "HARBOR_FAILED"
            result["error"] = "No result.json found"
            results.append(result)
            continue

        pass_rate, passed, total = parse_result_json(result_file)
        result["pass_rate"] = pass_rate
        result["passed"] = passed
        result["tested"] = total
        result["status"] = "VERIFIED"

        # Check discrepancy
        if old_rate is not None and pass_rate is not None:
            if abs(pass_rate - old_rate) > 20:
                result["discrepancy"] = True

        results.append(result)

    # Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Save markdown
    lines = [
        "# HuggingFace Dataset Verification Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {MODEL}",
        f"Total datasets: {len(results)}",
        "",
        "## Summary",
        "",
        f"- Verified: {sum(1 for r in results if r['status'] == 'VERIFIED')}",
        f"- Missing/Error: {sum(1 for r in results if r['status'] not in ('VERIFIED', 'OK'))}",
        f"- Discrepancies (>20% diff): {sum(1 for r in results if r.get('discrepancy'))}",
        "",
        "## Results Table",
        "",
        "| Dataset | Status | Total | New Rate | Old Rate | Diff | Flag |",
        "|---------|--------|-------|----------|----------|------|------|",
    ]

    for r in sorted(results, key=lambda x: x["repo_id"]):
        repo = r["repo_id"]
        status = r["status"]
        total = r["total_tasks"]
        new_rate = f"{r['pass_rate']:.1f}%" if r["pass_rate"] is not None else "N/A"
        old_rate_str = f"{r['old_pass_rate']:.1f}%" if r.get("old_pass_rate") is not None else "N/A"
        if r["pass_rate"] is not None and r.get("old_pass_rate") is not None:
            diff = f"{r['pass_rate'] - r['old_pass_rate']:+.1f}%"
        else:
            diff = "-"
        flag = ""
        if r.get("discrepancy"):
            flag = "DISCREPANCY"
        elif status not in ("VERIFIED",):
            flag = status
        lines.append(f"| {repo} | {status} | {total} | {new_rate} | {old_rate_str} | {diff} | {flag} |")

    # Problem section
    problems = [r for r in results if r["status"] != "VERIFIED" or r.get("discrepancy")]
    if problems:
        lines.extend(["", "## Action Required", ""])
        for r in problems:
            if r.get("discrepancy"):
                lines.append(f"- **{r['repo_id']}**: Pass rate {r['old_pass_rate']:.1f}% -> {r['pass_rate']:.1f}% — NEEDS REGENERATION")
            elif r["status"].startswith("ERROR"):
                lines.append(f"- **{r['repo_id']}**: {r['status']} — NEEDS RE-UPLOAD")
            else:
                lines.append(f"- **{r['repo_id']}**: {r['status']}")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    verified = [r for r in results if r["status"] == "VERIFIED"]
    if verified:
        rates = [r["pass_rate"] for r in verified if r["pass_rate"] is not None]
        print(f"Verified: {len(verified)}/{len(results)}")
        if rates:
            print(f"Pass rates: min={min(rates):.1f}%, max={max(rates):.1f}%, avg={sum(rates)/len(rates):.1f}%")

    disc = [r for r in results if r.get("discrepancy")]
    if disc:
        print(f"\nDISCREPANCIES ({len(disc)}) — need regeneration:")
        for r in disc:
            print(f"  {r['repo_id']}: {r['old_pass_rate']:.1f}% -> {r['pass_rate']:.1f}%")

    errors = [r for r in results if r["status"] not in ("VERIFIED", "OK")]
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for r in errors:
            print(f"  {r['repo_id']}: {r['status']}")

    print(f"\nResults: {RESULTS_JSON}")
    print(f"Summary: {RESULTS_MD}")


if __name__ == "__main__":
    main()
