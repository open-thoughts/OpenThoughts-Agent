#!/usr/bin/env python3
"""Rewire the three exp_rpt datasets with the canonical RL verifier pattern.

For every task we install the nemotron-cpp-v2 / rl_converters contract:
  * default-fail ``tests/test.sh`` (``echo 0 > reward.txt`` up front)
  * framework-appropriate runner, no pipe masking, exit codes captured directly
  * the verifier confirms tests actually ran before awarding reward 1
  * ``tests/test_state.py`` (+ the trailing ``python3 -m pytest /tests/test_state.py``)
  * a Dockerfile that carries ``python3-pytest`` so test_state.py can run

Source-specific work:
  * quixbugs-python -> real per-algorithm pytest suites (see quixbugs_gen) +
    a known-correct gold solution; standard python test.sh / Dockerfile.
  * bugswarm        -> self-contained java (javac + plain-assert runner) and
    python (validation-only) verifiers; the synthetic upstream test files that
    imported the real ``requests`` library / undefined org.json are replaced.
  * stack-rspec     -> ruby rspec runner with a /app load-path bootstrap, default-
    fail, no pipe, "examples ran" guard; Dockerfile gains python3 + python3-pytest.

Usage:
    python -m data.rl_converters.fix_exp_rpt process   --workdir <dir>
    python -m data.rl_converters.fix_exp_rpt smoke     --workdir <dir> [--source NAME]
    python -m data.rl_converters.fix_exp_rpt upload    --workdir <dir>
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import tarfile
from pathlib import Path

from . import templates as T
from .quixbugs_gen import ALGOS, build_gold_solution, build_test_solution

SOURCES = {
    "quixbugs": {
        "src_repo": "DCAgent/exp_rpt_quixbugs-python",
        "dst_repo": "laion/exp_rpt_quixbugs-v2",
        "language": "python",
    },
    "bugswarm": {
        "src_repo": "DCAgent/exp_rpt_bugswarm",
        "dst_repo": "laion/exp_rpt_bugswarm-v2",
        "language": "multi",
    },
    "stack-rspec": {
        "src_repo": "DCAgent/exp_rpt_stack-rspec",
        "dst_repo": "laion/exp_rpt_stack-rspec-v2",
        "language": "ruby",
    },
}


# python:3.10-slim ships its own interpreter in /usr/local, so the apt
# ``python3-pytest`` package (which targets /usr/bin/python3) is NOT visible to
# ``python3 -m pytest``.  Install pytest via pip into /usr/local too so the
# verifier's ``python3 -m pytest`` resolves; keep the apt package as required
# by the standard pattern.
PYTHON_DOCKERFILE = """\
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y python3-pytest && rm -rf /var/lib/apt/lists/* && \\
    pip install --no-cache-dir pytest
"""


# --------------------------------------------------------------------------- #
# tar <-> dict helpers
# --------------------------------------------------------------------------- #

def read_task(task_binary: bytes) -> dict[str, bytes]:
    """Decompress a gzip-tar task binary into {path: content}."""
    raw = gzip.decompress(task_binary)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for m in tar.getmembers():
            if m.isfile():
                files[m.name] = tar.extractfile(m).read()
    return files


def write_task(files: dict[str, bytes]) -> bytes:
    """Re-encode {path: content} as a gzip-tar binary."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            ti = tarfile.TarInfo(name=name)
            ti.size = len(content)
            tar.addfile(ti, io.BytesIO(content))
    return gzip.compress(buf.getvalue())


# --------------------------------------------------------------------------- #
# stack-rspec custom verifier
# --------------------------------------------------------------------------- #

STACK_RSPEC_DOCKERFILE = """\
FROM ruby:3.2-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ruby-dev \\
    libffi-dev \\
    bash \\
    python3 \\
    python3-pytest \\
    && rm -rf /var/lib/apt/lists/*

RUN gem install rspec minitest --no-document
"""

STACK_RSPEC_TEST_SH = """\
#!/bin/bash
set -u

REWARD=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$REWARD"

cd /app

# Best-effort Gemfile install (no internet -> ignored on failure).
if [ -f Gemfile ]; then
    bundle install --quiet >/dev/null 2>&1 || true
fi

# Loader: put /app on the load path and require every *.rb under /app so the
# spec's `require "foo/bar"` resolves to /app/foo/bar.rb.
cat > /tmp/load_app.rb <<'RUBY'
$LOAD_PATH.unshift('/app')
Dir.glob('/app/**/*.rb').sort.each do |file|
  begin
    require file
  rescue LoadError, NameError
    # a missing constant will surface as a failing example below
  end
end
RUBY

# No pipe -> we capture the real exit code of the runner.
if grep -Eq 'RSpec|describe|expect' /tests/test_solution.rb 2>/dev/null; then
    timeout 300 rspec --require /tmp/load_app.rb --format documentation \\
        /tests/test_solution.rb > /logs/verifier/test_output.txt 2>&1
    RUN_EXIT=$?
    # Reward 1 only when rspec passed AND at least one example actually ran.
    if [ "$RUN_EXIT" -eq 0 ] && grep -Eq '[1-9][0-9]* examples?' /logs/verifier/test_output.txt; then
        echo 1 > "$REWARD"
    fi
else
    timeout 300 ruby -r /tmp/load_app.rb /tests/test_solution.rb > /logs/verifier/test_output.txt 2>&1
    RUN_EXIT=$?
    if [ "$RUN_EXIT" -eq 0 ]; then
        echo 1 > "$REWARD"
    fi
fi

python3 -m pytest /tests/test_state.py --tb=short 2>/dev/null
exit 0
"""


# --------------------------------------------------------------------------- #
# bugswarm custom verifiers
# --------------------------------------------------------------------------- #

BUGSWARM_JAVA_DOCKERFILE = """\
FROM eclipse-temurin:17-jdk

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pytest && \\
    rm -rf /var/lib/apt/lists/*
"""

BUGSWARM_JAVA_TEST_SH = """\
#!/bin/bash
set -u

REWARD=/logs/verifier/reward.txt
mkdir -p /logs/verifier
echo 0 > "$REWARD"

cd /app
cp /tests/TestSolution.java /app/ 2>/dev/null

# Compile the agent's sources together with the JUnit-free test harness.
mkdir -p /app/out
javac -d /app/out /app/*.java > /logs/verifier/compile_output.txt 2>&1
CC=$?
if [ "$CC" -ne 0 ]; then
    echo "compile failed -> reward 0"
    python3 -m pytest /tests/test_state.py --tb=short 2>/dev/null
    exit 0
fi

timeout 120 java -cp /app/out TestSolution > /logs/verifier/test_output.txt 2>&1
RUN=$?
if [ "$RUN" -eq 0 ] && grep -q "ALL TESTS PASSED" /logs/verifier/test_output.txt; then
    echo 1 > "$REWARD"
fi

python3 -m pytest /tests/test_state.py --tb=short 2>/dev/null
exit 0
"""

# Self-contained gold for the JSONParser empty-string CI fix (no org.json dep).
BUGSWARM_JAVA_GOLD = """\
public class JSONParser {
    public Object parse(String json) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        if (json.startsWith("{")) {
            return parseObject(json);
        }
        return parseArray(json);
    }

    private Object parseObject(String json) {
        return new Object();
    }

    private Object parseArray(String json) {
        return new Object();
    }
}
"""

BUGSWARM_JAVA_TEST = """\
import java.util.Objects;

/** JUnit-free harness: the upstream CI project shipped no pom.xml / build
 *  descriptor, so we compile + run a plain main that asserts the empty/null
 *  guard the fix introduces. */
public class TestSolution {
    static int passed = 0;
    static int failed = 0;

    public static void main(String[] args) {
        JSONParser parser = new JSONParser();
        check(parser.parse("{}") != null, "parse object -> non-null");
        check(parser.parse("[]") != null, "parse array -> non-null");
        check(parser.parse("") == null, "parse empty string -> null");
        check(parser.parse(null) == null, "parse null -> null");
        System.out.println("RESULTS passed=" + passed + " failed=" + failed);
        if (failed == 0 && passed > 0) {
            System.out.println("ALL TESTS PASSED");
        } else {
            System.exit(1);
        }
    }

    static void check(boolean cond, String name) {
        if (cond) {
            passed++;
            System.out.println("PASS: " + name);
        } else {
            failed++;
            System.out.println("FAIL: " + name);
        }
    }
}
"""

# Self-contained gold for the requests URL-validation fix (no `requests` lib /
# network needed). The bug was "doesn't validate URL"; the fix raises on empty.
BUGSWARM_PY_GOLD = """\
def get(url, params=None, **kwargs):
    \"\"\"Sends a GET request.\"\"\"
    if not url:
        raise ValueError("URL is required")
    kwargs.setdefault('allow_redirects', True)
    return ('get', url, params, kwargs)


def post(url, data=None, json=None, **kwargs):
    \"\"\"Sends a POST request.\"\"\"
    if not url:
        raise ValueError("URL is required")
    return ('post', url, data, json, kwargs)
"""

BUGSWARM_PY_TEST = """\
import sys
sys.path.insert(0, '/app')

import pytest

from solution import get, post


def test_get_empty_string_raises():
    with pytest.raises(ValueError):
        get("")


def test_get_none_raises():
    with pytest.raises(ValueError):
        get(None)


def test_post_empty_string_raises():
    with pytest.raises(ValueError):
        post("")


def test_post_none_raises():
    with pytest.raises(ValueError):
        post(None)


def test_get_valid_url_returns_value():
    # The fixed get must not raise for a non-empty url and must return something.
    result = get("https://example.com/get")
    assert result is not None


def test_post_valid_url_returns_value():
    result = post("https://example.com/post")
    assert result is not None
"""


# --------------------------------------------------------------------------- #
# per-source transformers
# --------------------------------------------------------------------------- #

def _common_verifier(files: dict[str, bytes], language: str) -> None:
    """Drop in the shared default-fail scaffolding shared by all tasks."""
    files["environment/Dockerfile"] = T.get_dockerfile(language).encode("utf-8")
    files["tests/test.sh"] = T.get_test_sh(language).encode("utf-8")
    files["tests/test_state.py"] = T.get_test_state_py().encode("utf-8")
    files["tests/config.json"] = T.get_config_json().encode("utf-8")


def transform_quixbugs(files: dict[str, bytes], idx: int) -> dict[str, bytes]:
    name = ALGOS[idx]
    out = {k: v for k, v in files.items()
           if not k.startswith("tests/") and not k.startswith("environment/")
           and not k.startswith("solution/")}
    # keep instruction.md / task.toml / metadata.json from upstream
    _common_verifier(out, "python")
    out["environment/Dockerfile"] = PYTHON_DOCKERFILE.encode("utf-8")
    out["tests/test_solution.py"] = build_test_solution(name).encode("utf-8")
    out["solution/solution.py"] = build_gold_solution(name).encode("utf-8")
    out["task.toml"] = T.get_task_toml().encode("utf-8")
    return out


def transform_bugswarm(files: dict[str, bytes], idx: int) -> dict[str, bytes]:
    md = _parse_metadata(files)
    lang = md.get("language", "").lower()
    out = {k: v for k, v in files.items()
           if not k.startswith("tests/") and not k.startswith("environment/")
           and not k.startswith("solution/")}

    # Make the agent path consistent (/app), drop the synthetic /workspace ref.
    instr = out.get("instruction.md", b"").decode("utf-8", "replace")
    instr = instr.replace("/workspace/", "/app/")
    out["instruction.md"] = instr.encode("utf-8")

    if lang == "java":
        out["environment/Dockerfile"] = BUGSWARM_JAVA_DOCKERFILE.encode("utf-8")
        out["tests/test.sh"] = BUGSWARM_JAVA_TEST_SH.encode("utf-8")
        out["tests/test_state.py"] = T.get_test_state_py().encode("utf-8")
        out["tests/config.json"] = T.get_config_json().encode("utf-8")
        out["tests/TestSolution.java"] = BUGSWARM_JAVA_TEST.encode("utf-8")
        # Gold class file named for its public class so javac is happy.
        out["solution/JSONParser.java"] = BUGSWARM_JAVA_GOLD.encode("utf-8")
    else:  # python
        _common_verifier(out, "python")
        out["environment/Dockerfile"] = PYTHON_DOCKERFILE.encode("utf-8")
        out["tests/test_solution.py"] = BUGSWARM_PY_TEST.encode("utf-8")
        out["solution/solution.py"] = BUGSWARM_PY_GOLD.encode("utf-8")

    out["task.toml"] = T.get_task_toml().encode("utf-8")
    return out


def transform_stack_rspec(files: dict[str, bytes], idx: int) -> dict[str, bytes]:
    # Preserve everything (incl. the upstream test_solution.rb), only replacing
    # the verifier scaffolding we own.
    drop = {
        "environment/Dockerfile",
        "tests/test.sh", "tests/test_state.py", "tests/config.json",
        "task.toml",
    }
    out = {k: v for k, v in files.items() if k not in drop}
    out["environment/Dockerfile"] = STACK_RSPEC_DOCKERFILE.encode("utf-8")
    out["tests/test.sh"] = STACK_RSPEC_TEST_SH.encode("utf-8")
    out["tests/test_state.py"] = T.get_test_state_py().encode("utf-8")
    out["tests/config.json"] = T.get_config_json().encode("utf-8")
    out["task.toml"] = T.get_task_toml().encode("utf-8")
    return out


TRANSFORMS = {
    "quixbugs": transform_quixbugs,
    "bugswarm": transform_bugswarm,
    "stack-rspec": transform_stack_rspec,
}


def _parse_metadata(files: dict[str, bytes]) -> dict:
    import json
    raw = files.get("metadata.json", b"{}")
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #

def _load_source(src_repo: str, cache: Path) -> "list[tuple[str, bytes]]":
    import pandas as pd
    from huggingface_hub import hf_hub_download
    pq = hf_hub_download(
        repo_id=src_repo, filename="tasks.parquet", repo_type="dataset",
        local_dir=str(cache / src_repo.replace("/", "__")),
    )
    df = pd.read_parquet(pq)
    return [(r["path"], r["task_binary"]) for _, r in df.iterrows()]


def process(workdir: Path) -> dict[str, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    out_paths: dict[str, Path] = {}
    for key, cfg in SOURCES.items():
        rows = _load_source(cfg["src_repo"], workdir / "downloads")
        transform = TRANSFORMS[key]
        new_rows: list[tuple[str, bytes]] = []
        for idx, (path, blob) in enumerate(rows):
            files = read_task(blob)
            new_files = transform(files, idx)
            new_rows.append((path, write_task(new_files)))
        out_path = workdir / "out" / f"{key}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(new_rows, out_path)
        out_paths[key] = out_path
        print(f"[{key}] {len(new_rows)} tasks -> {out_path}")
    return out_paths


def _write_parquet(rows: list[tuple[str, bytes]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "path": [p for p, _ in rows],
        "task_binary": [b for _, b in rows],
    })
    pq.write_table(table, str(path))


# --------------------------------------------------------------------------- #
# docker smoke (empty /app -> reward 0) + gold gate where a solution exists
# --------------------------------------------------------------------------- #

def _materialize_task(files: dict[str, bytes], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _docker_run_verifier(image: str, task_dir: Path, populate_gold: bool, timeout: int = 240) -> tuple[int, str]:
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as wd:
        wd = Path(wd)
        app = wd / "app"; tests = wd / "tests"; logs = wd / "logs" / "verifier"  # noqa: E702
        app.mkdir(); tests.mkdir(); logs.mkdir(parents=True)  # noqa: E702
        src_tests = task_dir / "tests"
        if src_tests.exists():
            for f in src_tests.iterdir():
                shutil.copy2(f, tests / f.name)
        if populate_gold:
            sol = task_dir / "solution"
            if sol.exists():
                for f in sol.iterdir():
                    if f.name != "solve.sh":
                        shutil.copy2(f, app / f.name)
        # Mount the host app dir directly onto /app (Harbor semantics: the
        # agent's working directory IS /app, populated with the solution).
        cmd = (
            "mkdir -p /logs/verifier; cd /app; "
            "bash /tests/test.sh >/tmp/v.txt 2>&1; "
            "echo '---REWARD---'; cat /logs/verifier/reward.txt 2>/dev/null || echo NO_REWARD"
        )
        try:
            r = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{app}:/app",
                 "-v", f"{tests}:/tests",
                 "-v", f"{wd}/logs:/logs",
                 image, "bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"---REWARD---\s*\n?\s*([01])", out)
            reward = int(m.group(1)) if m else -1
            return reward, out
        except subprocess.TimeoutExpired:
            return -1, "TIMEOUT"
        except Exception as e:
            return -1, f"ERR {e}"


def _build_image(dockerfile_text: str, tag: str) -> str:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Dockerfile").write_text(dockerfile_text)
        subprocess.run(["docker", "build", "-q", "-t", tag, d], check=True,
                       capture_output=True)
    return tag


def smoke(workdir: Path, source: str | None = None, sample: int = 3) -> None:
    keys = [source] if source else list(SOURCES)
    for key in keys:
        SOURCES[key]
        pq_path = workdir / "out" / f"{key}.parquet"
        import pandas as pd
        df = pd.read_parquet(pq_path)
        n = min(sample, len(df))
        print(f"\n=== SMOKE {key} ({n} of {len(df)} tasks) ===")
        ok_empty0 = 0
        ok_gold1 = 0
        gold_exists = 0
        for i in range(n):
            blob = df.iloc[i]["task_binary"]
            files = read_task(blob)
            with tempfile_dir() as td:
                task_dir = td / "task"
                _materialize_task(files, task_dir)
                # Pick Dockerfile text + image tag.
                if key == "quixbugs":
                    df_text = PYTHON_DOCKERFILE; tag = "smoke_quixbugs"  # noqa: E702
                elif key == "bugswarm":
                    lang = _parse_metadata(files).get("language", "").lower()
                    if lang == "java":
                        df_text = BUGSWARM_JAVA_DOCKERFILE; tag = "smoke_bs_java"  # noqa: E702
                    else:
                        df_text = PYTHON_DOCKERFILE; tag = "smoke_bs_py"  # noqa: E702
                else:
                    df_text = STACK_RSPEC_DOCKERFILE; tag = "smoke_rspec"  # noqa: E702
                _build_image(df_text, tag)
                # empty /app -> must be reward 0
                r0, out0 = _docker_run_verifier(tag, task_dir, populate_gold=False)
                # gold -> should be reward 1 (only when a solution/ exists)
                has_gold = (task_dir / "solution").exists()
                if has_gold:
                    gold_exists += 1
                    r1, out1 = _docker_run_verifier(tag, task_dir, populate_gold=True)
                else:
                    r1, out1 = -2, "no-solution"
                print(f"  [{key}#{i}] empty_reward={r0} gold_reward={r1}"
                      + ("" if r0 == 0 else f"  <<< EMPTY NOT 0\n    {out0[-300:]}"))
                if r0 == 0:
                    ok_empty0 += 1
                if r1 == 1:
                    ok_gold1 += 1
                if r0 != 0 or (has_gold and r1 != 1):
                    print("    --- empty out tail ---"); print(out0[-500:])  # noqa: E702
                    if has_gold:
                        print("    --- gold out tail ---"); print(out1[-700:])  # noqa: E702
        print(f"  summary: empty->0 ok {ok_empty0}/{n}; gold->1 ok {ok_gold1}/{gold_exists}")


import contextlib  # noqa: E402


@contextlib.contextmanager
def tempfile_dir():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #

README_BODY = """\
# {dst}

Verifier-fixed version of `{src}`.

Each task follows the canonical RL verifier contract:

* `tests/test.sh` defaults to reward 0, runs the framework-appropriate test
  runner without pipe masking, captures the exit code directly, confirms tests
  actually ran, and only then writes reward 1.
* `tests/test_state.py` asserts `reward.txt == "1"` (run via
  `python3 -m pytest /tests/test_state.py` at the end of test.sh).
* The `environment/Dockerfile` carries `python3-pytest` so test_state.py runs.

{notes}

`{n_tasks}` tasks.
"""


def upload(workdir: Path) -> None:
    from huggingface_hub import HfApi, create_repo
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    notes = {
        "quixbugs": (
            "QuixBugs-Python bug-fix tasks. The upstream `test_solution.py` was a\n"
            "no-op stub (import-only, zero assertions); it is replaced with a real\n"
            "per-algorithm pytest suite (40 algorithms) whose expected outputs are\n"
            "computed from textbook-correct references. `solution/solution.py` is the\n"
            "known-correct gold. Graph/linked-list algorithms ship a `Node` fixture."
        ),
        "bugswarm": (
            "BugSwarm-style CI fail->pass pairs. The upstream synthetic tests\n"
            "imported the real `requests` library / undefined `org.json` and needed\n"
            "network; they are replaced with self-contained verifiers that compile\n"
            "and run the shipped gold (Java: javac + plain-assert harness; Python:\n"
            "URL-validation behavior)."
        ),
        "stack-rspec": (
            "Ruby (rspec) tasks from The Stack. The verifier is default-fail, runs\n"
            "rspec without pipe masking, requires at least one example to actually\n"
            "run before awarding reward 1, and bootstraps `/app` onto the load path.\n"
            "The image gains python3 + python3-pytest for test_state.py."
        ),
    }
    for key, cfg in SOURCES.items():
        pq_path = workdir / "out" / f"{key}.parquet"
        import pandas as pd
        n = len(pd.read_parquet(pq_path))
        readme = README_BODY.format(
            dst=cfg["dst_repo"], src=cfg["src_repo"], notes=notes[key], n_tasks=n,
        )
        create_repo(cfg["dst_repo"], repo_type="dataset", token=token, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(pq_path), path_in_repo="tasks.parquet",
            repo_id=cfg["dst_repo"], repo_type="dataset", token=token,
        )
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
            repo_id=cfg["dst_repo"], repo_type="dataset", token=token,
        )
        print(f"[{key}] uploaded -> https://huggingface.co/datasets/{cfg['dst_repo']}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["process", "smoke", "upload"])
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--source", default=None, help="smoke: single source key")
    p.add_argument("--sample", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "process":
        process(args.workdir)
    elif args.cmd == "smoke":
        smoke(args.workdir, args.source, args.sample)
    else:
        upload(args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
