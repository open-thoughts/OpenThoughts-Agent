#!/usr/bin/env python3
"""Build a self-contained, conservative stack-ruby v3 task population.

The v2 patcher exposed test symbols in the prompt but left dependency setup to
best-effort network installs in each verifier. It also treated many project
helpers as RubyGems. This module provides one pinned shared image and admits
only tests whose support contract is present in the test or in that image.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from data.patchers.patch_stack_ruby_tasks import parse_test_ruby

V3_MARKER = "<!-- laion v3 instruction patch: self-contained Ruby verifier -->"

KNOWN_INVALID_TASKS = frozenset(
    {
        "stack-ruby-0028",
        "stack-ruby-0033",
        "stack-ruby-0035",
        "stack-ruby-0036",
        "stack-ruby-0117",
        "stack-ruby-0147",
        "stack-ruby-0151",
        "stack-ruby-0152",
        "stack-ruby-0165",
        "stack-ruby-0174",
    }
)

# These require heads have an unambiguous package in the pinned image. A
# require whose head is absent is a project helper or an unsupported framework
# and causes the task to be rejected.
SUPPORTED_REQUIRE_HEADS = frozenset(
    {
        "English",
        "active_support",
        "bundler",
        "byebug",
        "capybara",
        "dry",
        "faraday",
        "jsonpath",
        "mocha",
        "nokogiri",
        "pp",
        "pry",
        "rack",
        "rake",
        "recursive-open-struct",
        "rest-client",
        "rspec-parameterized",
        "simplecov",
        "tilt",
        "timecop",
        "vcr",
        "webrick",
        "webmock",
    }
)

PINNED_GEMS = (
    ("rspec", "3.13.2"),
    ("minitest", "5.25.5"),
    ("simplecov", "0.22.0"),
    ("rack", "2.2.17"),
    ("rack-test", "2.2.0"),
    ("pry", "0.15.2"),
    ("byebug", "12.0.0"),
    ("nokogiri", "1.18.10"),
    ("webmock", "3.25.1"),
    ("timecop", "0.9.10"),
    ("webrick", "1.9.1"),
    ("vcr", "6.3.1"),
    ("rest-client", "2.1.0"),
    ("httparty", "0.23.1"),
    ("plist", "3.7.2"),
    ("mocha", "2.7.1"),
    ("rspec-parameterized", "1.0.2"),
    ("recursive-open-struct", "1.3.1"),
    ("dry-system", "1.2.4"),
    ("dry-validation", "1.11.1"),
    ("activesupport", "7.1.5.2"),
    ("faraday", "2.13.4"),
    ("jsonpath", "1.1.5"),
    ("capybara", "3.40.0"),
    ("bundler", "2.6.9"),
    ("rake", "13.2.1"),
    ("tilt", "2.6.0"),
)

_SHARED_DEFINITION = re.compile(
    r"""(?:shared_examples?|shared_context)\s*(?:\(?\s*)?"""
    r"""(?:["']([^"']+)["']|:([A-Za-z_][A-Za-z0-9_]*))"""
)
_SHARED_USE = re.compile(
    r"""(?:it_behaves_like|it_should_behave_like|include_examples|include_context)"""
    r"""\s*(?:\(?\s*)?(?:["']([^"']+)["']|:([A-Za-z_][A-Za-z0-9_]*))"""
)
_EXAMPLE = re.compile(r"""\b(?:it|specify|scenario|test)\s*(?:\(|["'])|\bdef\s+test_""")

# These calls require absent project fixtures, Rails plugins, factory
# definitions, or suite-level support. Their semantics cannot be recovered
# from a single transplanted test file.
_UNPACKAGED_SUPPORT = re.compile(
    r"""
    \b(?:build|build_stubbed|create|attributes_for|create_list|build_list)\s*(?:\(|:)
    |\b(?:Samples|fixture|fixtures|fixture_file_upload|file_fixture)\b
    |\b(?:PlatformHelpers|with_platform|unless_platform|Platform\.match)\b
    |\b(?:TildeConfigSpec|TestHelpers|SpecHelpers|SupportHelpers|[A-Z][A-Za-z0-9_]*HelperModule)\b
    |\bCompilationSupport\b
    |\b(?:travel_to|freeze_time|let_it_be|before_all|perform_enqueued_jobs)\b
    |\b(?:validate_[a-z_]+_of|have_db_[a-z_]+|have_many|have_one|belong_to)\b
    |\b(?:expect_offense|expect_no_offenses|inspect_source|expect_correction)\b
    |\b(?:RuboCop::Cop::Offense|source_range)\b
    |\b(?:fixture_file|mock_service_process|wait_for_mock_service_to_start|with_process)\s*\(
    |\bexpect_successful_request\s*\(
    |\btokens\s*\(
    |\b(?:a_value|inspect_[A-Z][A-Za-z0-9_]*Response)\b
    |\bRack::Builder\.parse_file\b
    |\bFaker::
    |\binclude\s+Capybara::DSL\b
    |\bENV\[['"]UI_TEST['"]\]
    |\bWebDriverUtils\b
    |\bexpect\(false\)\.to\s+eq\(true\)
    |\bFactoryBot\b
    |\bFabricate(?:\.times)?\b
    |\bTunesStubbing\b
    |\b(?:build_git|update_git|install_gemfile!?|bundled_app|lib_path|the_bundle)\b
    |\b(?:include|extend)\s+FileHelper\b
    |\bMiqPassword\b
    |\bENV\.fetch\b
    |\bcontext\.album\.sentences\b
    |\btype:\s*:(?:controller|feature|helper|job|mailer|model|request|routing|system|view)\b
    |\bVCR\.use_cassette\b
    |\brequire\s+['"]minitest/homework['"]
    |(?<![A-Za-z0-9_])__(?![A-Za-z0-9_])
    |\bmaybe\s+[^\n]+
    """,
    re.VERBOSE,
)
_UNPACKAGED_FILE_FIXTURE = re.compile(
    r"""(?:File\.(?:open|read|binread|readlines)|IO\.(?:read|readlines)|"""
    r"""YAML\.load_file|CSV\.(?:read|foreach)|Pathname(?:\.new|\())"""
    r"""[^\n]*(?:__FILE__|(?:spec|test|tests|data|fixtures|support|out)/)"""
)
_EXTERNAL_PROCESS = re.compile(r"\b(?:system|spawn|exec)\s*\(")

DOCKERFILE = (
    "FROM ruby:3.2-slim\n\n"
    "WORKDIR /app\n\n"
    "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
    "    bash build-essential git libffi-dev liblzma-dev libxml2-dev \\\n"
    "    libxslt1-dev pkg-config ruby-dev zlib1g-dev \\\n"
    "    && rm -rf /var/lib/apt/lists/*\n\n"
    + "RUN "
    + " && \\\n    ".join(
        f"gem install {name} --version {version} --no-document"
        for name, version in PINNED_GEMS
    )
    + "\n"
)

TEST_SH = r"""#!/bin/bash
set -u

REWARD=/logs/verifier/reward.txt
OUTPUT=/logs/verifier/test_output.txt
mkdir -p /logs/verifier
echo 0 > "$REWARD"
cd /app

cat > /tmp/load_app.rb <<'RUBY'
$LOAD_PATH.unshift('/app')
require 'minitest'
require 'active_support'
require 'active_support/core_ext'
require 'bundler'
require 'rack'
require 'dry/core'
require 'dry/validation'
require 'open3'
require 'plist'
MiniTest = Minitest unless defined?(MiniTest)
class String
  def strip_indent
    indent = scan(/^[ \t]*(?=\S)/).min_by(&:length)
    indent ? gsub(/^#{Regexp.escape(indent)}/, '') : self
  end
end
module EnvironmentValueTestHelper
  def with_env_values(values)
    previous = values.to_h { |key, _| [key, ENV[key]] }
    values.each { |key, value| ENV[key] = value }
    yield
  ensure
    previous.each do |key, value|
      value.nil? ? ENV.delete(key) : ENV[key] = value
    end
  end
end
RSpec.configure { |config| config.include EnvironmentValueTestHelper } if defined?(RSpec)
Dir.glob('/app/**/*.rb').sort.each do |file|
  begin
    require file
  rescue LoadError, NameError => error
    warn "AUTOLOAD #{file}: #{error.class}: #{error.message}"
  end
end
RUBY

if grep -Eq 'RSpec|describe|expect' /tests/test_solution.rb 2>/dev/null; then
    timeout 300 rspec --require /tmp/load_app.rb --format documentation \
        /tests/test_solution.rb 2>&1 | tee "$OUTPUT"
    RUN_EXIT=${PIPESTATUS[0]}
    if [ "$RUN_EXIT" -eq 0 ] && grep -Eq '[1-9][0-9]* examples?' "$OUTPUT"; then
        echo 1 > "$REWARD"
    fi
else
    timeout 300 ruby -r /tmp/load_app.rb /tests/test_solution.rb 2>&1 | tee "$OUTPUT"
    RUN_EXIT=${PIPESTATUS[0]}
    if [ "$RUN_EXIT" -eq 0 ] && \
       grep -Eq '([1-9][0-9]* runs?|[1-9][0-9]* tests?), [1-9][0-9]* assertions?' "$OUTPUT"; then
        echo 1 > "$REWARD"
    fi
fi

echo "runner_exit=$RUN_EXIT"
echo "reward=$(cat "$REWARD")"
exit 0
"""


def admission_reasons(task_name: str, test_source: str) -> tuple[str, ...]:
    """Return deterministic reasons that a transplanted test is not self-contained."""
    reasons: list[str] = []
    if task_name in KNOWN_INVALID_TASKS:
        reasons.append("audited-invalid")

    parsed = parse_test_ruby(test_source)
    if parsed is None:
        reasons.append("unparseable-contract")
    else:
        if parsed["requires_local_files"]:
            reasons.append("unpackaged-local-require")
        require_heads = {
            require.split("/", 1)[0] for require in parsed["requires_third_party_gems"]
        }
        if require_heads - SUPPORTED_REQUIRE_HEADS:
            reasons.append("unsupported-or-project-require")

    defined_shared = {
        name
        for match in _SHARED_DEFINITION.findall(test_source)
        for name in match
        if name
    }
    used_shared = {
        name for match in _SHARED_USE.findall(test_source) for name in match if name
    }
    missing_shared = used_shared - defined_shared
    if missing_shared:
        reasons.append("missing-shared-example")
    if defined_shared and not defined_shared.intersection(used_shared):
        reasons.append("uninvoked-shared-example")
    if _EXAMPLE.search(test_source) is None:
        reasons.append("no-executable-example")
    if _UNPACKAGED_SUPPORT.search(test_source):
        reasons.append("unpackaged-suite-support")
    if _UNPACKAGED_FILE_FIXTURE.search(test_source):
        reasons.append("unpackaged-file-fixture")
    if _EXTERNAL_PROCESS.search(test_source):
        reasons.append("external-process-contract")
    return tuple(dict.fromkeys(reasons))


def rewrite_instruction(instruction: str) -> str:
    """Update the v2 environment contract without altering task semantics."""
    old = (
        "- Pre-installed gems: `rspec`, `minitest`. Other 3rd-party gems "
        "(see list above) are best-effort `gem install`-ed by `tests/test.sh` "
        "before running the test."
    )
    new = (
        "- Required support gems are pinned in the shared image. The verifier "
        "does not install packages or access the network at runtime."
    )
    if old not in instruction:
        raise ValueError("v2 environment contract is absent")
    old_grader = (
        "- The grader (`tests/test.sh`) runs `bundle install` if `/app/Gemfile` "
        "exists, then auto-requires every `*.rb` file under `/app/` (recursively) "
        "via `$LOAD_PATH.unshift('/app')`. So you can place sources at any "
        "depth under `/app/` as long as relative-require paths resolve."
    )
    new_grader = (
        "- The grader auto-requires every `*.rb` file under `/app/` recursively "
        "via `$LOAD_PATH.unshift('/app')`. It does not run Bundler or install "
        "dependencies at verification time."
    )
    if old_grader not in instruction:
        raise ValueError("v2 grader contract is absent")
    return (
        instruction.replace(old, new, 1)
        .replace(old_grader, new_grader, 1)
        .replace(
            "<!-- laion v2 instruction patch: enriched with Ruby test contract -->",
            "<!-- laion v2 instruction patch: enriched with Ruby test contract -->\n\n"
            + V3_MARKER,
            1,
        )
    )


def patch_task(task_dir: Path) -> tuple[str, ...]:
    """Patch an extracted task or return its rejection reasons."""
    test_path = task_dir / "tests" / "test_solution.rb"
    if not test_path.is_file():
        return ("missing-test-solution",)
    source = test_path.read_text(encoding="utf-8", errors="replace")
    reasons = admission_reasons(task_dir.name, source)
    if reasons:
        return reasons
    instruction_path = task_dir / "instruction.md"
    instruction_path.write_text(
        rewrite_instruction(
            instruction_path.read_text(encoding="utf-8", errors="replace")
        ),
        encoding="utf-8",
    )
    (task_dir / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task_dir / "tests" / "test.sh").write_text(TEST_SH)
    return ()


def main() -> None:
    """Patch an extracted population and optionally remove rejected tasks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--drop-rejected", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    manifest: dict[str, list[str]] = {}
    retained = 0
    for task_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        reasons = patch_task(task_dir)
        if reasons:
            manifest[task_dir.name] = list(reasons)
            if args.drop_rejected:
                shutil.rmtree(task_dir)
        else:
            retained += 1
    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"retained={retained} rejected={len(manifest)}")


if __name__ == "__main__":
    main()
