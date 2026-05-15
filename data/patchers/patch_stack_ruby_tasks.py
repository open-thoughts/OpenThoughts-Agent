#!/usr/bin/env python3
"""
exp_rpt_stack-ruby v2 patcher.

Bug (v1 → v2): The QC sample shows ~8/10 trials fail with classic infra
errors — `LoadError: cannot load such file -- 'foo'` or
`NameError: uninitialized constant Foo::Bar`. The verifier's
`tests/test_solution.rb` requires real-world Ruby project gems (e.g.
`gapic/grpc/service_stub`, `chefspec`, `dalli_store`) and references SUT
classes (`AdventOfCode2017::Day2::SpreadsheetLine`, `BankAccount`,
`Stats`, ...) that the LLM-authored `instruction.md` does not enumerate
explicitly. The agent has no deterministic contract to satisfy.

Fix (v2): For each task, mechanically parse `tests/test_solution.rb`:
  - Extract `require '<gem>'` and `require_relative '<file>'` lines —
    these are the gems / local files the test pulls in.
  - Extract describe targets: `RSpec.describe X` / `describe X`. These
    are the top-level types under test.
  - Extract referenced fully-qualified constant references (e.g.
    `Foo::Bar`, `Foo.new`, `class X < Foo`) — the public API the agent
    must implement.
  - Extract method calls of the form `obj.method(...)` with rough arity
    — gives the agent a method-signature stub list.

Then rewrite `instruction.md` to prepend a `## Test contract` block.
The original LLM-generated description is preserved verbatim under a
sub-heading (`## Original task description`) so the agent has both the
deterministic contract AND the prose explanation.

We also patch `tests/test.sh` to `gem install` each required gem
(best-effort, `|| true`) before running rspec/ruby. This is gated by a
separate marker so the per-file test.sh edit is idempotent.

If `tests/test_solution.rb` is unparseable (no class references AND no
require statements), the task is DROPPED — not silently shipped with a
vacuous prompt. The patcher reports the count.

Output cap: 8000 chars per `instruction.md`.

Usage:
  python data/patchers/patch_stack_ruby_tasks.py --root <dir> [--dry-run] [--limit N]
  python data/patchers/patch_stack_ruby_tasks.py --root <dir> --skip-test-sh
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Ruby parsing helpers (regex-based; tests are simple enough that a real
# parser would be overkill, and we have no ripper/parser gem on the path).
# --------------------------------------------------------------------------- #

# require 'foo' / require "foo/bar"
_REQUIRE_RE = re.compile(
    r"""^\s*require\s+(?:['"])([^'"]+)(?:['"])\s*$""",
    re.MULTILINE,
)
# require_relative 'foo'
_REQUIRE_REL_RE = re.compile(
    r"""^\s*require_relative\s+(?:['"])([^'"]+)(?:['"])\s*$""",
    re.MULTILINE,
)
# require File.expand_path(...) / require File.dirname(__FILE__) + '...' / etc.
# We capture just the string literal portion as a hint.
_REQUIRE_DYNAMIC_RE = re.compile(
    r"""^\s*require\s+File\.[^'"\n]*?['"]([^'"]+)['"]""",
    re.MULTILINE,
)

# RSpec.describe Foo::Bar / describe Foo::Bar
_DESCRIBE_RE = re.compile(
    r"""\b(?:RSpec\.)?describe\s+([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)""",
)
# class Foo < Bar  /  class Foo::Bar < Baz
_CLASS_DECL_RE = re.compile(
    r"""\bclass\s+([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*(?:<\s*([A-Z][A-Za-z0-9_:]*))?""",
)
# module Foo / module Foo::Bar
_MODULE_DECL_RE = re.compile(
    r"""\bmodule\s+([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)""",
)
# Foo.new(...)  /  Foo::Bar.new(...)
_NEW_CALL_RE = re.compile(
    r"""\b([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\.new\s*\(([^)(]*)\)""",
)
# Foo::Bar  (any const reference)  -- collected to surface SUT API
_CONST_RE = re.compile(
    r"""\b([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)+)\b""",
)
# Bare top-level constants (single name, capitalised) — only kept if also
# referenced via `.new` or `class X < Y` / `describe X`. We do NOT mass-
# capture single CamelCase tokens since Ruby's stdlib (`String`, `Integer`,
# `Hash`, `Array`, `Proc`, `Time`, ...) would create huge noise.
_BARE_CONST_RE = re.compile(r"""\b[A-Z][a-z0-9][A-Za-z0-9_]+\b""")

# Method calls of the form `var.method(args)`. We exclude common stdlib /
# RSpec matcher helpers from the surfaced list to keep noise down.
_METHOD_CALL_RE = re.compile(
    r"""\.([a-z_][a-zA-Z0-9_]*[!?=]?)\s*\(([^)(]*)\)""",
)

# Comments and string literals — strip before identifier scans.
_LINE_COMMENT_RE = re.compile(r"#[^\n]*")
_DSTRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_SSTRING_RE = re.compile(r"'(?:\\.|[^'\\])*'")
# Heredoc / %w[] / %r{} are uncommon in test files; we don't bother.

# Ruby stdlib gem names — already on the ruby:3.2-slim base image and don't
# need a `gem install`. The Dockerfile (`gem install rspec minitest`) plus
# stdlib covers these; everything else is a 3rd-party gem the agent
# probably needs.
_RUBY_STDLIB_GEMS = frozenset({
    # core ruby stdlib (no gem install needed)
    "json", "yaml", "csv", "set", "ostruct", "stringio", "tempfile",
    "fileutils", "pathname", "uri", "net/http", "net/https", "net/smtp",
    "net/ftp", "net/imap", "open3", "open-uri", "openssl", "digest",
    "digest/md5", "digest/sha1", "digest/sha2", "base64", "securerandom",
    "socket", "tmpdir", "shellwords", "etc", "rbconfig", "delegate",
    "forwardable", "singleton", "observer", "thread", "monitor", "timeout",
    "logger", "benchmark", "cgi", "erb", "fiddle", "find", "io/console",
    "ipaddr", "irb", "matrix", "optparse", "prime", "psych", "racc",
    "readline", "rexml", "scanf", "sdbm", "shell", "syslog", "tracer",
    "tsort", "weakref", "win32ole", "zlib", "date", "time", "bigdecimal",
    "complex", "rational", "rubygems", "rss", "drb", "rinda",
    "io/wait", "io/event", "io/event/selector",
    # we explicitly install these in the dockerfile
    "rspec", "minitest", "minitest/autorun", "minitest/spec", "minitest/pride",
    "minitest/unit",
    # rspec sub-modules also present once rspec is installed
    "rspec/core", "rspec/expectations", "rspec/mocks", "rspec/version",
    "rspec/autorun",
})

# Common spec-helper file names — these are project-internal and need to be
# created by the agent (typically a no-op spec_helper.rb at /app or at the
# referenced relative path).
_SPEC_HELPER_HINTS = frozenset({
    "helper", "spec_helper", "test_helper", "minitest_helper",
    "rails_helper", "chef_helper", "vcr_helper", "init_test",
    "spec_fast_helper",
})

# Method names we DON'T want to surface as "SUT methods" because they're
# obvious RSpec / Ruby builtins and would be noise.
_RSPEC_BUILTIN_METHODS = frozenset({
    "to", "not_to", "to_not", "and", "or", "be", "be_a", "be_an",
    "be_kind_of", "be_instance_of", "eq", "eql", "equal", "match",
    "include", "have_attributes", "raise_error", "respond_to",
    "have", "should", "should_not", "expect", "allow", "receive",
    "and_return", "and_call_original", "and_raise", "with", "ordered",
    "describe", "context", "it", "specify", "before", "after", "around",
    "let", "let!", "subject", "shared_examples", "it_behaves_like",
    "it_should_behave_like", "include_examples", "shared_context",
    "double", "instance_double", "class_double", "object_double",
    "stub", "stub_const", "hide_const",
    # Ruby builtins
    "new", "inspect", "to_s", "to_str", "to_a", "to_ary", "to_h",
    "to_hash", "to_i", "to_int", "to_f", "to_proc", "to_sym",
    "size", "length", "count", "first", "last", "each", "map", "select",
    "reject", "reduce", "inject", "find", "any?", "all?", "none?",
    "empty?", "nil?", "is_a?", "kind_of?", "instance_of?",
    "respond_to?", "send", "public_send", "method", "methods",
    "instance_variable_get", "instance_variable_set",
    "class", "ancestors", "class_eval", "instance_eval", "module_eval",
    "freeze", "dup", "clone", "tap", "then", "yield_self",
    "puts", "print", "p", "pp", "warn", "raise", "throw", "catch",
    "format", "sprintf", "printf", "gets", "readline",
    "[]", "[]=", "<<", "push", "pop", "shift", "unshift", "concat",
    "merge", "merge!", "fetch", "delete", "store", "values", "keys",
    "split", "join", "strip", "chomp", "chop", "downcase", "upcase",
    "capitalize", "reverse", "sort", "sort_by", "uniq", "compact",
    "flatten", "zip", "group_by", "partition", "slice", "chunk",
    # Ruby's open-uri / IO basics
    "open", "close", "read", "write", "puts", "gets",
    # ActiveSupport-ish (often referenced)
    "create", "build", "save", "destroy", "update", "update!",
})


def _strip_comments_and_strings(src: str) -> str:
    """Remove # comments and "..." / '...' literals so identifier scans don't pick them up."""
    src = _LINE_COMMENT_RE.sub(" ", src)
    src = _DSTRING_RE.sub('""', src)
    src = _SSTRING_RE.sub("''", src)
    return src


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split a string by `sep` only at the top paren/angle/bracket depth."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def parse_test_ruby(src: str) -> dict | None:
    """Parse a test_solution.rb; return None if essential structure is missing."""
    requires: list[str] = []
    require_relatives: list[str] = []

    for m in _REQUIRE_RE.finditer(src):
        requires.append(m.group(1))
    for m in _REQUIRE_REL_RE.finditer(src):
        require_relatives.append(m.group(1))
    for m in _REQUIRE_DYNAMIC_RE.finditer(src):
        # File.expand_path-style requires — capture the literal portion as a
        # hint (typically `spec_helper`, `../spec_helper`, etc.)
        require_relatives.append(m.group(1))

    # describe targets — strong signal of the SUT
    describe_targets: list[str] = []
    for m in _DESCRIBE_RE.finditer(src):
        describe_targets.append(m.group(1))

    clean = _strip_comments_and_strings(src)

    # class/module declarations IN THE TEST FILE — agent doesn't need to
    # define these, but the parent classes (after `<`) are types it likely
    # needs.
    test_defined_consts: set[str] = set()
    parent_consts: set[str] = set()
    for m in _CLASS_DECL_RE.finditer(clean):
        test_defined_consts.add(m.group(1))
        if m.group(2):
            parent_consts.add(m.group(2))
    for m in _MODULE_DECL_RE.finditer(clean):
        test_defined_consts.add(m.group(1))

    # Constructor arities
    constructors: dict[str, set[int]] = {}
    for m in _NEW_CALL_RE.finditer(clean):
        cls = m.group(1)
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args, ",") if a.strip()])
        constructors.setdefault(cls, set()).add(arity)

    # Qualified constants (Foo::Bar — Ruby's `::` separator) — strong SUT signal
    qualified_consts: set[str] = set()
    for m in _CONST_RE.finditer(clean):
        qualified_consts.add(m.group(0))

    # Method calls
    method_calls: dict[str, set[int]] = {}
    for m in _METHOD_CALL_RE.finditer(clean):
        name = m.group(1)
        if name in _RSPEC_BUILTIN_METHODS:
            continue
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args, ",") if a.strip()])
        method_calls.setdefault(name, set()).add(arity)

    # Build the final SUT-type set:
    #   - every `describe` target
    #   - every parent in `class X < Y`
    #   - every `Foo.new` receiver
    #   - every qualified constant (Foo::Bar)
    # MINUS test-file-internal const declarations
    sut_types: set[str] = set()
    sut_types.update(describe_targets)
    sut_types.update(parent_consts)
    sut_types.update(constructors.keys())
    sut_types.update(qualified_consts)
    sut_types.difference_update(test_defined_consts)
    # Also remove a few obvious Ruby-stdlib namespaces
    _STDLIB_NS = {
        "Object", "BasicObject", "Class", "Module", "Kernel", "String",
        "Integer", "Float", "Numeric", "Array", "Hash", "Symbol", "Proc",
        "Lambda", "Range", "Regexp", "MatchData", "Time", "Date", "DateTime",
        "Struct", "OpenStruct", "Set", "File", "Dir", "IO", "StringIO",
        "Tempfile", "Pathname", "URI", "Net", "JSON", "YAML", "CSV",
        "Logger", "Mutex", "Thread", "Process", "Signal", "Math",
        "Comparable", "Enumerable", "Enumerator", "Numeric", "Random",
        "ENV", "ARGV", "STDOUT", "STDERR", "STDIN", "Errno", "Exception",
        "StandardError", "RuntimeError", "ArgumentError", "TypeError",
        "NameError", "NoMethodError", "LoadError", "NotImplementedError",
        "ZeroDivisionError", "RangeError", "IndexError", "KeyError",
        "FrozenError", "IOError", "EOFError", "SystemCallError",
        "RSpec", "Minitest", "MiniTest", "Test", "Test::Unit",
        "RSpec::Mocks", "RSpec::Matchers", "RSpec::Expectations",
        "RSpec::Core", "MiniTest::Unit", "MiniTest::Spec",
        "Mocha", "FactoryBot", "FactoryGirl", "VCR", "WebMock",
        # ActiveSupport / ActiveRecord / Rails
        "ActiveSupport", "ActiveRecord", "ActionController", "ActionView",
        "ActionMailer", "ActiveModel", "ActiveJob", "Rails",
    }
    sut_types -= _STDLIB_NS
    # Also drop anything that is itself a namespace prefix of a stdlib type
    sut_types = {s for s in sut_types if s.split("::")[0] not in _STDLIB_NS}

    # Decide if parseable: at minimum we need EITHER a describe target,
    # a constructor call, a parent class, OR at least one 3rd-party require
    # (so the agent gets the gem list even if the structure is light).
    if (
        not describe_targets
        and not constructors
        and not parent_consts
        and not requires  # any require at all
        and not require_relatives
    ):
        return None

    # Classify requires
    req_gems_third_party: list[str] = []
    req_local_files: list[str] = []
    # Known 3rd-party gem heads — once `require 'foo/bar'` matches one of
    # these, we treat it as a 3rd-party gem (the gem will provide foo/bar).
    # Otherwise, multi-segment requires are treated as local project files
    # the agent must create under /app/.
    _KNOWN_GEM_HEADS = frozenset({
        "rspec", "minitest", "test", "mocha", "chefspec", "serverspec",
        "rspec-mocks", "rspec-core", "rspec-expectations",
        "google", "gapic", "grpc", "protobuf", "googleapis",
        "active_support", "active_record", "active_model", "activerecord",
        "actionpack", "actioncontroller", "actionview", "rails", "rspec-rails",
        "factory_bot", "factory_girl", "faker", "vcr", "webmock", "capybara",
        "selenium", "watir", "cucumber", "aruba",
        "sinatra", "rack", "tilt", "haml", "erubis", "slim",
        "json", "yaml", "csv", "nokogiri", "oj", "msgpack",
        "redis", "dalli", "memcached", "mongo", "mongoid", "pg", "mysql2",
        "sequel", "sqlite3",
        "logstash", "fluentd", "fluent",
        "openssl", "bcrypt", "jwt", "sass", "uglifier", "compass",
        "rake", "thor", "ruby-progressbar", "tty",
        "cocaine", "paperclip", "carrierwave", "shrine",
        "kpeg", "treetop", "racc", "rly",
        "flipper", "rollout", "pundit", "cancancan", "cancan",
        "concurrent", "concurrent-ruby", "sidekiq", "resque", "delayed_job",
        "puma", "unicorn", "thin", "passenger",
        "msf", "metasploit", "hpcloud", "fog", "aws", "aws-sdk",
        "chef", "knife", "berkshelf", "test-kitchen", "kitchen",
        "puppet", "moneta", "padrino",
        "octokit", "github", "gitlab",
        "rubocop", "reek", "flog", "flay", "metric_fu",
        "graphql", "graphiql", "absinthe",
        "io", "logger", "ruby", "ruby-debug",
        "dry", "dry-system", "dry-container", "rom",
        "hanami", "trailblazer", "shrine",
        "elasticsearch", "kafka", "ruby-kafka",
        "ratchet", "websocket", "faye", "eventmachine", "celluloid",
        "rspec-puppet", "berkshelf",
        "openstruct", "ostruct",
        "mail", "actionmailer", "premailer",
        "rmagick", "minimagick", "mini_magick",
        "geocoder", "ruby-saml", "saml",
        "i18n", "globalize",
        "rspec-its",
        "fast_jsonapi", "active_model_serializers",
        "cancan", "cancancan", "rolify",
        "devise", "doorkeeper", "warden",
        "cbor", "msgpack", "bson",
        "ddtrace", "datadog", "statsd-ruby", "statsd",
        "raven", "sentry-raven", "appsignal", "newrelic_rpm", "new_relic",
        "logstash-event", "logstash-input-beats",
        "stripe", "braintree", "paypal-sdk",
        "twilio", "sendgrid",
        "kubernetes", "kube_client", "kubeclient",
        "thrift", "avro",
        "kpeg", "polyglot",
        "cnab_rb", "cfn-nag", "banhammer",
        "hydramata", "htty", "grably",
        "new_relic", "user_agent",
        "workato", "workato-connector-sdk",
        "ansi", "tty-prompt", "tty-spinner",
        "config", "dry-configurable",
        "kpeg",
        "cardano_wallet", "cardanowallet",
        "shrine", "rmagick", "vips",
        "logstash", "fluentd",
        "kpeg",
        "gooddata",
    })
    for r in dict.fromkeys(requires):  # preserve order, dedupe
        # Strip trailing '.rb' if present (some tests use require 'foo.rb')
        norm = r[:-3] if r.endswith(".rb") else r
        had_rb_suffix = r.endswith(".rb")
        # Local file? It starts with './' or contains a path separator and
        # the head looks like a project name (not a known stdlib lib)
        if r.startswith("./") or r.startswith("../"):
            req_local_files.append(r)
            continue
        # Heuristic: spec helper-ish names → local files
        head = norm.split("/")[0]
        if head in _SPEC_HELPER_HINTS or norm in _SPEC_HELPER_HINTS:
            req_local_files.append(r)
            continue
        # Stdlib?
        if norm in _RUBY_STDLIB_GEMS or head in _RUBY_STDLIB_GEMS:
            continue
        # `.rb` suffix is a strong local-file signal (canonical Ruby idiom for
        # "load this specific file from the load path", typically used for
        # project files, not gems): `require 'bank_account.rb'`
        if had_rb_suffix and head not in _KNOWN_GEM_HEADS:
            req_local_files.append(r)
            continue
        # If multi-segment ('foo/bar/baz') and head NOT in known gems list →
        # treat as local project file the agent must create under /app/. This
        # covers cases like `require 'advent_of_code_2017/day2/spreadsheet_line'`
        # where `advent_of_code_2017` is the project, not a gem on rubygems.org.
        if "/" in norm and head not in _KNOWN_GEM_HEADS:
            req_local_files.append(r)
            continue
        req_gems_third_party.append(r)

    # require_relative is always a local file
    req_local_files.extend(dict.fromkeys(require_relatives))

    return {
        "describe_targets": list(dict.fromkeys(describe_targets)),
        "requires_third_party_gems": list(dict.fromkeys(req_gems_third_party)),
        "requires_local_files": list(dict.fromkeys(req_local_files)),
        "sut_types": sorted(sut_types),
        "constructors": {k: sorted(v) for k, v in constructors.items()},
        "method_calls": {k: sorted(v) for k, v in method_calls.items()},
        "test_defined_consts": sorted(test_defined_consts),
    }


# --------------------------------------------------------------------------- #
# Gem-name resolution for `gem install`
# --------------------------------------------------------------------------- #

def _gem_name_for_require(req: str) -> str | None:
    """
    Given a `require` string, return the top-level gem name to install
    via `gem install <name>`, or None if we can't determine one.

    Heuristic: take the path component before the first '/'. Some gems
    have a different gem name vs require name (e.g. `require 'rspec/core'`
    is part of the `rspec-core` gem) — we collapse the first two path
    components with a hyphen for the most common cases:
      'rspec/core'  -> 'rspec-core'
      'gapic/grpc/service_stub' -> 'gapic-common' (we just use the head)
    For simplicity we use the head component; this is good enough for
    `gem install || true` since failures are silently ignored.
    """
    if not req or req.startswith(".") or "/" in req and req.split("/")[0] in _SPEC_HELPER_HINTS:
        return None
    norm = req[:-3] if req.endswith(".rb") else req
    head = norm.split("/")[0]
    if not head or not re.match(r"^[a-z][a-z0-9_-]*$", head):
        return None
    if head in _RUBY_STDLIB_GEMS:
        return None
    return head


# --------------------------------------------------------------------------- #
# Instruction rewriter
# --------------------------------------------------------------------------- #

_PROMPT_CAP = 8000  # max chars in the rewritten instruction.md

_V2_MARKER = "<!-- laion v2 instruction patch: enriched with Ruby test contract -->"
_V2_TESTSH_MARKER = "# laion v2 test.sh patch: gem-install third-party requires"


def _format_test_contract(parsed: dict) -> str:
    lines: list[str] = []
    lines.append("## Test contract (auto-extracted from `tests/test_solution.rb`)")
    lines.append("")
    lines.append(
        "Below is a deterministic summary of what the test file expects. "
        "Use this as the source of truth for require-paths and public API. "
        "If the original task description below conflicts with this section, "
        "**prefer this section**."
    )
    lines.append("")
    lines.append("**Language**: Ruby")

    # Test framework hint
    fw = "RSpec / minitest"  # generic; specific framework not always detectable
    lines.append(f"**Test framework hint**: {fw} (test.sh runs `rspec` if `RSpec`/`describe` "
                 "is detected, otherwise plain `ruby`)")

    if parsed["describe_targets"]:
        lines.append("")
        lines.append("**Top-level types under test (from `describe`/`RSpec.describe`)**:")
        lines.append("")
        for d in parsed["describe_targets"][:15]:
            lines.append(f"- `{d}`")

    if parsed["requires_third_party_gems"]:
        lines.append("")
        lines.append(
            "**Required 3rd-party gems** (the test issues `require '<gem>'` — these are "
            "auto-installed via `gem install` in `tests/test.sh`, but if a gem fails to "
            "install the test will not even start):"
        )
        lines.append("")
        for g in parsed["requires_third_party_gems"][:25]:
            lines.append(f"- `require '{g}'`")
        more = len(parsed["requires_third_party_gems"]) - 25
        if more > 0:
            lines.append(f"- ... and {more} more")

    if parsed["requires_local_files"]:
        lines.append("")
        lines.append(
            "**Required local files** (the test issues `require '<file>'` or "
            "`require_relative ...`; these must exist somewhere on the load path — "
            "typically `/app/<file>.rb`. Common helper files like `spec_helper`, "
            "`test_helper`, `helper` may need an empty stub):"
        )
        lines.append("")
        for r in parsed["requires_local_files"][:20]:
            lines.append(f"- `{r}`")
        more = len(parsed["requires_local_files"]) - 20
        if more > 0:
            lines.append(f"- ... and {more} more")

    if parsed["sut_types"]:
        lines.append("")
        lines.append(
            "**Symbols the test references that you likely need to define** "
            "(class/module names not from Ruby/RSpec/Rails stdlib; place under `/app/`):"
        )
        lines.append("")
        sample = parsed["sut_types"][:30]
        more = len(parsed["sut_types"]) - len(sample)
        for s in sample:
            ctor = parsed["constructors"].get(s)
            if ctor is not None:
                ar_str = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in ctor)
                lines.append(f"- `{s}` — constructed via `{s}.new(...)` with arities: {ar_str}")
            else:
                lines.append(f"- `{s}`")
        if more > 0:
            lines.append(f"- ... and {more} more")

    methods = sorted(
        (n, a) for n, a in parsed["method_calls"].items()
        if not n.startswith("_")
    )
    if methods:
        lines.append("")
        lines.append(
            "**Methods invoked on instances/objects** (you must expose at least these "
            "with the indicated arity; return type is inferred from how the test uses "
            "the result):"
        )
        lines.append("")
        sample = methods[:25]
        more = len(methods) - len(sample)
        for name, arities in sample:
            ar_str = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in arities)
            lines.append(f"- `.{name}(...)` — called with: {ar_str}")
        if more > 0:
            lines.append(f"- ... and {more} more")

    lines.append("")
    lines.append("## Build & test environment")
    lines.append("")
    lines.append("- Base image: `ruby:3.2-slim` with `build-essential`, `ruby-dev`, "
                 "`libffi-dev` for native extensions.")
    lines.append("- Pre-installed gems: `rspec`, `minitest`. Other 3rd-party gems "
                 "(see list above) are best-effort `gem install`-ed by `tests/test.sh` "
                 "before running the test.")
    lines.append("- The grader (`tests/test.sh`) runs `bundle install` if `/app/Gemfile` "
                 "exists, then auto-requires every `*.rb` file under `/app/` (recursively) "
                 "via `$LOAD_PATH.unshift('/app')`. So you can place sources at any "
                 "depth under `/app/` as long as relative-require paths resolve.")
    lines.append("- For each `require 'foo/bar'` in the test, ensure either the `foo` "
                 "gem provides it OR you create `/app/foo/bar.rb` with the required "
                 "constants.")
    lines.append("")

    return "\n".join(lines)


def rewrite_instruction(original: str, parsed: dict) -> str:
    """Produce the new instruction.md content."""
    contract = _format_test_contract(parsed)

    title_hint = parsed["describe_targets"][0] if parsed["describe_targets"] else "Ruby task"
    header = (
        f"# {title_hint} — Ruby task\n\n"
        f"{_V2_MARKER}\n\n"
        "Implement Ruby sources under `/app/` so that the test file at "
        "`/tests/test_solution.rb` requires and runs cleanly under `rspec` "
        "(or plain `ruby` if the file uses Test::Unit / minitest).\n\n"
    )

    body = (
        header
        + contract
        + "\n## Original task description (LLM-generated; may be partial — defer to the contract above)\n\n"
        + original.strip()
        + "\n"
    )

    if len(body) > _PROMPT_CAP:
        keep = _PROMPT_CAP - len(header) - len(contract) - 200
        if keep < 500:
            keep = 500
        truncated_orig = original.strip()[:keep] + "\n\n[... description truncated to fit prompt budget ...]\n"
        body = (
            header
            + contract
            + "\n## Original task description (LLM-generated; may be partial — defer to the contract above)\n\n"
            + truncated_orig
        )

    return body


# --------------------------------------------------------------------------- #
# test.sh patcher (gem-install required gems before running the tests)
# --------------------------------------------------------------------------- #

def patch_test_sh(test_sh: str, gem_names: list[str]) -> str:
    """Insert `gem install <name> --no-document || true` lines for each gem before
    the `# Install dependencies if Gemfile exists` step. Idempotent via marker."""
    if _V2_TESTSH_MARKER in test_sh:
        return test_sh
    if not gem_names:
        return test_sh

    install_block_lines = [_V2_TESTSH_MARKER]
    for g in dict.fromkeys(gem_names):
        # quote-safe: gem names are restricted to [a-z0-9_-] by _gem_name_for_require
        install_block_lines.append(f"gem install {g} --no-document >/dev/null 2>&1 || true")
    install_block = "\n".join(install_block_lines) + "\n"

    # Insert just before the `# Install dependencies if Gemfile exists` line
    needle = "# Install dependencies if Gemfile exists"
    if needle in test_sh:
        return test_sh.replace(needle, install_block + "\n" + needle, 1)

    # Fallback: insert after `cd /app`
    cd_needle = "cd /app\n"
    if cd_needle in test_sh:
        return test_sh.replace(cd_needle, cd_needle + "\n" + install_block + "\n", 1)

    # Last-resort: prepend at the top after shebang
    if test_sh.startswith("#!"):
        first_nl = test_sh.find("\n")
        return test_sh[: first_nl + 1] + "\n" + install_block + test_sh[first_nl + 1 :]
    return install_block + "\n" + test_sh


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-test-sh", action="store_true",
                   help="Don't modify tests/test.sh; only rewrite instruction.md")
    p.add_argument("--drop-unparseable", action="store_true",
                   help="Delete task dir if test_solution.rb can't be parsed (default: leave untouched)")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "instruction.md").exists())
    if not task_dirs:
        print(f"No task dirs (with instruction.md) under {root}", file=sys.stderr)
        return 2

    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_changed = 0
    n_skipped_no_test = 0
    n_dropped_unparseable = 0
    n_already_patched = 0
    n_oversized = 0
    n_test_sh_patched = 0
    gem_freq: dict[str, int] = {}

    for i, d in enumerate(task_dirs, 1):
        test_path = d / "tests" / "test_solution.rb"
        if not test_path.is_file():
            n_skipped_no_test += 1
            continue
        try:
            test_src = test_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            continue
        parsed = parse_test_ruby(test_src)
        if parsed is None:
            n_dropped_unparseable += 1
            if args.drop_unparseable and not args.dry_run:
                shutil.rmtree(d)
            continue

        for g in parsed["requires_third_party_gems"]:
            gem_freq[g] = gem_freq.get(g, 0) + 1

        instr_path = d / "instruction.md"
        original = instr_path.read_text(encoding="utf-8", errors="replace")
        if _V2_MARKER in original:
            n_already_patched += 1
            continue

        new_text = rewrite_instruction(original, parsed)
        if len(new_text) >= _PROMPT_CAP:
            n_oversized += 1

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(new_text, encoding="utf-8")

        # Patch test.sh
        if not args.skip_test_sh:
            test_sh_path = d / "tests" / "test.sh"
            if test_sh_path.is_file():
                test_sh_src = test_sh_path.read_text(encoding="utf-8", errors="replace")
                gem_install_names: list[str] = []
                for r in parsed["requires_third_party_gems"]:
                    name = _gem_name_for_require(r)
                    if name:
                        gem_install_names.append(name)
                if gem_install_names and _V2_TESTSH_MARKER not in test_sh_src:
                    new_sh = patch_test_sh(test_sh_src, gem_install_names)
                    if new_sh != test_sh_src:
                        if not args.dry_run:
                            test_sh_path.write_text(new_sh, encoding="utf-8")
                        n_test_sh_patched += 1

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"skipped_no_test={n_skipped_no_test} "
                f"dropped_unparseable={n_dropped_unparseable} "
                f"already_patched={n_already_patched} "
                f"oversized={n_oversized} "
                f"test_sh_patched={n_test_sh_patched}",
                flush=True,
            )

    # Top 10 most-required gems (for reporting)
    top_gems = sorted(gem_freq.items(), key=lambda kv: -kv[1])[:10]

    print(
        f"\nDone. {n_changed}/{n_total} instruction.md files modified "
        f"(dry_run={args.dry_run}).\n"
        f"  skipped_no_test         = {n_skipped_no_test}\n"
        f"  dropped_unparseable     = {n_dropped_unparseable} "
        f"({'deleted' if args.drop_unparseable else 'left untouched'})\n"
        f"  already_patched_skip    = {n_already_patched}\n"
        f"  oversized_capped        = {n_oversized}\n"
        f"  test_sh_patched         = {n_test_sh_patched}\n"
        f"\nTop 10 gems referenced in tests:\n"
        + "\n".join(f"  {n:5d}  {g}" for g, n in top_gems)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
