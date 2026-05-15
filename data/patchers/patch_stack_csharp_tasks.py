#!/usr/bin/env python3
"""
exp_rpt_stack-csharp v2 patcher.

Bug (v1 → v2): 10/10 sampled trials reward=0 because the verifier creates a
fresh `dotnet new xunit` project, drops `tests/TestSolution.cs` into it, and
runs `dotnet test`. The freshly-templated `.csproj` has only `xunit`,
`xunit.runner.visualstudio`, and `Microsoft.NET.Test.Sdk` as NuGet refs, so
real-world tests that `using NUnit.Framework`, `using Moq`,
`using Newtonsoft.Json`, `using Microsoft.EntityFrameworkCore`, etc. fail
with CS0234/CS0246 ("type or namespace not found") at compile time. The
agent has no way to know which NuGet packages are required because
`instruction.md` doesn't enumerate them.

Fix (v2): For each task, mechanically parse `tests/TestSolution.cs` to
extract:
  - The full `using <NS>;` import list (top-level + dotted).
  - The namespace declaration (`namespace Foo.Bar { ... }`).
  - Top-level identifiers / constructor calls that signal needed types.

Then:
  1) **Modify `tests/test.sh`** to inject `dotnet add package <pkg>` lines
     after `dotnet new xunit` and before `dotnet test`. Mapping is from a
     curated namespace→NuGet-package table covering the top of the corpus
     (which dominates the long tail). Unknown namespaces are skipped silently;
     they are most often project-internal (the SUT the agent must build).
  2) **Modify `instruction.md`** to surface the using list, the test
     framework, and the namespace declaration so the agent has a
     deterministic ground-truth contract to satisfy.
  3) **Drop tasks** that depend on packages we know are NOT recoverable on
     a stock .NET 8 SDK (e.g. `FastTests`, `RavenTestBase`, `System.Web.UI`
     ASP.NET Classic, `UnityEngine` runtime, internal Roslyn test infra,
     etc.). These can never compile in the verifier sandbox regardless of
     what the agent does.
  4) **Idempotent** via markers:
       - `<!-- laion v2 instruction patch: enriched with C# test contract -->`
         in instruction.md
       - `# --- laion v2 patch: dotnet add package ---`
         in test.sh

Usage:
  python data/patchers/patch_stack_csharp_tasks.py \
      --root <dir> [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Markers (idempotency)
# --------------------------------------------------------------------------- #

_INSTRUCTION_MARKER = (
    "<!-- laion v2 instruction patch: enriched with C# test contract -->"
)
_TESTSH_MARKER = "# --- laion v2 patch: dotnet add package ---"
_TESTSH_END_MARKER = "# --- end laion v2 patch ---"

_PROMPT_CAP = 8000

# --------------------------------------------------------------------------- #
# Namespace -> NuGet package mapping
#
# Strategy: longest-prefix match. We list the package name and (optionally) a
# version pin. Where a top-level namespace is "owned" by a single big package
# (Newtonsoft, Moq, NUnit, AutoMapper, ...), we map the top-level alone.
# Where a namespace is part of the .NET BCL (System.*, Microsoft.NET.*) we
# map nothing — those come for free with the SDK.
#
# When in doubt, prefer the more specific mapping; a single `dotnet add
# package` of a meta-package will pull transitive deps via NuGet's resolver.
# --------------------------------------------------------------------------- #

# Namespaces that are part of the .NET 8 SDK / BCL — never add a package for these
_BCL_PREFIXES = (
    "System",
    "Microsoft.CSharp",
    "Microsoft.Win32",
    "Microsoft.VisualBasic",
)

# Test-framework namespaces — these are *the* test runner. xUnit comes
# pre-installed via `dotnet new xunit`; NUnit / MSTest need explicit refs.
_TEST_FRAMEWORKS = {
    "Xunit": None,                                 # already present from `dotnet new xunit`
    "Xunit.Abstractions": None,                    # already present
    "NUnit": "NUnit",                              # also need NUnit3TestAdapter
    "NUnit.Framework": "NUnit",
    "Microsoft.VisualStudio.TestTools.UnitTesting": "MSTest.TestFramework",
}

# Project-internal / unrecoverable prefixes — drop tasks that depend on these.
# These are RavenDB internal test base classes, Roslyn internal test
# utilities, Unity runtime, ASP.NET Classic (System.Web full framework), etc.
# A stock .NET 8 SDK + NuGet cannot resolve them.
_DROP_PREFIXES = (
    "FastTests",
    "FastTests.Smuggler",
    "FastTests.Server",
    "FastTests.Voron",
    "RavenTestBase",
    "SlowTests",
    "Tests.Infrastructure",
    "Roslyn.Test.Utilities",
    "Microsoft.CodeAnalysis.Test.Utilities",
    "Microsoft.CodeAnalysis.CSharp.Test.Utilities",
    "Microsoft.CodeAnalysis.VisualBasic.Test.Utilities",
    "Microsoft.CodeAnalysis.UnitTests",
    "UnityEngine",
    "UnityEngine.TestTools",
    "UnityEditor",
    "Unity.",
    "System.Web.UI",                # ASP.NET classic — needs full framework
    "System.Web.Mvc",               # ASP.NET classic
    "System.Web.Http",              # ASP.NET classic
    "System.Web.Razor",
    "System.Web.WebPages",
    "ManagedTests",                 # Internal Microsoft test infra
    "Internal.Cryptography",        # Internal corefx test
    "ScriptCs",                     # Discontinued
    "Microsoft.AspNet.",            # ASP.NET classic (vs AspNetCore)
)

# Explicit namespace -> NuGet mapping (longest-prefix match).
# Sorted long-first below at module load.
_NAMESPACE_TO_PACKAGE: dict[str, str] = {
    # --- Test frameworks (handled specially above, but listed for clarity) ---
    "NUnit": "NUnit",
    "Xunit": "xunit",  # already in template; harmless if re-added
    # --- JSON / serialization ---
    "Newtonsoft.Json": "Newtonsoft.Json",
    "System.Text.Json": "System.Text.Json",
    "ProtoBuf": "protobuf-net",
    "MessagePack": "MessagePack",
    "YamlDotNet": "YamlDotNet",
    # --- Mocking / assertion ---
    "Moq": "Moq",
    "NSubstitute": "NSubstitute",
    "FakeItEasy": "FakeItEasy",
    "FluentAssertions": "FluentAssertions",
    "Shouldly": "Shouldly",
    "AutoFixture": "AutoFixture",
    # --- Mapping / DI ---
    "AutoMapper": "AutoMapper",
    "Autofac": "Autofac",
    # --- Microsoft Extensions ---
    "Microsoft.EntityFrameworkCore": "Microsoft.EntityFrameworkCore",
    "Microsoft.Extensions.DependencyInjection": "Microsoft.Extensions.DependencyInjection",
    "Microsoft.Extensions.Logging": "Microsoft.Extensions.Logging",
    "Microsoft.Extensions.Logging.Abstractions": "Microsoft.Extensions.Logging.Abstractions",
    "Microsoft.Extensions.Configuration": "Microsoft.Extensions.Configuration",
    "Microsoft.Extensions.Configuration.Json": "Microsoft.Extensions.Configuration.Json",
    "Microsoft.Extensions.Configuration.Abstractions": "Microsoft.Extensions.Configuration.Abstractions",
    "Microsoft.Extensions.Options": "Microsoft.Extensions.Options",
    "Microsoft.Extensions.Hosting": "Microsoft.Extensions.Hosting",
    "Microsoft.Extensions.Caching.Memory": "Microsoft.Extensions.Caching.Memory",
    "Microsoft.Extensions.Http": "Microsoft.Extensions.Http",
    "Microsoft.Extensions.FileProviders": "Microsoft.Extensions.FileProviders",
    "Microsoft.Extensions.Primitives": "Microsoft.Extensions.Primitives",
    # --- ASP.NET Core (NOT classic ASP.NET) ---
    "Microsoft.AspNetCore.Mvc": "Microsoft.AspNetCore.Mvc.Core",
    "Microsoft.AspNetCore.Mvc.Testing": "Microsoft.AspNetCore.Mvc.Testing",
    "Microsoft.AspNetCore.TestHost": "Microsoft.AspNetCore.TestHost",
    "Microsoft.AspNetCore.Hosting": "Microsoft.AspNetCore.Hosting",
    "Microsoft.AspNetCore.Http": "Microsoft.AspNetCore.Http.Abstractions",
    "Microsoft.AspNetCore.Routing": "Microsoft.AspNetCore.Routing",
    "Microsoft.AspNetCore.Components": "Microsoft.AspNetCore.Components",
    "Microsoft.AspNetCore.Components.Web": "Microsoft.AspNetCore.Components.Web",
    "Microsoft.AspNetCore.SignalR": "Microsoft.AspNetCore.SignalR",
    "Microsoft.AspNetCore.SignalR.Client": "Microsoft.AspNetCore.SignalR.Client",
    "Microsoft.AspNetCore.Identity": "Microsoft.AspNetCore.Identity",
    "Microsoft.AspNetCore.Authentication": "Microsoft.AspNetCore.Authentication",
    "Microsoft.AspNetCore.Authorization": "Microsoft.AspNetCore.Authorization",
    "Microsoft.AspNetCore.Cors": "Microsoft.AspNetCore.Cors",
    "Microsoft.AspNetCore.WebUtilities": "Microsoft.AspNetCore.WebUtilities",
    "Microsoft.AspNetCore.Server.Kestrel": "Microsoft.AspNetCore.Server.Kestrel.Core",
    # --- Roslyn (for non-test-utilities subnamespaces) ---
    "Microsoft.CodeAnalysis": "Microsoft.CodeAnalysis",
    "Microsoft.CodeAnalysis.CSharp": "Microsoft.CodeAnalysis.CSharp",
    "Microsoft.CodeAnalysis.CSharp.Syntax": "Microsoft.CodeAnalysis.CSharp",
    "Microsoft.CodeAnalysis.VisualBasic": "Microsoft.CodeAnalysis.VisualBasic",
    "Microsoft.CodeAnalysis.Workspaces": "Microsoft.CodeAnalysis.Workspaces.Common",
    # --- Logging / metrics ---
    "Serilog": "Serilog",
    "NLog": "NLog",
    "log4net": "log4net",
    "Microsoft.Extensions.Logging.Console": "Microsoft.Extensions.Logging.Console",
    # --- HTTP / Networking ---
    "RestSharp": "RestSharp",
    "Refit": "Refit",
    "Polly": "Polly",
    "Flurl": "Flurl.Http",
    "Flurl.Http": "Flurl.Http",
    # --- DB / ORM ---
    "Dapper": "Dapper",
    "ServiceStack.OrmLite": "ServiceStack.OrmLite",
    "MongoDB.Driver": "MongoDB.Driver",
    "MongoDB.Bson": "MongoDB.Bson",
    "StackExchange.Redis": "StackExchange.Redis",
    "MySql.Data": "MySql.Data",
    "Npgsql": "Npgsql",
    "System.Data.SqlClient": "System.Data.SqlClient",
    # --- Cloud SDKs ---
    "Azure.Core": "Azure.Core",
    "Azure.Storage.Blobs": "Azure.Storage.Blobs",
    "Azure.Identity": "Azure.Identity",
    "Microsoft.Azure.Cosmos": "Microsoft.Azure.Cosmos",
    "Amazon.S3": "AWSSDK.S3",
    "Amazon.DynamoDBv2": "AWSSDK.DynamoDBv2",
    "Google.Protobuf": "Google.Protobuf",
    "Google.Apis": "Google.Apis",
    "Google.Cloud.Storage.V1": "Google.Cloud.Storage.V1",
    # --- Image / media ---
    "SixLabors.ImageSharp": "SixLabors.ImageSharp",
    "ImageMagick": "Magick.NET-Q16-AnyCPU",
    "SkiaSharp": "SkiaSharp",
    # --- Misc useful ---
    "CommandLine": "CommandLineParser",
    "Humanizer": "Humanizer.Core",
    "MediatR": "MediatR",
    "Quartz": "Quartz",
    "Hangfire": "Hangfire.Core",
    "FluentValidation": "FluentValidation",
    "OpenTelemetry": "OpenTelemetry",
    "Microsoft.Bcl.AsyncInterfaces": "Microsoft.Bcl.AsyncInterfaces",
    "K4os.Compression.LZ4": "K4os.Compression.LZ4",
    "DotNetty": "DotNetty.Common",
    "ICSharpCode.SharpZipLib": "SharpZipLib",
    "Mono.Cecil": "Mono.Cecil",
    "Lucene.Net": "Lucene.Net",
    "ServiceStack": "ServiceStack",
    "ServiceStack.Text": "ServiceStack.Text",
    "Apache.NMS": "Apache.NMS",
    "Apache.Avro": "Apache.Avro",
    # --- Linux / interop ---
    "TerraFX.Interop.Windows": None,  # Windows-only; can't add as a package on Linux test runner safely
}

# Materialise as a sorted-by-length list so longest prefix wins.
_NS_MAP_SORTED: list[tuple[str, str | None]] = sorted(
    _NAMESPACE_TO_PACKAGE.items(), key=lambda kv: -len(kv[0])
)


# --------------------------------------------------------------------------- #
# C# parsing helpers
# --------------------------------------------------------------------------- #

# Strip BOM, block comments, and string literals before parsing.
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_VERBATIM_STRING_RE = re.compile(r'@"(?:""|[^"])*"', re.DOTALL)

# Matches:  using [static] Foo.Bar.Baz [= alias];
# We don't capture aliases on the RHS (e.g. `using IO = System.IO;`) because the
# RHS is what we want to extract for package mapping in the alias case, but the
# template form `using NS;` is what we care about for almost all real cases.
_USING_RE = re.compile(
    r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)

# `using Foo = System.IO;`  — we want the RHS for mapping.
_USING_ALIAS_RE = re.compile(
    r"^\s*using\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)

# `namespace Foo.Bar { ... }` or file-scoped `namespace Foo.Bar;`
_NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+([A-Za-z_][\w\.]*)\s*[{;]",
    re.MULTILINE,
)

# Test-class declaration heuristic: `class FooTests` / `class FooTest` /
# `public class Foo : Bar` — we just want the first class name to display.
_CLASS_DECL_RE = re.compile(
    r"\b(?:public\s+|internal\s+|sealed\s+|abstract\s+|static\s+|partial\s+)*class\s+(\w+)",
)


def _strip_comments_strings(src: str) -> str:
    """Remove comments and string literals so identifier scans don't pick them up."""
    src = src.lstrip("﻿")
    src = _BLOCK_COMMENT_RE.sub(" ", src)
    src = _LINE_COMMENT_RE.sub(" ", src)
    src = _VERBATIM_STRING_RE.sub('""', src)
    src = _STRING_RE.sub('""', src)
    return src


def parse_test_cs(src: str) -> dict | None:
    """Parse a TestSolution.cs; return None if essential structure is missing."""
    cleaned = _strip_comments_strings(src)

    usings: list[str] = []
    for m in _USING_RE.finditer(cleaned):
        ns = m.group(1)
        usings.append(ns)
    # alias usings — keep RHS (the actual namespace/type referenced)
    for m in _USING_ALIAS_RE.finditer(cleaned):
        rhs = m.group(2)
        # exclude top-level System aliases since they're BCL
        usings.append(rhs)

    if not usings:
        # No `using` directives: fully qualified file or empty test. Still
        # parseable but we can't add any packages.
        usings = []

    ns_match = _NAMESPACE_RE.search(cleaned)
    namespace = ns_match.group(1) if ns_match else None

    cls_match = _CLASS_DECL_RE.search(cleaned)
    test_class = cls_match.group(1) if cls_match else None

    return {
        "usings": list(dict.fromkeys(usings)),  # de-dup, preserve order
        "namespace": namespace,
        "test_class": test_class,
    }


# --------------------------------------------------------------------------- #
# Mapping logic
# --------------------------------------------------------------------------- #


def _is_bcl(ns: str) -> bool:
    """Is `ns` a .NET 8 BCL / SDK namespace (no package needed)?"""
    return any(ns == p or ns.startswith(p + ".") for p in _BCL_PREFIXES)


def _is_drop(ns: str) -> bool:
    """Is `ns` a known-unrecoverable namespace (drop the task)?"""
    return any(ns == p or ns.startswith(p.rstrip(".") + ".") or ns.startswith(p) and p.endswith(".")
               for p in _DROP_PREFIXES)


def map_using_to_package(ns: str) -> str | None:
    """Map a using namespace to a NuGet package, or None if unknown / no-op.

    Returns the package name. Returns None if:
      - The namespace is BCL (no package needed)
      - The namespace is unmapped (likely project-internal SUT — agent must build it)
      - The namespace is in our explicit "no package" allowlist (e.g. Xunit
        already in the template)
    """
    if _is_bcl(ns):
        return None
    # Test-framework table takes precedence (they have None values for stuff
    # already in the template).
    if ns in _TEST_FRAMEWORKS:
        return _TEST_FRAMEWORKS[ns]
    # Longest-prefix match against the curated table.
    for prefix, pkg in _NS_MAP_SORTED:
        if ns == prefix or ns.startswith(prefix + "."):
            return pkg
    # Top-level fallback: if the entire namespace is one well-known package
    # name (e.g. `Moq`, `NSubstitute`), the explicit table above already
    # caught it, so this fallback is intentionally *not* greedy. We don't add
    # a package for unknown namespaces — they're typically project-internal.
    return None


def packages_for_usings(usings: list[str]) -> tuple[list[str], list[str], bool]:
    """For a list of `using NS;` directives, return:
        (packages, dropped_namespaces, should_drop_task)

    `packages` is a deduped list of NuGet package names to add via
    `dotnet add package`.

    `dropped_namespaces` is the list of namespaces that match a drop-prefix —
    if non-empty, the task should be skipped entirely.
    """
    pkgs: list[str] = []
    dropped: list[str] = []
    seen_pkg: set[str] = set()

    has_nunit_framework = False
    has_mstest = False

    for ns in usings:
        if _is_drop(ns):
            dropped.append(ns)
            continue
        pkg = map_using_to_package(ns)
        if pkg and pkg not in seen_pkg:
            pkgs.append(pkg)
            seen_pkg.add(pkg)
        if ns == "NUnit.Framework" or ns.startswith("NUnit.Framework."):
            has_nunit_framework = True
        if ns == "Microsoft.VisualStudio.TestTools.UnitTesting":
            has_mstest = True

    # If the test uses NUnit, we additionally need the NUnit3 test adapter
    # (xUnit's `dotnet test` runner needs an adapter to discover NUnit
    # tests; otherwise tests are silently ignored).
    if has_nunit_framework:
        if "NUnit3TestAdapter" not in seen_pkg:
            pkgs.append("NUnit3TestAdapter")
            seen_pkg.add("NUnit3TestAdapter")

    # If the test uses MSTest, we also need MSTest.TestAdapter.
    if has_mstest:
        if "MSTest.TestAdapter" not in seen_pkg:
            pkgs.append("MSTest.TestAdapter")
            seen_pkg.add("MSTest.TestAdapter")

    return pkgs, dropped, bool(dropped)


# --------------------------------------------------------------------------- #
# test.sh patcher
# --------------------------------------------------------------------------- #


def patch_test_sh(original: str, packages: list[str]) -> str | None:
    """Inject `dotnet add package` lines into the original test.sh.

    Returns the new contents, or None if the file is already patched (i.e.
    contains `_TESTSH_MARKER`) or doesn't match the expected template.

    Strategy: locate the line `dotnet new xunit -n TestProject` and insert
    `cd TestProject && dotnet add package <pkg> ...` AFTER `cp /tests/...`
    but BEFORE `cd /app` resumes. Specifically, we insert the patch block
    inside the `if [ ! -f *.csproj ]; then ... fi` so it only runs once.
    """
    if _TESTSH_MARKER in original:
        return None

    # Sanity check: the template we expect.
    if "dotnet new xunit" not in original or "cp /tests/TestSolution.cs" not in original:
        # Unfamiliar test.sh — bail.
        return None

    # We append the patch block right after the `cp /tests/TestSolution.cs .`
    # line (which follows `cd TestProject`). At that point cwd is
    # `/app/TestProject` and we have a fresh xunit project — perfect place
    # to `dotnet add package`.
    add_lines: list[str] = []
    add_lines.append("")
    add_lines.append(f"    {_TESTSH_MARKER}")
    add_lines.append("    # NuGet packages required by tests/TestSolution.cs (auto-added by laion v2 patch).")
    add_lines.append("    # If your sources reference additional packages, add them here too.")
    if not packages:
        add_lines.append("    # (no extra packages required for this task)")
    else:
        for pkg in packages:
            add_lines.append(f'    dotnet add package {pkg} >/dev/null 2>&1 || dotnet add package {pkg}')
    add_lines.append("    # Restore so the new refs land in obj/project.assets.json before build/test.")
    add_lines.append("    dotnet restore >/dev/null 2>&1 || true")
    add_lines.append(f"    {_TESTSH_END_MARKER}")
    add_lines.append("")

    insertion = "\n".join(add_lines)

    # Insert after the `cp /tests/TestSolution.cs .` line.
    pattern = re.compile(r"(cp /tests/TestSolution\.cs\s*\.\s*\n)")
    if not pattern.search(original):
        return None
    new = pattern.sub(r"\1" + insertion + "\n", original, count=1)

    return new


# --------------------------------------------------------------------------- #
# instruction.md rewriter
# --------------------------------------------------------------------------- #


def _format_test_contract(parsed: dict, packages: list[str]) -> str:
    """Produce the markdown 'Test contract' section appended to instruction.md."""
    lines: list[str] = []
    lines.append("## Test contract (auto-extracted from `tests/TestSolution.cs`)")
    lines.append("")
    lines.append(
        "Below is a deterministic summary of what the C# test file expects. "
        "Use this as the source of truth for namespace, using directives, and required packages. "
        "If the original task description below conflicts with this section, prefer this section."
    )
    lines.append("")
    lines.append("**Language**: C# (.NET 8 SDK)")
    lines.append("**Test framework**: xUnit / NUnit / MSTest as indicated by the `using` list below; "
                 "the verifier creates a `dotnet new xunit` project and runs `dotnet test`. "
                 "If the test uses NUnit or MSTest, the verifier auto-adds the matching adapter so tests are discovered.")
    if parsed.get("namespace"):
        lines.append(f"**Test namespace**: `{parsed['namespace']}` — your sources may live in any namespace, "
                     "but the types referenced by the test must be reachable (either in the same namespace or via `using` in your source).")
    if parsed.get("test_class"):
        lines.append(f"**Test class**: `{parsed['test_class']}`")
    lines.append("")

    if parsed["usings"]:
        sample = parsed["usings"][:40]
        more = len(parsed["usings"]) - len(sample)
        lines.append("**`using` directives in the test file** (you must make every symbol referenced from these namespaces reachable):")
        lines.append("")
        for u in sample:
            lines.append(f"- `using {u};`")
        if more > 0:
            lines.append(f"- ... and {more} more")
        lines.append("")

    if packages:
        lines.append("**NuGet packages auto-added by the verifier** (you do NOT need to add them yourself; they will be on the classpath when `dotnet test` runs):")
        lines.append("")
        for p in packages:
            lines.append(f"- `{p}`")
        lines.append("")
        lines.append(
            "If the test references symbols from a namespace that is NOT in this list and NOT a "
            "BCL `System.*` namespace, that namespace is project-internal — you must define those "
            "types yourself under `/app/` so the test compiles and links."
        )
        lines.append("")
    else:
        lines.append("**NuGet packages**: none beyond the xUnit template defaults. "
                     "Any non-BCL types referenced by the test are project-internal — define them under `/app/`.")
        lines.append("")

    lines.append("## Build & test environment")
    lines.append("")
    lines.append(
        "- The verifier creates a fresh `dotnet new xunit -n TestProject` under `/app/TestProject/`, "
        "copies `tests/TestSolution.cs` into it, then runs `dotnet add package` for each package "
        "listed above (auto-injected). Finally it runs `dotnet test` and scores PASS iff exit code 0."
    )
    lines.append(
        "- Place your source `.cs` files under `/app/TestProject/` (or anywhere reachable to "
        "`dotnet test` from `/app/`). Files under `/app/TestProject/*.cs` are compiled together "
        "with `TestSolution.cs` automatically by the SDK's default globbing."
    )
    lines.append(
        "- You DO NOT need to edit `TestProject.csproj` manually — `dotnet add package` already "
        "handles NuGet refs, and the SDK auto-includes all `*.cs` files in the project directory."
    )
    lines.append(
        "- If you need additional packages beyond those listed above, add them with "
        "`dotnet add package <Name>` from inside `/app/TestProject/`."
    )
    lines.append("")

    return "\n".join(lines)


def rewrite_instruction(original: str, parsed: dict, packages: list[str]) -> str:
    """Produce the new instruction.md content."""
    contract = _format_test_contract(parsed, packages)
    title_class = parsed.get("test_class") or "C#"
    title_class = title_class.replace("Tests", "").replace("Test", "") or "C#"

    header = (
        f"# {title_class} - C# task\n\n"
        f"{_INSTRUCTION_MARKER}\n\n"
        "Implement C# sources under `/app/TestProject/` so that the test file at "
        "`tests/TestSolution.cs` (which the verifier copies into the same project) "
        "compiles and passes via `dotnet test`.\n\n"
    )

    body = (
        header
        + contract
        + "\n## Original task description (LLM-generated; may be partial - defer to the contract above)\n\n"
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
            + "\n## Original task description (LLM-generated; may be partial - defer to the contract above)\n\n"
            + truncated_orig
        )

    return body


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
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
    n_skipped_unparseable = 0
    n_already_patched = 0
    n_dropped_unrecoverable = 0
    n_no_packages_needed = 0
    n_oversized = 0
    n_testsh_unfamiliar = 0

    # Track top-level namespace counts in the kept set for the report.
    top_ns: dict[str, int] = {}

    for i, d in enumerate(task_dirs, 1):
        test_path = d / "tests" / "TestSolution.cs"
        testsh_path = d / "tests" / "test.sh"
        if not test_path.is_file() or not testsh_path.is_file():
            n_skipped_no_test += 1
            continue
        try:
            test_src = test_path.read_text(encoding="utf-8", errors="replace")
            testsh_src = testsh_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            continue
        parsed = parse_test_cs(test_src)
        if parsed is None:
            n_skipped_unparseable += 1
            continue

        instr_path = d / "instruction.md"
        original_instr = instr_path.read_text(encoding="utf-8", errors="replace")
        if _INSTRUCTION_MARKER in original_instr and _TESTSH_MARKER in testsh_src:
            n_already_patched += 1
            continue

        packages, dropped, should_drop = packages_for_usings(parsed["usings"])
        if should_drop:
            n_dropped_unrecoverable += 1
            continue

        new_testsh = patch_test_sh(testsh_src, packages)
        if new_testsh is None and _TESTSH_MARKER not in testsh_src:
            n_testsh_unfamiliar += 1
            continue
        # If new_testsh is None due to already-patched test.sh but unpatched
        # instruction, we still rewrite instruction below (defensive).
        # If new_testsh is None due to unfamiliar template, we already
        # incremented and continued.

        new_instr = rewrite_instruction(original_instr, parsed, packages)
        if len(new_instr) >= _PROMPT_CAP:
            n_oversized += 1

        if not packages:
            n_no_packages_needed += 1

        for u in parsed["usings"]:
            top = u.split(".")[0]
            top_ns[top] = top_ns.get(top, 0) + 1

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(new_instr, encoding="utf-8")
            if new_testsh is not None:
                testsh_path.write_text(new_testsh, encoding="utf-8")

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"dropped_unrecoverable={n_dropped_unrecoverable} "
                f"skipped_no_test={n_skipped_no_test} "
                f"skipped_unparseable={n_skipped_unparseable} "
                f"already_patched={n_already_patched} "
                f"testsh_unfamiliar={n_testsh_unfamiliar} "
                f"no_pkgs={n_no_packages_needed} "
                f"oversized={n_oversized}",
                flush=True,
            )

    print(
        f"\nDone. {n_changed}/{n_total} task dirs modified (dry_run={args.dry_run}).\n"
        f"  dropped_unrecoverable = {n_dropped_unrecoverable}\n"
        f"  skipped_no_test       = {n_skipped_no_test}\n"
        f"  skipped_unparseable   = {n_skipped_unparseable}\n"
        f"  already_patched_skip  = {n_already_patched}\n"
        f"  testsh_unfamiliar     = {n_testsh_unfamiliar}\n"
        f"  no_pkgs_needed        = {n_no_packages_needed}\n"
        f"  oversized_capped      = {n_oversized}\n"
    )
    print("Top-level namespaces in kept tasks:")
    for ns, c in sorted(top_ns.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {c:>5}  {ns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
