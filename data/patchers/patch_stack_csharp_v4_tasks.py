#!/usr/bin/env python3
"""
exp_rpt_stack-csharp v4 patcher.

Bug (v3 -> v4): v3 validation showed 6/200 (3%) solved, 194 fails. Failure-mode
breakdown:
  - 64 (33%) CS0234 namespace does not exist -- real NuGet packages NOT in
    v2's known-package allowlist (Microsoft.Extensions.DependencyInjection,
    MathNet.Numerics, Akka.*, AutoMapper, Polly, MediatR, AspNetCore family,
    Microsoft.Azure.Services.AppAuthentication, Microsoft.Azure.Cosmos,
    Microsoft.EntityFrameworkCore, FluentAssertions, Moq, Serilog, NUnit, ...).
  - 40 (21%) CS0246 type not found -- project-internal types the model didn't
    write (already addressed by v3's project-internal enrichment; not v4's job).
  - 45 (23%) CS0117/1061/1503 method/API signature mismatch -- model wrote a
    class but with wrong methods (also a v3-style issue; not v4's job).
  -  5 (3%)  CS0227 unsafe code without `/unsafe` flag -- the test.sh's
    `dotnet new xunit` template doesn't enable AllowUnsafeBlocks; if the test
    or solution uses `unsafe { }` the compile fails.
  - 40 (21%) other.

Fix (v4): three SEPARATE, conservative, idempotent fixes layered on top of
the v2 + v3 patches. v4 is NOT a destructive filter; it MUTATES survivors but
never drops tasks.

  1. **Expand v2's NuGet allowlist** (`_NAMESPACE_TO_PACKAGE` in
     `patch_stack_csharp_tasks.py`). For each task we re-parse
     `tests/TestSolution.cs`, recompute the package list using the EXPANDED
     map, and append a `dotnet add package` block to `tests/test.sh`.
     We do NOT rewrite the v2 block (preserves idempotency / debugability);
     we APPEND a v4 block AFTER v2's `# --- end laion v2 patch ---` marker.
     Packages already injected by v2 are filtered out so we don't double-add.

  2. **Enable unsafe code** in the freshly-templated TestProject. After the
     `dotnet new xunit -n TestProject` line, inject a one-line sed that
     adds `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>` to the FIRST
     `<PropertyGroup>` in `TestProject.csproj`. Runs inside the same
     `if [ ! -f *.csproj ]` initialisation guard as v2, so it only
     executes once per fresh project.

  3. **Idempotency markers** (separate from v2/v3) so v4 is safe to re-run:
       - `# --- laion v4 patch: extra dotnet add package ---`
       - `# --- end laion v4 patch ---`
       - `# --- laion v4 patch: enable unsafe code ---`

Constraints:
  - DO NOT modify `tests/TestSolution.cs` -- that's the verifier source of truth.
  - DO NOT mutate or remove v2/v3 markers in test.sh / instruction.md.
  - DO NOT drop tasks. v4 mutates everyone or no-ops; survivor counts stay flat.
  - Conservative: only NuGet packages whose names we are confident in are added.
    A wrong package name causes `dotnet restore` to fail and HIDES real errors.

Usage:
  python data/patchers/patch_stack_csharp_v4_tasks.py \
      --root /path/to/exp_rpt_stack-csharp-v4 \
      [--dry-run] [--limit N] [--drop-log path.tsv]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Markers (idempotency)
# --------------------------------------------------------------------------- #

_V2_TESTSH_MARKER = "# --- laion v2 patch: dotnet add package ---"
_V2_TESTSH_END_MARKER = "# --- end laion v2 patch ---"

_V4_TESTSH_PKG_MARKER = "# --- laion v4 patch: extra dotnet add package ---"
_V4_TESTSH_PKG_END_MARKER = "# --- end laion v4 patch ---"
_V4_TESTSH_UNSAFE_MARKER = "# --- laion v4 patch: enable unsafe code ---"

# --------------------------------------------------------------------------- #
# C# parsing helpers (mirror v2's, kept self-contained so v4 doesn't import
# from sibling patchers and entangle module load order).
# --------------------------------------------------------------------------- #

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_VERBATIM_STRING_RE = re.compile(r'@"(?:""|[^"])*"', re.DOTALL)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

_USING_RE = re.compile(
    r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)
_USING_ALIAS_RE = re.compile(
    r"^\s*using\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)


def _strip_comments_strings(src: str) -> str:
    src = src.lstrip("﻿")
    src = _BLOCK_COMMENT_RE.sub(" ", src)
    src = _LINE_COMMENT_RE.sub(" ", src)
    src = _VERBATIM_STRING_RE.sub('""', src)
    src = _STRING_RE.sub('""', src)
    return src


def parse_usings(src: str) -> list[str]:
    cleaned = _strip_comments_strings(src)
    usings: list[str] = []
    for m in _USING_RE.finditer(cleaned):
        usings.append(m.group(1))
    for m in _USING_ALIAS_RE.finditer(cleaned):
        usings.append(m.group(2))
    return list(dict.fromkeys(usings))  # de-dup, preserve order


# --------------------------------------------------------------------------- #
# Namespace -> NuGet package mapping (v2 base + v4 expansion).
#
# We mirror v2's ordering invariant: longest-prefix-match wins. Anything in
# this table will be auto-`dotnet add package`'d by either v2 (if v2 already
# had it) or v4 (the expansion below).
#
# Entries marked `# v4 NEW` are the v4 additions. Every NEW entry's package
# name must be a real, restorable NuGet ID -- the cost of a bogus name is a
# silent restore failure that masks every other diagnostic, so we only add
# packages we are sure of (Microsoft.* official, mainstream OSS, well-known
# test frameworks).
# --------------------------------------------------------------------------- #

_BCL_PREFIXES = (
    "System",
    "Microsoft.CSharp",
    "Microsoft.Win32",
    "Microsoft.VisualBasic",
)

_TEST_FRAMEWORKS = {
    "Xunit": None,
    "Xunit.Abstractions": None,
    "NUnit": "NUnit",
    "NUnit.Framework": "NUnit",
    "Microsoft.VisualStudio.TestTools.UnitTesting": "MSTest.TestFramework",
}

# Drop-prefixes (carried verbatim from v2; v4 doesn't need these because we
# don't drop tasks, but we still want to skip these namespaces during package
# mapping so we don't try to `dotnet add package FastTests`).
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
    "System.Web.UI",
    "System.Web.Mvc",
    "System.Web.Http",
    "System.Web.Razor",
    "System.Web.WebPages",
    "ManagedTests",
    "Internal.Cryptography",
    "ScriptCs",
    "Microsoft.AspNet.",
)

# Project-internal prefixes -- these are explicit "do NOT try to map" hints.
# v3 enrichment in instruction.md tells the agent to define them locally;
# v4 must therefore NOT add a package for them (a bogus restore would
# clobber the actual error). DotNetNuke is a CMS whose modular packages
# rarely restore cleanly; IO.Swagger.* is a generated client namespace.
_V4_PROJECT_INTERNAL_PREFIXES = (
    "DotNetNuke",
    "IO.Swagger",
)

# v2's mapping (replicated; longest-prefix wins).
_V2_NAMESPACE_TO_PACKAGE: dict[str, str | None] = {
    "NUnit": "NUnit",
    "Xunit": "xunit",
    "Newtonsoft.Json": "Newtonsoft.Json",
    "System.Text.Json": "System.Text.Json",
    "ProtoBuf": "protobuf-net",
    "MessagePack": "MessagePack",
    "YamlDotNet": "YamlDotNet",
    "Moq": "Moq",
    "NSubstitute": "NSubstitute",
    "FakeItEasy": "FakeItEasy",
    "FluentAssertions": "FluentAssertions",
    "Shouldly": "Shouldly",
    "AutoFixture": "AutoFixture",
    "AutoMapper": "AutoMapper",
    "Autofac": "Autofac",
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
    "Microsoft.CodeAnalysis": "Microsoft.CodeAnalysis",
    "Microsoft.CodeAnalysis.CSharp": "Microsoft.CodeAnalysis.CSharp",
    "Microsoft.CodeAnalysis.CSharp.Syntax": "Microsoft.CodeAnalysis.CSharp",
    "Microsoft.CodeAnalysis.VisualBasic": "Microsoft.CodeAnalysis.VisualBasic",
    "Microsoft.CodeAnalysis.Workspaces": "Microsoft.CodeAnalysis.Workspaces.Common",
    "Serilog": "Serilog",
    "NLog": "NLog",
    "log4net": "log4net",
    "Microsoft.Extensions.Logging.Console": "Microsoft.Extensions.Logging.Console",
    "RestSharp": "RestSharp",
    "Refit": "Refit",
    "Polly": "Polly",
    "Flurl": "Flurl.Http",
    "Flurl.Http": "Flurl.Http",
    "Dapper": "Dapper",
    "ServiceStack.OrmLite": "ServiceStack.OrmLite",
    "MongoDB.Driver": "MongoDB.Driver",
    "MongoDB.Bson": "MongoDB.Bson",
    "StackExchange.Redis": "StackExchange.Redis",
    "MySql.Data": "MySql.Data",
    "Npgsql": "Npgsql",
    "System.Data.SqlClient": "System.Data.SqlClient",
    "Azure.Core": "Azure.Core",
    "Azure.Storage.Blobs": "Azure.Storage.Blobs",
    "Azure.Identity": "Azure.Identity",
    "Microsoft.Azure.Cosmos": "Microsoft.Azure.Cosmos",
    "Amazon.S3": "AWSSDK.S3",
    "Amazon.DynamoDBv2": "AWSSDK.DynamoDBv2",
    "Google.Protobuf": "Google.Protobuf",
    "Google.Apis": "Google.Apis",
    "Google.Cloud.Storage.V1": "Google.Cloud.Storage.V1",
    "SixLabors.ImageSharp": "SixLabors.ImageSharp",
    "ImageMagick": "Magick.NET-Q16-AnyCPU",
    "SkiaSharp": "SkiaSharp",
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
    "TerraFX.Interop.Windows": None,
}

# v4 expansion -- explicit additions to plug the CS0234 holes from v3 QC.
# Each entry's package name is verified against the standard NuGet
# nuget.org-style mental model (Microsoft.* official; Akka official OSS;
# AspNetCore family from the .NET team; MathNet.Numerics; etc.).
#
# Notes on a few gotchas:
#  - `Microsoft.Azure.Services.AppAuthentication` -- official Azure SDK
#    package; deprecated in newer runtimes but still restorable on .NET 8.
#  - `MathNet.Numerics.LinearAlgebra` and other `MathNet.Numerics.*`
#    sub-namespaces all live in the single `MathNet.Numerics` package; the
#    longer-prefix entries below all map to the same package id, which is
#    fine -- `dotnet add package` is idempotent on repeat installs.
#  - `Akka.Actor` / `Akka.Routing` are part of the core `Akka` package;
#    `Akka.TestKit.Xunit2` is a separate package (xunit2 adapter).
#  - `Microsoft.AspNetCore.Builder` is part of the framework reference,
#    which a `dotnet new xunit` project can pull via the ASP.NET Core
#    framework-reference package; `Microsoft.AspNetCore.App` is the
#    standard meta-package id.
#  - `Microsoft.EntityFrameworkCore.InMemory` etc. are sub-packages, not
#    namespaces, but mapping via the namespace prefix is correct: the
#    namespace `Microsoft.EntityFrameworkCore.InMemory` is shipped by the
#    `Microsoft.EntityFrameworkCore.InMemory` package.
_V4_NAMESPACE_TO_PACKAGE_ADDITIONS: dict[str, str | None] = {
    # --- Microsoft.Extensions family (sub-packages not in v2) ---
    "Microsoft.Extensions.DependencyInjection.Abstractions":
        "Microsoft.Extensions.DependencyInjection.Abstractions",
    "Microsoft.Extensions.Hosting.Abstractions":
        "Microsoft.Extensions.Hosting.Abstractions",
    "Microsoft.Extensions.Options.ConfigurationExtensions":
        "Microsoft.Extensions.Options.ConfigurationExtensions",
    "Microsoft.Extensions.Caching.Distributed":
        "Microsoft.Extensions.Caching.Abstractions",
    "Microsoft.Extensions.Caching.Abstractions":
        "Microsoft.Extensions.Caching.Abstractions",
    "Microsoft.Extensions.Configuration.Binder":
        "Microsoft.Extensions.Configuration.Binder",
    "Microsoft.Extensions.Configuration.EnvironmentVariables":
        "Microsoft.Extensions.Configuration.EnvironmentVariables",
    "Microsoft.Extensions.Configuration.UserSecrets":
        "Microsoft.Extensions.Configuration.UserSecrets",
    "Microsoft.Extensions.Configuration.CommandLine":
        "Microsoft.Extensions.Configuration.CommandLine",
    "Microsoft.Extensions.Logging.Debug":
        "Microsoft.Extensions.Logging.Debug",
    # --- Azure SDK (legacy + cosmos sub-packages) ---
    "Microsoft.Azure.Services.AppAuthentication":
        "Microsoft.Azure.Services.AppAuthentication",
    "Microsoft.Azure.KeyVault":
        "Microsoft.Azure.KeyVault",
    "Microsoft.Azure.Storage":
        "Microsoft.Azure.Storage.Common",
    # --- ASP.NET Core (extra subpaths missing in v2) ---
    "Microsoft.AspNetCore.Builder":
        "Microsoft.AspNetCore.App.Ref",
    "Microsoft.AspNetCore.Mvc.Core":
        "Microsoft.AspNetCore.Mvc.Core",
    "Microsoft.AspNetCore.Mvc.ViewFeatures":
        "Microsoft.AspNetCore.Mvc.ViewFeatures",
    "Microsoft.AspNetCore.Mvc.RazorPages":
        "Microsoft.AspNetCore.Mvc.RazorPages",
    "Microsoft.AspNetCore.Mvc.Abstractions":
        "Microsoft.AspNetCore.Mvc.Abstractions",
    "Microsoft.AspNetCore.Mvc.Filters":
        "Microsoft.AspNetCore.Mvc.Core",
    "Microsoft.AspNetCore.Http.Abstractions":
        "Microsoft.AspNetCore.Http.Abstractions",
    "Microsoft.AspNetCore.Http.Features":
        "Microsoft.AspNetCore.Http.Features",
    "Microsoft.AspNetCore.DataProtection":
        "Microsoft.AspNetCore.DataProtection",
    "Microsoft.AspNetCore.Diagnostics":
        "Microsoft.AspNetCore.Diagnostics.Abstractions",
    # --- EntityFrameworkCore providers ---
    "Microsoft.EntityFrameworkCore.InMemory":
        "Microsoft.EntityFrameworkCore.InMemory",
    "Microsoft.EntityFrameworkCore.SqlServer":
        "Microsoft.EntityFrameworkCore.SqlServer",
    "Microsoft.EntityFrameworkCore.Sqlite":
        "Microsoft.EntityFrameworkCore.Sqlite",
    "Microsoft.EntityFrameworkCore.Design":
        "Microsoft.EntityFrameworkCore.Design",
    "Microsoft.EntityFrameworkCore.Relational":
        "Microsoft.EntityFrameworkCore.Relational",
    # --- Classic EF6 (separate package) ---
    "EntityFramework": "EntityFramework",
    "System.Data.Entity": "EntityFramework",
    # --- MathNet (single package covers all MathNet.Numerics.* sub-namespaces) ---
    "MathNet.Numerics": "MathNet.Numerics",
    "MathNet.Numerics.LinearAlgebra": "MathNet.Numerics",
    "MathNet.Numerics.Distributions": "MathNet.Numerics",
    "MathNet.Numerics.Statistics": "MathNet.Numerics",
    "MathNet.Numerics.Random": "MathNet.Numerics",
    "MathNet.Numerics.Optimization": "MathNet.Numerics",
    "MathNet.Numerics.Integration": "MathNet.Numerics",
    "MathNet.Numerics.Interpolation": "MathNet.Numerics",
    # --- Akka.NET ---
    "Akka": "Akka",
    "Akka.Actor": "Akka",
    "Akka.Routing": "Akka",
    "Akka.Configuration": "Akka",
    "Akka.Event": "Akka",
    "Akka.Util": "Akka",
    "Akka.Dispatch": "Akka",
    "Akka.IO": "Akka",
    "Akka.Pattern": "Akka",
    "Akka.Cluster": "Akka.Cluster",
    "Akka.Cluster.Tools": "Akka.Cluster.Tools",
    "Akka.Cluster.Sharding": "Akka.Cluster.Sharding",
    "Akka.Persistence": "Akka.Persistence",
    "Akka.Remote": "Akka.Remote",
    "Akka.Streams": "Akka.Streams",
    "Akka.TestKit": "Akka.TestKit",
    "Akka.TestKit.Xunit2": "Akka.TestKit.Xunit2",
    "Akka.TestKit.NUnit": "Akka.TestKit.NUnit",
    "Akka.DependencyInjection": "Akka.DependencyInjection",
    "Akka.Serialization": "Akka",
    # --- Test framework adapters (these were only partially in v2) ---
    "Xunit.Sdk": "xunit",
    "Xunit.Extensions": "xunit",
    "NUnit3TestAdapter": "NUnit3TestAdapter",
    "MSTest.TestFramework": "MSTest.TestFramework",
    "MSTest.TestAdapter": "MSTest.TestAdapter",
    # --- AutoMapper sub-packages ---
    "AutoMapper.Extensions.Microsoft.DependencyInjection":
        "AutoMapper.Extensions.Microsoft.DependencyInjection",
    "AutoMapper.QueryableExtensions": "AutoMapper",
    # --- FluentAssertions extras ---
    "FluentAssertions.Execution": "FluentAssertions",
    "FluentAssertions.Primitives": "FluentAssertions",
    "FluentAssertions.Collections": "FluentAssertions",
    "FluentAssertions.Numeric": "FluentAssertions",
    # --- Polly extras ---
    "Polly.Extensions.Http": "Polly.Extensions.Http",
    "Polly.Caching": "Polly.Caching",
    "Polly.Retry": "Polly",
    "Polly.Timeout": "Polly",
    "Polly.CircuitBreaker": "Polly",
    # --- Serilog sinks ---
    "Serilog.Sinks.Console": "Serilog.Sinks.Console",
    "Serilog.Sinks.File": "Serilog.Sinks.File",
    "Serilog.Extensions.Logging": "Serilog.Extensions.Logging",
    "Serilog.AspNetCore": "Serilog.AspNetCore",
    # --- MediatR extras ---
    "MediatR.Extensions.Microsoft.DependencyInjection":
        "MediatR.Extensions.Microsoft.DependencyInjection",
    # --- Newtonsoft helpers ---
    "Newtonsoft.Json.Linq": "Newtonsoft.Json",
    "Newtonsoft.Json.Schema": "Newtonsoft.Json.Schema",
    "Newtonsoft.Json.Bson": "Newtonsoft.Json.Bson",
    "Newtonsoft.Json.Converters": "Newtonsoft.Json",
    "Newtonsoft.Json.Serialization": "Newtonsoft.Json",
    # --- Moq extras ---
    "Moq.Protected": "Moq",
    "Moq.AutoMock": "Moq.AutoMock",
    # --- xUnit assert package (split out in modern xUnit) ---
    "Xunit.Assert": "xunit.assert",
    # --- Misc widely-used OSS ---
    "Castle.DynamicProxy": "Castle.Core",
    "Castle.Core": "Castle.Core",
    "NodaTime": "NodaTime",
    "Bogus": "Bogus",
}

# Merge: v2 first, then v4 additions (additions only fill gaps; we don't
# overwrite a v2 mapping with a different package, but if v2 already had
# the same key, dict-update keeps v4's value -- which is fine because v4's
# values for v2-overlapping keys are deliberately the same package id).
_NAMESPACE_TO_PACKAGE: dict[str, str | None] = {
    **_V2_NAMESPACE_TO_PACKAGE,
    **_V4_NAMESPACE_TO_PACKAGE_ADDITIONS,
}

# Sorted for longest-prefix-first matching.
_NS_MAP_SORTED: list[tuple[str, str | None]] = sorted(
    _NAMESPACE_TO_PACKAGE.items(), key=lambda kv: -len(kv[0])
)

# How many v4-NEW NuGet entries did we add (used by the report)?
N_V4_PACKAGE_ADDITIONS: int = sum(
    1 for k in _V4_NAMESPACE_TO_PACKAGE_ADDITIONS
    if k not in _V2_NAMESPACE_TO_PACKAGE
)


# --------------------------------------------------------------------------- #
# Mapping logic
# --------------------------------------------------------------------------- #


def _is_bcl(ns: str) -> bool:
    return any(ns == p or ns.startswith(p + ".") for p in _BCL_PREFIXES)


def _is_drop(ns: str) -> bool:
    for p in _DROP_PREFIXES:
        pclean = p.rstrip(".")
        if ns == pclean or ns.startswith(pclean + "."):
            return True
        if p.endswith(".") and ns.startswith(p):
            return True
    return False


def _is_project_internal(ns: str) -> bool:
    return any(
        ns == p or ns.startswith(p + ".") for p in _V4_PROJECT_INTERNAL_PREFIXES
    )


def map_using_to_package(ns: str) -> str | None:
    if _is_bcl(ns):
        return None
    if _is_drop(ns):
        return None
    if _is_project_internal(ns):
        return None
    if ns in _TEST_FRAMEWORKS:
        return _TEST_FRAMEWORKS[ns]
    for prefix, pkg in _NS_MAP_SORTED:
        if ns == prefix or ns.startswith(prefix + "."):
            return pkg
    return None


def packages_for_usings(usings: list[str]) -> list[str]:
    """Return deduped, ordered list of NuGet package ids implied by `usings`."""
    pkgs: list[str] = []
    seen: set[str] = set()
    has_nunit_framework = False
    has_mstest = False
    for ns in usings:
        pkg = map_using_to_package(ns)
        if pkg and pkg not in seen:
            pkgs.append(pkg)
            seen.add(pkg)
        if ns == "NUnit.Framework" or ns.startswith("NUnit.Framework."):
            has_nunit_framework = True
        if ns == "Microsoft.VisualStudio.TestTools.UnitTesting":
            has_mstest = True
    # Adapters required for NUnit/MSTest discovery under `dotnet test`.
    if has_nunit_framework and "NUnit3TestAdapter" not in seen:
        pkgs.append("NUnit3TestAdapter")
        seen.add("NUnit3TestAdapter")
    if has_mstest and "MSTest.TestAdapter" not in seen:
        pkgs.append("MSTest.TestAdapter")
        seen.add("MSTest.TestAdapter")
    return pkgs


def _packages_already_in_v2_block(testsh: str) -> set[str]:
    """Extract the set of package ids that v2 already injected into test.sh.

    Returns an empty set if the v2 block isn't present (caller decides what
    to do with that). v4 uses this to compute the DELTA -- only newly-known
    packages should be appended.
    """
    if _V2_TESTSH_MARKER not in testsh:
        return set()
    try:
        start = testsh.index(_V2_TESTSH_MARKER)
        end_idx = testsh.find(_V2_TESTSH_END_MARKER, start)
        block = testsh[start:end_idx] if end_idx != -1 else testsh[start:]
    except ValueError:
        return set()
    pkgs: set[str] = set()
    for line in block.split("\n"):
        # Match `dotnet add package <NAME>` -- we want the NAME.
        m = re.match(r"\s*dotnet\s+add\s+package\s+([\w\.\-]+)", line)
        if m:
            pkgs.add(m.group(1))
    return pkgs


# --------------------------------------------------------------------------- #
# test.sh patching
# --------------------------------------------------------------------------- #


def patch_test_sh(
    original: str, packages: list[str]
) -> tuple[str | None, dict]:
    """Apply the v4 test.sh patches: extra-package block + AllowUnsafeBlocks.

    Returns (new_contents, info_dict). `new_contents` is None if no change
    was made (already-patched on both v4 markers, OR unfamiliar template).
    info_dict has keys:
      - 'pkg_appended': bool
      - 'unsafe_appended': bool
      - 'pkg_count': int (number of packages this v4 run added)
      - 'reason_no_pkg' / 'reason_no_unsafe': str | None
    """
    info = {
        "pkg_appended": False,
        "unsafe_appended": False,
        "pkg_count": 0,
        "reason_no_pkg": None,
        "reason_no_unsafe": None,
    }

    new = original

    # --- Sub-patch A: extra `dotnet add package` block ---
    if _V4_TESTSH_PKG_MARKER in new:
        info["reason_no_pkg"] = "already_patched"
    else:
        existing_pkgs = _packages_already_in_v2_block(new)
        delta = [p for p in packages if p not in existing_pkgs]
        if not delta:
            info["reason_no_pkg"] = "no_new_packages"
        else:
            # We need an anchor. Preferred: append immediately after v2's end
            # marker. Fallback: append after the `cp /tests/TestSolution.cs .`
            # line (same anchor v2 used).
            block_lines: list[str] = []
            block_lines.append("")
            block_lines.append(f"    {_V4_TESTSH_PKG_MARKER}")
            block_lines.append(
                "    # Extra NuGet packages from v4's expanded namespace allowlist."
            )
            for pkg in delta:
                block_lines.append(
                    f'    dotnet add package {pkg} >/dev/null 2>&1 || dotnet add package {pkg}'
                )
            block_lines.append(
                "    dotnet restore >/dev/null 2>&1 || true"
            )
            block_lines.append(f"    {_V4_TESTSH_PKG_END_MARKER}")
            block_lines.append("")
            block = "\n".join(block_lines)

            if _V2_TESTSH_END_MARKER in new:
                # Insert AFTER v2's end marker line.
                pat = re.compile(
                    r"(^\s*" + re.escape(_V2_TESTSH_END_MARKER) + r"\s*\n)",
                    re.MULTILINE,
                )
                if pat.search(new):
                    new = pat.sub(r"\1" + block + "\n", new, count=1)
                    info["pkg_appended"] = True
                    info["pkg_count"] = len(delta)
                else:
                    info["reason_no_pkg"] = "v2_end_marker_unmatched"
            else:
                # No v2 block -- fall back to the same anchor v2 itself uses.
                pat = re.compile(r"(cp /tests/TestSolution\.cs\s*\.\s*\n)")
                if pat.search(new):
                    new = pat.sub(r"\1" + block + "\n", new, count=1)
                    info["pkg_appended"] = True
                    info["pkg_count"] = len(delta)
                else:
                    info["reason_no_pkg"] = "no_anchor_for_pkg_block"

    # --- Sub-patch B: enable unsafe code in TestProject.csproj ---
    if _V4_TESTSH_UNSAFE_MARKER in new:
        info["reason_no_unsafe"] = "already_patched"
    else:
        # We inject a sed line that runs RIGHT AFTER `dotnet new xunit -n
        # TestProject` (when cwd is still /app, before the script `cd`s
        # into TestProject). The sed inserts `<AllowUnsafeBlocks>true
        # </AllowUnsafeBlocks>` immediately after the first `<PropertyGroup>`
        # opener in TestProject/TestProject.csproj.
        unsafe_block_lines: list[str] = []
        unsafe_block_lines.append(f"    {_V4_TESTSH_UNSAFE_MARKER}")
        unsafe_block_lines.append(
            "    # Some tests/sources use `unsafe { }` blocks (CS0227 without this)."
        )
        unsafe_block_lines.append(
            "    sed -i '0,/<PropertyGroup>/{s|<PropertyGroup>|<PropertyGroup><AllowUnsafeBlocks>true</AllowUnsafeBlocks>|}' TestProject/TestProject.csproj || true"
        )
        unsafe_block_lines.append("")
        unsafe_block = "\n".join(unsafe_block_lines)

        # Anchor: the line that runs `dotnet new xunit -n TestProject`. Match
        # the whole line (including any trailing args/flags + newline), then
        # insert our block immediately after.
        pat = re.compile(
            r"(^\s*dotnet\s+new\s+xunit\s+-n\s+TestProject[^\n]*\n)",
            re.MULTILINE,
        )
        if pat.search(new):
            new = pat.sub(r"\1" + unsafe_block, new, count=1)
            info["unsafe_appended"] = True
        else:
            info["reason_no_unsafe"] = "no_dotnet_new_xunit_anchor"

    # If neither sub-patch fired AND the file already has both v4 markers,
    # treat as "no changes needed".
    if (not info["pkg_appended"]) and (not info["unsafe_appended"]):
        return None, info

    return new, info


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--drop-log",
        type=str,
        default=None,
        help=(
            "Optional path to write a TSV of (task_dir<TAB>reason) for tasks "
            "that v4 could NOT mutate (e.g. unfamiliar template). v4 is "
            "non-destructive, so this is informational, not a delete-list."
        ),
    )
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "instruction.md").exists()
    )
    if not task_dirs:
        print(f"No task dirs (with instruction.md) under {root}", file=sys.stderr)
        return 2

    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_changed = 0
    n_already_patched = 0
    n_skipped_no_test = 0
    n_skipped_unparseable = 0
    n_pkg_only = 0
    n_unsafe_only = 0
    n_pkg_and_unsafe = 0
    n_no_change_needed = 0
    n_pkg_anchor_missing = 0
    n_unsafe_anchor_missing = 0

    pkg_counter: Counter[str] = Counter()
    drop_log_lines: list[str] = []

    for i, d in enumerate(task_dirs, 1):
        test_path = d / "tests" / "TestSolution.cs"
        testsh_path = d / "tests" / "test.sh"
        if not test_path.is_file() or not testsh_path.is_file():
            n_skipped_no_test += 1
            drop_log_lines.append(f"{d.name}\tno_test_files")
            continue
        try:
            test_src = test_path.read_text(encoding="utf-8", errors="replace")
            testsh_src = testsh_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            drop_log_lines.append(f"{d.name}\tread_failed")
            continue

        usings = parse_usings(test_src)
        if not usings:
            # Tasks with no `using` directives are valid (fully-qualified
            # tests); we still want to apply the unsafe-code patch.
            pass

        packages = packages_for_usings(usings)

        # Already fully patched?
        if (
            _V4_TESTSH_PKG_MARKER in testsh_src
            and _V4_TESTSH_UNSAFE_MARKER in testsh_src
        ):
            n_already_patched += 1
            continue

        new_testsh, info = patch_test_sh(testsh_src, packages)

        if info["reason_no_pkg"] == "no_anchor_for_pkg_block":
            n_pkg_anchor_missing += 1
            drop_log_lines.append(f"{d.name}\tpkg_anchor_missing")
        if info["reason_no_unsafe"] == "no_dotnet_new_xunit_anchor":
            n_unsafe_anchor_missing += 1
            drop_log_lines.append(f"{d.name}\tunsafe_anchor_missing")

        if new_testsh is None:
            # Both sub-patches no-op'd -- either fully patched on a
            # previous run OR template is unfamiliar AND v2 block missing.
            n_no_change_needed += 1
            continue

        if info["pkg_appended"] and info["unsafe_appended"]:
            n_pkg_and_unsafe += 1
        elif info["pkg_appended"]:
            n_pkg_only += 1
        elif info["unsafe_appended"]:
            n_unsafe_only += 1

        if info["pkg_appended"]:
            for pkg in packages:
                pkg_counter[pkg] += 1

        n_changed += 1
        if not args.dry_run:
            testsh_path.write_text(new_testsh, encoding="utf-8")

        if i % 200 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"pkg_only={n_pkg_only} unsafe_only={n_unsafe_only} "
                f"pkg_and_unsafe={n_pkg_and_unsafe} "
                f"already_patched={n_already_patched} "
                f"no_change={n_no_change_needed} "
                f"skipped_no_test={n_skipped_no_test}",
                flush=True,
            )

    print(
        f"\nDone. {n_changed}/{n_total} task dirs modified "
        f"(dry_run={args.dry_run}).\n"
        f"  pkg_only                 = {n_pkg_only}\n"
        f"  unsafe_only              = {n_unsafe_only}\n"
        f"  pkg_and_unsafe           = {n_pkg_and_unsafe}\n"
        f"  already_patched_skip     = {n_already_patched}\n"
        f"  no_change_needed         = {n_no_change_needed}\n"
        f"  skipped_no_test          = {n_skipped_no_test}\n"
        f"  skipped_unparseable      = {n_skipped_unparseable}\n"
        f"  pkg_anchor_missing       = {n_pkg_anchor_missing}\n"
        f"  unsafe_anchor_missing    = {n_unsafe_anchor_missing}\n"
        f"  v4_new_nuget_entries     = {N_V4_PACKAGE_ADDITIONS}\n"
    )
    print("Top 20 most-frequent NuGet packages added by v4 (incl. v2 overlap):")
    for pkg, c in pkg_counter.most_common(20):
        print(f"  {c:>5}  {pkg}")

    if args.drop_log and drop_log_lines:
        Path(args.drop_log).write_text("\n".join(drop_log_lines) + "\n")
        print(f"\nDrop log: {args.drop_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
