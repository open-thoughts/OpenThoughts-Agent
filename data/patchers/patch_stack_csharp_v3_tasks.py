#!/usr/bin/env python3
"""
exp_rpt_stack-csharp v3 patcher.

Bug (v2 -> v3): v2 enriched `instruction.md` with `using` directives + auto-
restored NuGet packages, which fixed the "missing-package CS0234" wave. But
QC sampling (10 trials, 1 PASS, 9 FAIL) showed the residual failures are now
*project-internal* `CS0103`/`CS0234`/`CS0246` errors:

    - The type or namespace name 'SchemaVersion' could not be found
    - The type or namespace name 'ExportController' does not exist
    - The type or namespace name 'IFhirRequestContextAccessor' could not be found
    - 'Stride.Core.Presentation.Collections' does not exist in the namespace 'Stride.Core'

These are types from the original repo's source files. The agent's
`Solution.cs` doesn't include them and the prompt doesn't enumerate them.

Fix (v3): Mirror `patch_stack_junit_v3_tasks.py`. For each task:
  1. Parse `tests/TestSolution.cs` (already extracted by v2; we leave the
     test file untouched -- it is the verifier's ground truth).
  2. Extract:
     - All top-level `using <NS>;` directives (preserve v2's NuGet allowlist
       split: well-known vs project-internal namespaces).
     - All capitalised identifier tokens referenced in the test body --
       these are the class/enum/interface names the agent must define.
     - `new Foo(...)` constructor sites (with arity).
     - `.method(...)` invocation sites (method name + arity).
     - Inheritance hints: `class X : Base, IFoo`.
     - Filter out:
        * BCL types (System.* / standard C# keywords / value types).
        * Test-framework decorators (Fact, Theory, Test, SetUp, ...).
        * Names that the test file itself defines (nested helper classes,
          method declarations).
  3. Rewrite `instruction.md` to PRESERVE the v2 contract verbatim and
     PREPEND a v3 "Project-internal symbols" section listing the symbols
     the agent must implement, plus inferred constructor/method stubs.
  4. Cap rewritten prompt at 8000 chars; truncate the original-task tail
     (NOT the contract) if needed.
  5. Idempotent via marker
        <!-- laion v3 instruction patch: enriched with C# test contract -->
     (separate from the v2 marker so we can patch on top of v2 cleanly).
  6. Drop tasks where TestSolution.cs cannot be parsed (no class decl).

Constraints (vs v2):
  - DO NOT modify `tests/TestSolution.cs` -- that's the verifier source of
    truth.
  - DO NOT remove or rewrite v2's `using`/NuGet block in instruction.md --
    v3 is cumulative.
  - DO NOT touch `tests/test.sh` -- v2 already injected `dotnet add package`
    + `dotnet restore`; v3 has no test.sh changes.

Usage:
  python data/patchers/patch_stack_csharp_v3_tasks.py \
      --root /path/to/exp_rpt_stack-csharp-v3 [--dry-run] [--limit N]
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

_V3_MARKER = (
    "<!-- laion v3 instruction patch: enriched with C# test contract -->"
)
_V2_MARKER = (
    "<!-- laion v2 instruction patch: enriched with C# test contract -->"
)

_PROMPT_CAP = 8000

# --------------------------------------------------------------------------- #
# C# parsing helpers
# --------------------------------------------------------------------------- #

_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_VERBATIM_STRING_RE = re.compile(r'@"(?:""|[^"])*"', re.DOTALL)
_INTERP_STRING_RE = re.compile(r'\$"(?:\\.|[^"\\])*"')  # best-effort; doesn't handle nested braces
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_CHAR_RE = re.compile(r"'(?:\\.|[^'\\])'")

# `using [static] Foo.Bar.Baz;`  -- captures the FQN
_USING_RE = re.compile(
    r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)
# `using Alias = Foo.Bar.Baz;` -- captures the RHS
_USING_ALIAS_RE = re.compile(
    r"^\s*using\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_][\w\.]*)\s*;",
    re.MULTILINE,
)

# `namespace Foo.Bar { ... }` or file-scoped `namespace Foo.Bar;`
_NAMESPACE_RE = re.compile(
    r"^\s*namespace\s+([A-Za-z_][\w\.]*)\s*[{;]",
    re.MULTILINE,
)

# Type-decl heuristic. Matches `class Foo`, `interface IFoo`, `record Foo`,
# `struct Foo`, `enum Foo`, optionally with generic params and inheritance.
_TYPE_DECL_RE = re.compile(
    r"\b(?:public\s+|internal\s+|private\s+|protected\s+|sealed\s+|abstract\s+|"
    r"static\s+|partial\s+|readonly\s+)*"
    r"(class|interface|record|struct|enum)\s+(\w+)"
    r"(?:\s*<[^{>]+>)?"  # optional generic args
    r"(?:\s*:\s*([\w\.\,<>\s]+?))?"  # optional inheritance list (greedy-but-stop-at-brace)
    r"\s*[\{\n]"
)

# `new Foo(...)`, `new Foo<T>(...)`, `new Foo.Bar(...)`
_NEW_RE = re.compile(
    r"\bnew\s+([A-Z][\w\.]*)\s*(?:<[^()<>]*>)?\s*\(([^)(]*)\)"
)

# `.method(...)` invocation
_METHOD_CALL_RE = re.compile(
    r"\.([a-z_]\w*)\s*\(([^)(]*)\)"
)

# All PascalCase / CamelCase TYPE identifiers in body. Excludes member accesses
# (preceded by `.`) and dotted FQN-internal segments. The negative lookbehind
# `(?<![A-Za-z0-9_.])` guarantees the token starts a fresh identifier path
# (not a continuation like `Foo.Bar` -> we only capture `Foo`, never `Bar`).
_IDENT_RE = re.compile(r"(?<![A-Za-z0-9_.])([A-Z][A-Za-z0-9_]*)\b")

# Method declaration heuristic (cs):
#   [modifiers] [<generic>] returnType MethodName(...
_METHOD_HEADER_LINE = re.compile(
    r"^\s*(?:\[[^\]\n]*\]\s*)*"  # decoration attributes
    r"(?:(?:public|private|protected|internal|static|virtual|override|"
    r"abstract|sealed|async|extern|partial|new|readonly)\s+){0,5}"
    r"(?:[\w\.<>,\s\?\[\]]+?)\s+"  # return type
    r"([A-Za-z_]\w*)\s*(?:<[^<>\n]{0,200}>)?\s*\(",
)

# C# keywords / built-in value types we never want to flag as project-internal.
_CSHARP_KEYWORDS_LOWER = {
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch",
    "char", "checked", "class", "const", "continue", "decimal", "default",
    "delegate", "do", "double", "else", "enum", "event", "explicit",
    "extern", "false", "finally", "fixed", "float", "for", "foreach",
    "goto", "if", "implicit", "in", "int", "interface", "internal", "is",
    "lock", "long", "namespace", "new", "null", "object", "operator",
    "out", "override", "params", "private", "protected", "public",
    "readonly", "ref", "return", "sbyte", "sealed", "short", "sizeof",
    "stackalloc", "static", "string", "struct", "switch", "this", "throw",
    "true", "try", "typeof", "uint", "ulong", "unchecked", "unsafe",
    "ushort", "using", "virtual", "void", "volatile", "while", "yield",
    "var", "async", "await", "partial", "record", "init", "get", "set",
    "nameof", "where", "from", "select", "join", "let", "orderby",
    "ascending", "descending", "group", "into", "by", "on", "equals",
    "global", "value", "add", "remove", "when",
}

# BCL / common-stdlib type names we should never list as "must implement".
# Mirrors the Java reserved set; tuned for C#.
_BCL_TYPES = {
    # primitive aliases
    "Boolean", "Byte", "SByte", "Char", "Decimal", "Double", "Single",
    "Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64", "IntPtr",
    "UIntPtr", "String", "Object", "Type",
    # System common
    "Exception", "ArgumentException", "ArgumentNullException",
    "ArgumentOutOfRangeException", "InvalidOperationException",
    "NotSupportedException", "NotImplementedException",
    "FormatException", "IndexOutOfRangeException",
    "NullReferenceException", "OverflowException", "DivideByZeroException",
    "InvalidCastException", "InvalidDataException", "IOException",
    "FileNotFoundException", "DirectoryNotFoundException",
    "UnauthorizedAccessException", "TimeoutException",
    "OperationCanceledException", "TaskCanceledException",
    "AggregateException",
    "Console", "Convert", "Math", "Environment", "DateTime", "DateTimeOffset",
    "TimeSpan", "TimeZone", "TimeZoneInfo", "Guid", "Random", "Tuple",
    "ValueTuple", "Enum", "Nullable", "Action", "Func", "Predicate",
    "Comparison", "Lazy", "WeakReference", "GC", "Buffer", "Array",
    "Activator", "Attribute", "AttributeUsage", "Flags", "Obsolete",
    "Serializable", "NonSerialized", "ThreadStatic", "Conditional",
    # Collections
    "IEnumerable", "IEnumerator", "ICollection", "IList", "IDictionary",
    "IReadOnlyList", "IReadOnlyCollection", "IReadOnlyDictionary",
    "List", "Dictionary", "HashSet", "SortedSet", "SortedList",
    "SortedDictionary", "Queue", "Stack", "LinkedList", "LinkedListNode",
    "KeyValuePair", "ReadOnlyCollection", "ReadOnlyDictionary",
    "ImmutableList", "ImmutableArray", "ImmutableDictionary",
    "ImmutableHashSet", "ImmutableSortedSet", "ImmutableSortedDictionary",
    "ConcurrentBag", "ConcurrentDictionary", "ConcurrentQueue",
    "ConcurrentStack", "BlockingCollection", "ICollection",
    # IO / threading
    "Stream", "MemoryStream", "FileStream", "StreamReader", "StreamWriter",
    "BinaryReader", "BinaryWriter", "TextReader", "TextWriter",
    "BufferedStream", "GZipStream", "DeflateStream", "Path", "File",
    "Directory", "FileInfo", "DirectoryInfo", "FileMode", "FileAccess",
    "FileShare", "SearchOption", "SeekOrigin",
    "Task", "ValueTask", "TaskCompletionSource", "CancellationToken",
    "CancellationTokenSource", "Thread", "ThreadPool", "Mutex",
    "Semaphore", "SemaphoreSlim", "ManualResetEvent", "AutoResetEvent",
    "Monitor", "Interlocked", "SpinLock", "SpinWait",
    # Linq / reflection
    "Enumerable", "Queryable", "IGrouping", "IOrderedEnumerable",
    "Expression", "ParameterExpression", "MemberExpression",
    "MethodCallExpression", "LambdaExpression", "Func",
    "PropertyInfo", "FieldInfo", "MethodInfo", "ConstructorInfo",
    "MemberInfo", "Assembly", "BindingFlags", "ParameterInfo",
    # Text / regex
    "StringBuilder", "Encoding", "ASCIIEncoding", "UTF8Encoding",
    "UnicodeEncoding", "Regex", "Match", "MatchCollection", "Group",
    "RegexOptions", "Capture",
    # Net
    "Uri", "UriBuilder", "WebClient", "HttpClient", "HttpRequestMessage",
    "HttpResponseMessage", "HttpStatusCode", "HttpMethod", "WebRequest",
    "WebResponse", "Cookie", "CookieContainer",
    # Json / xml
    "XmlDocument", "XmlElement", "XmlNode", "XmlNodeList", "XmlReader",
    "XmlWriter", "XDocument", "XElement", "XAttribute",
    # Common test-framework stuff (xunit / nunit / mstest / moq)
    "Assert", "Assertions", "Theory", "Fact", "InlineData", "MemberData",
    "ClassData", "Trait", "Skip", "Test", "TestCase", "TestCaseSource",
    "TestFixture", "TestFixtureSource", "SetUp", "TearDown",
    "OneTimeSetUp", "OneTimeTearDown", "Setup", "Teardown",
    "TestMethod", "TestClass", "TestInitialize", "TestCleanup",
    "ClassInitialize", "ClassCleanup", "AssemblyInitialize",
    "AssemblyCleanup", "DataRow", "DataTestMethod", "Ignore", "Category",
    "Description", "Author", "Property", "Order", "Parallelizable",
    "ParallelScope", "Apartment", "ApartmentState", "Timeout",
    "Repeat", "Retry", "MaxTime", "Platform", "Culture",
    "Mock", "Mock`1", "Times", "It", "Setup", "Returns", "Throws",
    "Verifiable", "Verify", "VerifyAll", "Callback", "MockRepository",
    "MockBehavior", "MockSequence", "Record", "Substitute", "Returns",
    # Common attribute markers
    "Trait", "Collection", "Order", "DisplayName", "Skip",
    # nameof / typeof / generics
    "T", "TKey", "TValue", "TResult", "TInput", "TOutput", "TItem",
    "TElement", "TEntity", "TModel", "TViewModel", "TService", "TException",
    "TSource", "TTarget", "TArg", "TReturn", "TFirst", "TSecond", "TThird",
    # Globalization / runtime / interop / debug
    "CultureInfo", "RegionInfo", "NumberFormatInfo", "DateTimeFormatInfo",
    "Calendar", "Marshal", "GCHandle", "GCHandleType", "RuntimeInformation",
    "OSPlatform", "Architecture", "Debug", "Debugger", "Trace", "Stopwatch",
    "Process", "ProcessStartInfo", "EnvironmentVariableTarget",
    # IDisposable + common interfaces
    "IDisposable", "IAsyncDisposable", "ICloneable", "IFormattable",
    "IConvertible", "IComparable", "IEquatable", "IEqualityComparer",
    "IComparer", "ISerializable", "IFormatter",
    # xUnit-2 helper interfaces
    "ITestOutputHelper", "IClassFixture", "ICollectionFixture",
    "IAsyncLifetime", "IUseFixture",
    # MSTest helpers commonly aliased into _BCL_TYPES region
    "TestCategory", "Priority", "Owner", "WorkItem", "DataRow",
    # NUnit asserts beyond Assert
    "CollectionAssert", "FileAssert", "StringAssert", "DirectoryAssert",
    # NUnit constraints
    "Is", "Has", "Does", "Throws", "Contains", "Iz",
    # Entity Framework Core
    "DbContext", "DbSet", "DbContextOptions", "DbContextOptionsBuilder",
    "ModelBuilder", "EntityTypeBuilder", "PropertyBuilder",
    # Common ASP.NET Core
    "HttpContext", "HttpRequest", "HttpResponse", "IServiceCollection",
    "IServiceProvider", "ServiceCollection", "ServiceProvider",
    # Windows / interop
    "DllImport", "MarshalAs", "UnmanagedType", "StructLayout",
    "LayoutKind", "FieldOffset", "InAttribute", "OutAttribute",
    # Misc
    "Activator", "AppDomain", "AppContext", "Environment", "Volatile",
    # More BCL value-type / enum names
    "StringComparison", "StringComparer", "StringSplitOptions",
    "DateTimeKind", "DateTimeStyles", "NumberStyles", "DayOfWeek",
    "TimeSpan", "TimeUnit", "DayOfWeek", "TypeCode",
    "OperatingSystem", "PlatformID", "Version",
    # Net IPAddress
    "IPAddress", "IPEndPoint", "EndPoint", "AddressFamily",
    "SocketType", "ProtocolType", "Socket", "NetworkStream",
    "WebSocket", "WebSocketState", "WebSocketMessageType",
    "WebSocketCloseStatus", "Dns", "DnsEndPoint", "IPHostEntry",
    # MSTest legacy
    "ExpectedException", "AssemblyInitialize", "AssemblyCleanup",
    "DeploymentItem", "HostType", "AspNetDevelopmentServerHost",
    # Newtonsoft.Json
    "JsonConvert", "JsonSerializer", "JsonReader", "JsonWriter",
    "JsonObject", "JsonArray", "JsonValue", "JsonProperty", "JsonToken",
    "JsonSerializerSettings", "JObject", "JArray", "JToken", "JProperty",
    "JsonIgnore", "JsonProperty", "JsonConverter",
    # System.Text.Json
    "JsonSerializerOptions", "JsonElement", "JsonDocument",
    "JsonNamingPolicy", "JsonSerializerContext",
    # Microsoft.Extensions.Configuration
    "ConfigurationBuilder", "IConfiguration", "IConfigurationRoot",
    "IConfigurationBuilder", "IConfigurationSection",
    # Microsoft.Extensions.DependencyInjection
    "ServiceLifetime", "ServiceDescriptor",
    # Microsoft.Extensions.Logging
    "ILogger", "ILoggerFactory", "ILoggerProvider", "LogLevel",
    "LoggerMessage",
    # Moq common
    "MockSequence", "Capture", "ItExpr", "Times",
    # FluentAssertions / Shouldly common
    "Should", "ShouldBe", "ShouldBeTrue", "ShouldBeFalse",
}

# Identifiers that look like they belong to the test framework (decorators)
# even when they show up in attribute brackets.
_TEST_FRAMEWORK_IDS = {
    "Fact", "Theory", "InlineData", "MemberData", "ClassData", "Trait",
    "Skip", "TestCaseSource", "TestCase", "Test", "TestFixture",
    "TestFixtureSource", "SetUp", "TearDown", "OneTimeSetUp",
    "OneTimeTearDown", "Setup", "Teardown", "TestMethod", "TestClass",
    "TestInitialize", "TestCleanup", "ClassInitialize", "ClassCleanup",
    "AssemblyInitialize", "AssemblyCleanup", "DataRow", "DataTestMethod",
    "Ignore", "Category", "Description", "Author", "Property", "Order",
    "Parallelizable", "ParallelScope", "Apartment", "ApartmentState",
    "Timeout", "Repeat", "Retry", "MaxTime", "Platform", "Culture",
    "Collection", "DisplayName", "UnitTest",
}

# v2's NuGet allowlist of "well-known" namespace top-levels. v3 must NOT
# re-list symbols whose using namespace is already covered by the v2
# auto-`dotnet add package` block. We mirror the prefix list used in
# patch_stack_csharp_tasks.py (BCL + the curated nuget map).
_V2_KNOWN_TOP_NAMESPACES = {
    # BCL / SDK
    "System", "Microsoft", "Internal",
    # Top-level packages from v2's NUGET map
    "Xunit", "NUnit", "Newtonsoft", "ProtoBuf", "MessagePack", "YamlDotNet",
    "Moq", "NSubstitute", "FakeItEasy", "FluentAssertions", "Shouldly",
    "AutoFixture", "AutoMapper", "Autofac", "Serilog", "NLog", "log4net",
    "RestSharp", "Refit", "Polly", "Flurl", "Dapper", "ServiceStack",
    "MongoDB", "StackExchange", "MySql", "Npgsql", "Azure", "Amazon",
    "Google", "SixLabors", "ImageMagick", "SkiaSharp", "CommandLine",
    "Humanizer", "MediatR", "Quartz", "Hangfire", "FluentValidation",
    "OpenTelemetry", "K4os", "DotNetty", "ICSharpCode", "Mono",
    "Lucene", "Apache",
}


def _strip_comments_strings(src: str) -> str:
    """Remove comments and string/char literals so identifier scans don't pick them up."""
    src = src.lstrip("﻿")
    src = _BLOCK_COMMENT_RE.sub(" ", src)
    src = _LINE_COMMENT_RE.sub(" ", src)
    src = _VERBATIM_STRING_RE.sub('""', src)
    src = _INTERP_STRING_RE.sub('""', src)
    src = _STRING_RE.sub('""', src)
    src = _CHAR_RE.sub("' '", src)
    return src


def _split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split `s` by `sep` only at top paren/angle/bracket depth."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def _is_v2_known(ns: str) -> bool:
    """Is namespace `ns` covered by v2's NuGet allowlist (so v3 should not flag it)?"""
    top = ns.split(".", 1)[0]
    return top in _V2_KNOWN_TOP_NAMESPACES


def _short_name(fqn: str) -> str:
    return fqn.rsplit(".", 1)[-1]


def parse_test_cs(src: str) -> dict | None:
    """Parse TestSolution.cs; return None if essential structure is missing."""
    cleaned = _strip_comments_strings(src)

    usings: list[str] = []
    for m in _USING_RE.finditer(cleaned):
        usings.append(m.group(1))
    for m in _USING_ALIAS_RE.finditer(cleaned):
        usings.append(m.group(2))
    usings = list(dict.fromkeys(usings))  # de-dup

    ns_match = _NAMESPACE_RE.search(cleaned)
    namespace = ns_match.group(1) if ns_match else None

    # Strip `using ...;` and `namespace ...;`/`namespace ... {` lines from a
    # body-only view so identifier scans don't pick up FQN segments inside
    # using/namespace declarations (those names belong to a different
    # universe -- they're already covered by the `using` extraction).
    body_view = cleaned
    body_view = _USING_RE.sub(" ", body_view)
    body_view = _USING_ALIAS_RE.sub(" ", body_view)
    body_view = _NAMESPACE_RE.sub(" ", body_view)

    # Find the FIRST top-level (non-nested) class declaration -- that's the
    # test class. Nested helpers come later in the file.
    type_decls: list[tuple[str, str, str | None]] = []  # (kind, name, inherits)
    for m in _TYPE_DECL_RE.finditer(cleaned):
        kind = m.group(1)
        name = m.group(2)
        inherits = m.group(3)
        if inherits:
            inherits = inherits.strip().rstrip(",")
        type_decls.append((kind, name, inherits))

    if not type_decls:
        return None

    # Take the first (outermost) type decl as the "test class".
    test_kind, test_class, test_inherits = type_decls[0]

    # Names declared by the test file (test class + nested helpers + enums) --
    # they are NOT project-internal types the agent must define.
    self_declared_names: set[str] = {name for _kind, name, _ in type_decls}

    # Method-declaration names so we can exclude them from "must implement"
    # symbol candidates (they're test methods, not types).
    method_decl_names: set[str] = set()
    for line in cleaned.split("\n"):
        if "(" not in line:
            continue
        m = _METHOD_HEADER_LINE.match(line)
        if m:
            method_decl_names.add(m.group(1))

    # PascalCase identifier set in body (capitalised tokens). Use the
    # body-only view to skip `using ...;` and `namespace ...;` lines.
    identifiers: set[str] = set()
    for m in _IDENT_RE.finditer(body_view):
        tok = m.group(1)
        if tok in _BCL_TYPES:
            continue
        if tok in _TEST_FRAMEWORK_IDS:
            continue
        if tok.lower() in _CSHARP_KEYWORDS_LOWER:
            continue
        if tok in self_declared_names:
            continue
        if tok in method_decl_names:
            continue
        # Top-level namespace tokens (e.g. `System.Type` -> `System` is the
        # leading token; the regex's negative-lookbehind cuts it from
        # `Type` already, but `System` itself slips through). Skip any
        # token that is the head of a v2-known package namespace or BCL.
        if tok in _V2_KNOWN_TOP_NAMESPACES:
            continue
        identifiers.add(tok)

    # Constructor sites: `new Foo(...)` -> arity
    new_calls: dict[str, set[int]] = {}
    for m in _NEW_RE.finditer(body_view):
        cls = _short_name(m.group(1))
        if cls in _BCL_TYPES or cls in self_declared_names:
            continue
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args) if a.strip()])
        new_calls.setdefault(cls, set()).add(arity)

    # `.method(...)` invocations -> arity
    method_calls: dict[str, set[int]] = {}
    _SKIP_METHOD_NAMES = {
        "ToString", "GetHashCode", "Equals", "GetType", "ReferenceEquals",
        "ContainsKey", "ContainsValue", "Contains", "IndexOf", "LastIndexOf",
        "Add", "Remove", "Clear", "TryGetValue", "Count", "Any", "All",
        "First", "FirstOrDefault", "Single", "SingleOrDefault", "Last",
        "LastOrDefault", "ToList", "ToArray", "ToDictionary", "ToHashSet",
        "Select", "Where", "OrderBy", "OrderByDescending", "GroupBy",
        "Sum", "Min", "Max", "Average", "Distinct", "Skip", "Take",
        "Setup", "Returns", "Throws", "Verify", "VerifyAll", "When",
        "Mock", "MockBehavior", "Substitute",
        "AreEqual", "AreNotEqual", "AreSame", "AreNotSame", "IsTrue",
        "IsFalse", "IsNull", "IsNotNull", "Throws", "DoesNotThrow",
        "Equal", "NotEqual", "Same", "NotSame", "True", "False",
        "Null", "NotNull", "Empty", "NotEmpty", "Contains",
        "DoesNotContain", "ThrowsAsync",
        "WriteLine", "Write",
        "GetProperty", "GetMethod", "GetField", "GetType", "GetCustomAttribute",
        "GetCustomAttributes", "Invoke",
    }
    for m in _METHOD_CALL_RE.finditer(body_view):
        name = m.group(1)
        if name in _SKIP_METHOD_NAMES:
            continue
        if name[0].isupper() and name in _BCL_TYPES:
            continue
        # Skip lower-case/Pascal helpers we already excluded; but most real
        # SUT method calls start lower-case (camelCase) or Pascal.
        # The regex already requires lower-case start, but we keep this
        # branch to allow extending.
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args) if a.strip()])
        method_calls.setdefault(name, set()).add(arity)

    # Project-internal namespaces (for the new "you must define types under
    # these namespaces" sub-heading). Take all `using NS;` whose top-level
    # is NOT in v2's allowlist.
    project_internal_namespaces = sorted(
        ns for ns in usings if not _is_v2_known(ns) and "." in ns
    )

    # Symbols are project-internal if:
    #   - referenced in body
    #   - not declared in this test file
    #   - not a BCL/test-framework name
    sut_candidates = sorted(s for s in identifiers if len(s) > 1)

    # Sort method calls; cap inside formatter.
    return {
        "namespace": namespace,
        "test_class": test_class,
        "test_inherits": test_inherits,
        "usings": usings,
        "project_internal_namespaces": project_internal_namespaces,
        "sut_candidates": sut_candidates,
        "constructors": {k: sorted(v) for k, v in new_calls.items()},
        "method_calls": {k: sorted(v) for k, v in method_calls.items()},
        "self_declared_names": sorted(self_declared_names),
    }


# --------------------------------------------------------------------------- #
# Instruction rewriter
# --------------------------------------------------------------------------- #


def _format_v3_block(parsed: dict) -> str:
    """Produce the v3 'Project-internal symbols' block to PREPEND to v2 output."""
    lines: list[str] = []
    lines.append(f"{_V3_MARKER}")
    lines.append("")
    lines.append("## Test contract (v3 enrichment): project-internal symbols required by the test")
    lines.append("")
    lines.append(
        "v2 enriched this prompt with the test's `using` directives and auto-restored "
        "all known NuGet packages. The remaining failure mode is that the test references "
        "**project-internal** types (classes/interfaces/enums from the original repo) that "
        "no NuGet package can supply -- you must define them yourself under `/app/`. "
        "Below is a deterministic list extracted from `tests/TestSolution.cs`."
    )
    lines.append("")

    if parsed.get("test_inherits"):
        lines.append(
            f"**Test class inherits**: `{parsed['test_inherits']}` -- if this base "
            "class is not from a known package (Xunit, NUnit, MSTest, ...), you "
            "must define it as well."
        )
        lines.append("")

    if parsed["project_internal_namespaces"]:
        sample = parsed["project_internal_namespaces"][:25]
        more = len(parsed["project_internal_namespaces"]) - len(sample)
        lines.append(
            "**Project-internal namespaces** the test imports (your source files "
            "must define types in these namespaces -- the verifier's auto-`dotnet "
            "add package` cannot supply them):"
        )
        lines.append("")
        for ns in sample:
            lines.append(f"- `{ns}`")
        if more > 0:
            lines.append(f"- ... and {more} more")
        lines.append("")

    sut = [s for s in parsed["sut_candidates"] if len(s) > 1]
    if sut:
        sample = sut[:40]
        more = len(sut) - len(sample)
        lines.append(
            "**Project-internal symbols referenced by the test** (each is a class/"
            "interface/enum/struct/record name not coming from the .NET BCL or a "
            "known NuGet package; you must define them):"
        )
        lines.append("")
        for s in sample:
            ctor = parsed["constructors"].get(s)
            if ctor is not None:
                arities = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in ctor)
                lines.append(f"- `{s}` -- constructed via `new {s}(...)` with arities: {arities}")
            else:
                lines.append(f"- `{s}`")
        if more > 0:
            lines.append(f"- ... and {more} more")
        lines.append("")

    methods = sorted(
        ((n, a) for n, a in parsed["method_calls"].items() if not n.startswith("_")),
        key=lambda kv: kv[0],
    )
    if methods:
        sample = methods[:30]
        more = len(methods) - len(sample)
        lines.append(
            "**Methods the test invokes on instances** (your types must expose "
            "at least these method names with the indicated arities; return types "
            "are inferred from how the test uses the result):"
        )
        lines.append("")
        for name, arities in sample:
            ar_str = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in arities)
            lines.append(f"- `.{name}(...)` -- called with: {ar_str}")
        if more > 0:
            lines.append(f"- ... and {more} more")
        lines.append("")

    lines.append(
        "Implement the symbols above as `Solution.cs` (or split across multiple "
        "`*.cs` files) under `/app/TestProject/`. Match the namespace listed in "
        "the matching `using` directive so the test resolves the type at compile time."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def rewrite_instruction(original: str, parsed: dict) -> str:
    """PREPEND the v3 block while preserving v2's contract verbatim."""
    if _V3_MARKER in original:
        return original  # already patched (defensive; main loop also checks)

    v3_block = _format_v3_block(parsed)

    # Strategy: insert the v3 block AFTER the top-level title (first `# ...`
    # line) but BEFORE the v2 marker / contract. This keeps v2 verbatim and
    # makes the v3 enrichment the first content the agent sees.
    title_match = re.match(r"^(#[^\n]*\n)\n*", original)
    if title_match:
        title = title_match.group(1)
        rest = original[title_match.end():]
        body = title + "\n" + v3_block + rest
    else:
        body = v3_block + original

    if len(body) > _PROMPT_CAP:
        # Trim the original-task tail (the section after "## Original task
        # description"), preserving title + v3 + v2-contract.
        marker = "## Original task description"
        idx = body.find(marker)
        if idx == -1:
            # Fall back: trim the very end.
            keep = _PROMPT_CAP - 200
            body = body[:keep] + (
                "\n\n[... description truncated to fit prompt budget ...]\n"
            )
        else:
            head = body[:idx + len(marker)]
            tail = body[idx + len(marker):]
            # Find the first newline after the marker to preserve the rest of the heading line.
            nl = tail.find("\n")
            if nl != -1:
                heading_rest = tail[:nl + 1]
                tail_body = tail[nl + 1:]
            else:
                heading_rest, tail_body = tail, ""
            keep = _PROMPT_CAP - len(head) - len(heading_rest) - 200
            if keep < 200:
                keep = 200
            tail_body = tail_body.strip()[:keep] + (
                "\n\n[... description truncated to fit prompt budget ...]\n"
            )
            body = head + heading_rest + tail_body

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
    n_skipped_no_test = 0
    n_skipped_unparseable = 0
    n_already_patched = 0
    n_oversized = 0
    n_no_sut_symbols = 0

    sut_counter: Counter[str] = Counter()

    for i, d in enumerate(task_dirs, 1):
        test_path = d / "tests" / "TestSolution.cs"
        if not test_path.is_file():
            n_skipped_no_test += 1
            continue
        try:
            test_src = test_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            continue

        parsed = parse_test_cs(test_src)
        if parsed is None:
            n_skipped_unparseable += 1
            continue

        instr_path = d / "instruction.md"
        original_instr = instr_path.read_text(encoding="utf-8", errors="replace")
        if _V3_MARKER in original_instr:
            n_already_patched += 1
            continue

        new_instr = rewrite_instruction(original_instr, parsed)
        if len(new_instr) >= _PROMPT_CAP:
            n_oversized += 1

        if not parsed["sut_candidates"]:
            n_no_sut_symbols += 1

        for s in parsed["sut_candidates"]:
            sut_counter[s] += 1

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(new_instr, encoding="utf-8")

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"skipped_no_test={n_skipped_no_test} "
                f"skipped_unparseable={n_skipped_unparseable} "
                f"already_patched={n_already_patched} "
                f"oversized={n_oversized} "
                f"no_sut_symbols={n_no_sut_symbols}",
                flush=True,
            )

    print(
        f"\nDone. {n_changed}/{n_total} instruction.md files modified "
        f"(dry_run={args.dry_run}).\n"
        f"  skipped_no_test       = {n_skipped_no_test}\n"
        f"  skipped_unparseable   = {n_skipped_unparseable}\n"
        f"  already_patched_skip  = {n_already_patched}\n"
        f"  oversized_capped      = {n_oversized}\n"
        f"  no_sut_symbols        = {n_no_sut_symbols}\n"
    )
    print("Top 20 most-frequent project-internal symbols:")
    for sym, c in sut_counter.most_common(20):
        print(f"  {c:>6}  {sym}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
