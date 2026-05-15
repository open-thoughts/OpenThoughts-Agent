#!/usr/bin/env python3
"""
exp_rpt_stack-junit v3 patcher.

Bug (v2 → v3): 100% of v2 trials reward=0 because every trial fails with
`javac failed (exit 1), refusing to score as PASS`. The Java test files
reference symbols (Node, JsonObject, Connection, Channel, Annotator, BLACK,
ImportanceTransferTransaction, JavaFileObject, ...) that the agent has no
way to know about — the LLM-written instruction.md doesn't enumerate the
test's import set or required public API. Even with 35 valid tool calls
the agent cannot reconstruct the right package layout / imports.

Fix (v3): For each task, mechanically parse `tests/TestSolution.java` to
extract:
  - The package name
  - The full import list (split into well-known / project-internal)
  - The parent class (if `extends X`)
  - All identifiers referenced in the test body that are not in the
    "well-known" allowlist (java.*, javax.*, org.junit.*, org.mockito.*,
    java built-ins) — these are the classes the agent must define.
  - Method calls on the SUT instances, with arity (signature stub).

Then rewrite `instruction.md` to:
  - Preserve the original LLM-authored description as `## Original task
    description`.
  - Append a `## Test contract` block that surfaces the package, imports,
    parent class, and inferred public API stubs so the agent has a
    deterministic ground-truth contract to satisfy.

If TestSolution.java cannot be parsed (no `class` keyword, no package, etc.)
the task is skipped (left untouched) — the patcher reports the count.

Output cap: 8000 chars total per instruction.md to keep prompts manageable.

Usage:
  python data/patchers/patch_stack_junit_v3_tasks.py --root <dir> [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Java parsing helpers (regex-based; the test files are simple enough that a
# full parser would be overkill, and we have no JDT/javalang on the path).
# --------------------------------------------------------------------------- #

_PACKAGE_RE = re.compile(r"^\s*package\s+([a-zA-Z_][\w.]*)\s*;", re.MULTILINE)
_IMPORT_RE = re.compile(
    r"^\s*import\s+(static\s+)?([a-zA-Z_][\w.]*(?:\.\*)?)\s*;",
    re.MULTILINE,
)
# class Foo extends Bar implements Baz { ... }
_CLASS_DECL_RE = re.compile(
    r"\bclass\s+(\w+)\s*(?:<[^{>]+>)?\s*"
    r"(?:extends\s+([\w.<>,\s]+?))?\s*"
    r"(?:implements\s+([\w.<>,\s]+?))?\s*\{"
)
_METHOD_CALL_RE = re.compile(
    # captures `Receiver.method(` or `var.method(` — used for signature inference
    r"\b([A-Za-z_]\w*)\s*\(",
)
# Identifier-like tokens (we use this on a comment-stripped body)
_IDENT_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
# `new Foo(arg, arg)` and `new Foo<...>(arg)`
_NEW_RE = re.compile(r"\bnew\s+([A-Z][\w.]*)\s*(?:<[^()<>]*>)?\s*\(([^)(]*)\)")

# Comment / string stripper (best effort; tests don't generally embed code in strings).
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

# Names we consider part of "Java/JUnit/standard" stdlib — the agent should
# NOT need to define classes with these names.
_STDLIB_PREFIXES = (
    "java.",
    "javax.",
    "jakarta.",
    "sun.",
    "com.sun.",
    "kotlin.",
    "scala.",
    "org.junit",
    "org.junit.jupiter",
    "junit.",
    "org.hamcrest",
    "org.mockito",
    "org.assertj",
    "org.slf4j",
    "org.apache.commons.lang",
    "org.apache.commons.io",
    "org.apache.logging",
    "ch.qos.logback",
    "org.testng",
    "lombok.",
    "com.google.common",
    "com.google.gson",
    "com.fasterxml.jackson",
    "io.netty",
    "okhttp3",
    "okio",
)

# Single-name primitives / builtins reserved by Java itself.
_JAVA_RESERVED = {
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Byte",
    "Short", "Character", "Object", "Number", "Math", "System", "Class",
    "Throwable", "Exception", "RuntimeException", "Error", "Thread",
    "Runnable", "Iterable", "Iterator", "Comparable", "Comparator",
    "Override", "Deprecated", "SuppressWarnings", "FunctionalInterface",
    "Test", "Before", "BeforeEach", "BeforeAll", "After", "AfterEach",
    "AfterAll", "Rule", "ClassRule", "DisplayName", "Tag", "Disabled",
    "Ignore", "ParameterizedTest", "ValueSource", "MethodSource",
    "CsvSource", "Nested", "Mock", "InjectMocks", "Spy", "Captor",
    "RunWith", "Parameterized", "Parameters", "Category", "TestRule",
    "ExpectedException", "TemporaryFolder", "Timeout", "Mockito",
    "ArgumentMatchers", "ArgumentCaptor", "Assert", "Assertions",
    "Assume", "AssumptionViolatedException", "List", "ArrayList",
    "LinkedList", "Map", "HashMap", "LinkedHashMap", "TreeMap",
    "Set", "HashSet", "LinkedHashSet", "TreeSet", "Collection",
    "Collections", "Arrays", "Optional", "Function", "Supplier",
    "Consumer", "BiConsumer", "BiFunction", "Predicate", "Stream",
    "Collectors", "IntStream", "LongStream", "DoubleStream",
    "File", "Files", "Path", "Paths", "InputStream", "OutputStream",
    "Reader", "Writer", "BufferedReader", "BufferedWriter",
    "FileInputStream", "FileOutputStream", "FileReader", "FileWriter",
    "IOException", "FileNotFoundException", "InterruptedException",
    "IllegalArgumentException", "IllegalStateException", "NullPointerException",
    "UnsupportedOperationException", "NumberFormatException",
    "ConcurrentModificationException", "ClassCastException",
    "Date", "Calendar", "TimeZone", "LocalDate", "LocalDateTime", "LocalTime",
    "Instant", "Duration", "Period", "ZoneId", "ZonedDateTime", "OffsetDateTime",
    "DateTimeFormatter", "Month", "Year", "DayOfWeek", "Random", "UUID",
    "Pattern", "Matcher", "Locale", "Charset", "StandardCharsets",
    "AtomicInteger", "AtomicLong", "AtomicBoolean", "AtomicReference",
    "ConcurrentHashMap", "CopyOnWriteArrayList", "CountDownLatch",
    "ExecutorService", "Executors", "Future", "TimeUnit", "Callable",
    "Logger", "Level", "LoggerFactory", "BigDecimal", "BigInteger",
    "Iterator", "Spliterator", "Comparator", "Comparable",
    "True", "False", "Null", "True", "False",
}

_JAVA_KEYWORDS_LOWER = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "throw", "throws", "try", "catch",
    "finally", "new", "this", "super", "void", "true", "false", "null",
    "public", "private", "protected", "static", "final", "abstract",
    "synchronized", "volatile", "transient", "native", "strictfp",
    "class", "interface", "enum", "extends", "implements", "package",
    "import", "instanceof", "var",
    "int", "long", "short", "byte", "float", "double", "boolean", "char",
    "assert",
}


def _strip_comments_and_strings(src: str) -> str:
    """Remove /* */ and // comments and string literals so identifier scans don't pick them up."""
    src = _BLOCK_COMMENT_RE.sub(" ", src)
    src = _LINE_COMMENT_RE.sub(" ", src)
    src = _STRING_RE.sub('""', src)
    return src


def _is_stdlib(import_path: str) -> bool:
    return any(import_path == p.rstrip(".") or import_path.startswith(p) for p in _STDLIB_PREFIXES)


def _short_name(import_path: str) -> str:
    """Return the simple name from a fully qualified import path."""
    last = import_path.rsplit(".", 1)[-1]
    return last


def parse_test_java(src: str) -> dict | None:
    """Parse a TestSolution.java; return None if essential structure is missing."""
    pkg_match = _PACKAGE_RE.search(src)
    if pkg_match:
        package = pkg_match.group(1)
    else:
        # Default (unnamed) package — Java's empty package. Tests written this
        # way can still be compiled; we just need to flag it for the prompt.
        package = ""

    imports: list[tuple[bool, str]] = []  # (is_static, fqn)
    for m in _IMPORT_RE.finditer(src):
        is_static = bool(m.group(1))
        fqn = m.group(2)
        imports.append((is_static, fqn))

    cls_match = _CLASS_DECL_RE.search(src)
    if not cls_match:
        return None
    test_class = cls_match.group(1)
    parent = cls_match.group(2)
    parent = parent.strip() if parent else None
    implements = cls_match.group(3)
    implements = implements.strip() if implements else None

    body_start = cls_match.end()
    body = src[body_start:]
    clean_body = _strip_comments_and_strings(body)

    # --- find method-declaration names so we can exclude them from "SUT
    # classes" (some test methods use CamelCase names like FirstNameOfRecord65
    # which would otherwise be misclassified as types). ---
    # Use a line-based, anchored regex to avoid catastrophic backtracking on
    # files with many `@Alerts(...)` annotations or huge string concatenations.
    method_decl_names: set[str] = set()
    # Match a single line that starts a method header:
    #   [modifiers] returnType methodName(... [partial args possible]
    # We don't try to match closing `{` on the same line — many headers span
    # multiple lines (e.g. `throws Foo {`). Simple-name return types only;
    # generic types are matched as a single run of [\w.<>,\s?]+ but bounded
    # to a line so backtracking is cheap.
    _METHOD_HEADER_LINE = re.compile(
        r"^\s*(?:@\w+(?:\([^\n]*\))?\s+)*"       # optional annotations on same line
        r"(?:(?:public|private|protected|static|final|synchronized|abstract|default|native)\s+){0,5}"
        r"(?:[\w]+(?:\s*<[^<>\n]{0,200}>)?(?:\[\])?)\s+"  # return type (single token, no backtracking)
        r"([A-Za-z_]\w*)\s*\(",
    )
    for line in clean_body.split("\n"):
        if "(" not in line:
            continue
        m = _METHOD_HEADER_LINE.match(line)
        if m:
            method_decl_names.add(m.group(1))

    # --- collect referenced simple identifiers (CamelCase) from the body ---
    identifiers: set[str] = set()
    for m in _IDENT_RE.finditer(clean_body):
        tok = m.group(0)
        if tok in _JAVA_RESERVED:
            continue
        if tok.lower() in _JAVA_KEYWORDS_LOWER:
            continue
        if tok in method_decl_names:
            continue
        identifiers.add(tok)

    # `new Foo(arg, arg)` for arity hint
    new_calls: dict[str, set[int]] = {}
    for m in _NEW_RE.finditer(clean_body):
        cls = m.group(1).split(".")[-1]
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args, ",") if a.strip()])
        new_calls.setdefault(cls, set()).add(arity)

    # Build the import name set: simple-name -> (fqn, stdlib?)
    imported_simple: dict[str, tuple[str, bool]] = {}
    static_imports: list[str] = []
    well_known_imports: list[str] = []
    project_imports: list[str] = []
    wildcard_imports: list[str] = []  # only NON-static wildcard imports of project packages
    for is_static, fqn in imports:
        if is_static:
            static_imports.append(fqn)
            # Static imports never define a class name in the body; skip class-resolution
            continue
        if fqn.endswith(".*"):
            base = fqn[:-2]
            if _is_stdlib(base):
                well_known_imports.append(fqn)
            else:
                project_imports.append(fqn)
                wildcard_imports.append(fqn)
            continue
        sn = _short_name(fqn)
        std = _is_stdlib(fqn)
        imported_simple[sn] = (fqn, std)
        if std:
            well_known_imports.append(fqn)
        else:
            project_imports.append(fqn)

    # Identifiers used in body that are NOT in stdlib imports or reserved.
    # These are the classes the agent must implement OR that the test imports
    # from project-internal packages.
    sut_candidates: set[str] = set()
    for tok in identifiers:
        if tok == test_class:
            continue
        if tok in imported_simple:
            fqn, std = imported_simple[tok]
            if std:
                continue
            sut_candidates.add(tok)
        else:
            # not imported at all and not in reserved -> likely defined in
            # the test's own package, in a wildcard imported package, or it's
            # a generic-type name the agent must define
            sut_candidates.add(tok)

    # Method calls of the form `var.method(...)` — capture the method name and arity
    method_calls: dict[str, set[int]] = {}
    for m in re.finditer(r"\.([a-z]\w*)\s*\(([^)(]*)\)", clean_body):
        name = m.group(1)
        if name in {"equals", "hashCode", "toString", "size", "length",
                    "expect", "expectMessage", "andReturn", "thenReturn",
                    "thenThrow", "when", "verify", "mock", "spy"}:
            continue
        args = m.group(2).strip()
        arity = 0 if not args else len([a for a in _split_top_level(args, ",") if a.strip()])
        method_calls.setdefault(name, set()).add(arity)

    return {
        "package": package,
        "test_class": test_class,
        "parent": parent,
        "implements": implements,
        "imports_well_known": sorted(set(well_known_imports)),
        "imports_project": sorted(set(project_imports)),
        "static_imports": sorted(set(static_imports)),
        "wildcard_imports": sorted(set(wildcard_imports)),
        "sut_candidates": sorted(sut_candidates),
        "constructors": {k: sorted(v) for k, v in new_calls.items()},
        "method_calls": {k: sorted(v) for k, v in method_calls.items()},
    }


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split a string by `sep` only at the top paren/angle/bracket depth."""
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


# --------------------------------------------------------------------------- #
# Instruction rewriter
# --------------------------------------------------------------------------- #

_PROMPT_CAP = 8000  # max chars in the rewritten instruction.md

# v3 marker we add to the prompt so we can detect already-patched files
_V3_MARKER = "<!-- laion v3 instruction patch: enriched with test contract -->"


def _format_test_contract(parsed: dict) -> str:
    """Produce the markdown 'Test contract' section appended to instruction.md."""
    lines: list[str] = []
    lines.append("## Test contract (auto-extracted from `tests/TestSolution.java`)")
    lines.append("")
    lines.append(
        "Below is a deterministic summary of what the JUnit test file expects. "
        "Use this as the source of truth for package, imports, and public API. "
        "If the original task description above conflicts with this section, prefer this section."
    )
    lines.append("")
    lines.append(f"**Language**: Java")
    if parsed["package"]:
        lines.append(f"**Test package**: `{parsed['package']}`")
    else:
        lines.append(
            "**Test package**: *(default/unnamed package)* — your sources should also be "
            "in the default package (placed directly in `/app/`, with no `package` "
            "declaration)."
        )
    lines.append(f"**Test class**: `{parsed['test_class']}`")
    if parsed["parent"]:
        lines.append(
            f"**Test extends**: `{parsed['parent']}` — you must also provide this base class "
            f"(or it must be importable from the test's package)."
        )
    if parsed["implements"]:
        lines.append(f"**Test implements**: `{parsed['implements']}`")
    lines.append("")

    # Imports — split into well-known vs project-internal
    if parsed["imports_well_known"]:
        sample = parsed["imports_well_known"][:25]
        more = len(parsed["imports_well_known"]) - len(sample)
        lines.append("**Well-known imports** (JUnit, JDK, common libs — already on the classpath):")
        lines.append("")
        for imp in sample:
            lines.append(f"- `{imp}`")
        if more > 0:
            lines.append(f"- … and {more} more")
        lines.append("")

    if parsed["imports_project"]:
        sample = parsed["imports_project"][:30]
        more = len(parsed["imports_project"]) - len(sample)
        lines.append(
            "**Project-internal imports** (you MUST provide these classes; place each in the matching package directory):"
        )
        lines.append("")
        for imp in sample:
            lines.append(f"- `{imp}`")
        if more > 0:
            lines.append(f"- … and {more} more")
        lines.append("")

    if parsed["static_imports"]:
        sample = parsed["static_imports"][:15]
        more = len(parsed["static_imports"]) - len(sample)
        lines.append("**Static imports** (constants/methods referenced statically):")
        lines.append("")
        for imp in sample:
            lines.append(f"- `import static {imp};`")
        if more > 0:
            lines.append(f"- … and {more} more")
        lines.append("")

    if parsed["wildcard_imports"]:
        lines.append("**Wildcard package imports** (implement classes from these packages as needed):")
        lines.append("")
        for imp in parsed["wildcard_imports"]:
            lines.append(f"- `{imp}`")
        lines.append("")

    # SUT class candidates — classes referenced but not in stdlib
    sut = [
        s for s in parsed["sut_candidates"]
        # Filter out things that look like exceptions or simple type params
        if len(s) > 1
    ]
    if sut:
        sample = sut[:30]
        more = len(sut) - len(sample)
        lines.append(
            "**Symbols the test references that you likely need to define** "
            "(class/enum/interface names not coming from JDK or JUnit):"
        )
        lines.append("")
        for s in sample:
            ctor = parsed["constructors"].get(s)
            if ctor is not None:
                arities = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in ctor)
                lines.append(f"- `{s}` — constructed via `new {s}(...)` with arities: {arities}")
            else:
                lines.append(f"- `{s}`")
        if more > 0:
            lines.append(f"- … and {more} more")
        lines.append("")

    # Method-call signature stubs
    methods = [
        (n, a) for n, a in parsed["method_calls"].items()
        if not n.startswith("_")
    ]
    methods.sort()
    if methods:
        sample = methods[:30]
        more = len(methods) - len(sample)
        lines.append(
            "**Methods the test invokes on instances** (you must expose at least these "
            "with the indicated arity; return type inferred from how the test uses the result):"
        )
        lines.append("")
        for name, arities in sample:
            ar_str = ", ".join(f"{a} arg{'s' if a != 1 else ''}" for a in arities)
            lines.append(f"- `.{name}(...)` — called with: {ar_str}")
        if more > 0:
            lines.append(f"- … and {more} more")
        lines.append("")

    lines.append("## Build & test environment")
    lines.append("")
    lines.append(
        "- The grader compiles your sources together with `tests/TestSolution.java` against "
        "JUnit's standalone console (`/junit/junit-platform-console-standalone.jar`) — both "
        "JUnit 4 (`org.junit.*`) and JUnit Jupiter (`org.junit.jupiter.api.*`) APIs are available."
    )
    lines.append(
        "- If a `pom.xml` exists in `/app`, the grader runs `mvn test` instead. Otherwise it "
        "runs `javac` over every `.java` file under `/app` plus `tests/TestSolution.java`, then "
        "executes JUnit Console with `--scan-class-path`."
    )
    lines.append(
        "- Place your sources under `/app/` in directories that match the package declaration. "
        "For example, a class in `package com.foo.bar;` must live at "
        "`/app/com/foo/bar/<ClassName>.java` (or under a Maven layout like "
        "`/app/src/main/java/com/foo/bar/`)."
    )
    lines.append(
        "- Project-internal imports referenced above (`imports_project`, wildcard-imported "
        "packages, and the test's own package) must resolve to source files you create."
    )
    lines.append("")

    return "\n".join(lines)


def rewrite_instruction(original: str, parsed: dict) -> str:
    """Produce the new instruction.md content."""
    contract = _format_test_contract(parsed)

    # Preserve the original task description verbatim under a sub-heading so
    # any nuance the LLM captured is still available, but make it
    # subordinate to the deterministic test contract.
    header = (
        f"# {parsed['test_class'].replace('Tests', '').replace('Test', '')} — Java task\n\n"
        f"{_V3_MARKER}\n\n"
        "Implement Java sources under `/app/` so that the JUnit test file at "
        "`/tests/TestSolution.java` compiles and passes. " + (
            f"The test is in package `{parsed['package']}` — your sources must be "
            "reachable via the same package layout (e.g. `/app/<package>/<Class>.java`)."
            if parsed["package"]
            else "The test uses the default (unnamed) package — your sources should also "
                 "be in the default package (place files directly under `/app/` "
                 "without a `package` declaration)."
        ) + "\n\n"
    )

    body = (
        header
        + contract
        + "\n## Original task description (LLM-generated; may be partial — defer to the contract above)\n\n"
        + original.strip()
        + "\n"
    )

    if len(body) > _PROMPT_CAP:
        # Trim the original-description tail rather than the contract.
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
    n_oversized = 0
    n_wildcard = 0

    for i, d in enumerate(task_dirs, 1):
        test_path = d / "tests" / "TestSolution.java"
        if not test_path.is_file():
            n_skipped_no_test += 1
            continue
        try:
            test_src = test_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            n_skipped_no_test += 1
            continue
        parsed = parse_test_java(test_src)
        if parsed is None:
            n_skipped_unparseable += 1
            continue
        if parsed["wildcard_imports"]:
            n_wildcard += 1

        instr_path = d / "instruction.md"
        original = instr_path.read_text(encoding="utf-8", errors="replace")
        if _V3_MARKER in original:
            n_already_patched += 1
            continue

        new_text = rewrite_instruction(original, parsed)
        if len(new_text) >= _PROMPT_CAP:
            n_oversized += 1

        n_changed += 1
        if not args.dry_run:
            instr_path.write_text(new_text, encoding="utf-8")

        if i % 250 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] changed={n_changed} "
                f"skipped_no_test={n_skipped_no_test} "
                f"skipped_unparseable={n_skipped_unparseable} "
                f"already_patched={n_already_patched} "
                f"oversized={n_oversized} "
                f"wildcard={n_wildcard}",
                flush=True,
            )

    print(
        f"\nDone. {n_changed}/{n_total} instruction.md files modified "
        f"(dry_run={args.dry_run}).\n"
        f"  skipped_no_test       = {n_skipped_no_test}\n"
        f"  skipped_unparseable   = {n_skipped_unparseable}\n"
        f"  already_patched_skip  = {n_already_patched}\n"
        f"  oversized_capped      = {n_oversized}\n"
        f"  wildcard_imports_used = {n_wildcard}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
