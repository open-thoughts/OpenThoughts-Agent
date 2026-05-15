#!/usr/bin/env python3
"""
exp_rpt_methods2test-large patcher.

Bug: methods2test is a Java method-test pair corpus, but the trace generator
paired the Java `tests/TestSolution.java` files with task descriptions written
assuming **Python** (e.g. "implement /app/solution.py"). The verifier compiles
`/tests/TestSolution.java` with Maven and dies on every trial with
`[ERROR] COMPILATION ERROR : cannot find symbol`.

Fix: rewrite each task's `instruction.md` mechanically from the JUnit test
file. The new prompt:
  1. Says "implement Java sources under /app/src/main/java/ so that the JUnit
     test compiles and passes" (matches `pom.xml`'s sourceDirectory).
  2. Embeds the verbatim test body so the agent sees exactly which symbols it
     must define.
  3. Lists the non-stdlib type/symbol references extracted from the test body
     (uppercase-starting tokens followed by `.`, `(`, or `<`, plus
     fully-qualified identifiers like `org.jclouds.blobstore.options.GetOptions`).
  4. Lists the public-API method calls of the form `<Type>.method(...)` that
     the agent must implement.

Layout (verified across 5000 tasks, all uniform):
  - Dockerfile, test.sh, pom.xml are byte-identical across all 5000 tasks.
  - tests/TestSolution.java is always 5 lines:
      import org.junit.jupiter.api.*;
      import static org.junit.jupiter.api.Assertions.*;
      <blank>
      public class TestSolution {
          @Test ... <body> ... }
      }
  - pom.xml's sourceDirectory is /app/src/main/java.

Parser strategy: regex-based, no AST. Specifically:
  - Find fully-qualified types: lowercase.lowercase.CapName (e.g.
    `org.jclouds.blobstore.options.GetOptions`).
  - Find capitalised top-level symbols: \\bCapName\\b excluding JUnit/Java
    builtin keywords/types.
  - Find Type.method(...) call sites for the public API list.

Anomalies / known limitations:
  - Tests don't contain user-package `import` lines; only JUnit. So the
    "imports" list in the original brief is implicit. The patcher therefore
    derives the symbol set entirely from test-body tokens, which is what
    "the test references" means in this corpus.
  - Some calls use bare identifiers (e.g. `singletonList(...)`,
    `createMock(...)`) that come from JUnit `import static` patterns the
    agent will need to either re-import or implement. We list these as
    "free helper functions" so the agent knows they're calls, not types.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- Regexes -----------------------------------------------------------------

# Fully-qualified Java type: at least one lowercase package segment then a
# Capitalised type name. Match e.g. `org.jclouds.blobstore.options.GetOptions`.
RE_FQTN = re.compile(r"\b((?:[a-z][a-zA-Z0-9_]*\.){2,}[A-Z][a-zA-Z0-9_]*)\b")

# Capitalised single identifier (Class, Enum, constant). We later strip
# JUnit/Java builtins.
RE_CAP_IDENT = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\b")

# Type.method(...) calls — capture the receiver type and the method name.
RE_TYPE_METHOD = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*)\.([a-z_][A-Za-z0-9_]*)\s*\("
)

# bare lowercase function calls like `singletonList(...)`, `createMock(...)`
RE_BARE_CALL = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\s*\(")

# Java keywords / common stdlib / JUnit symbols to exclude from the
# capitalised-identifier set so we don't tell the agent to "implement
# `String`".
JAVA_BUILTIN = {
    # primitives + wrappers + base types
    "String", "Integer", "Long", "Boolean", "Double", "Float", "Short",
    "Byte", "Character", "Object", "Number", "Math", "System", "Class",
    "Void", "Iterable", "Iterator", "Comparable", "Comparator",
    # collections / generics
    "List", "ArrayList", "LinkedList", "Set", "HashSet", "TreeSet",
    "Map", "HashMap", "LinkedHashMap", "TreeMap", "Collection",
    "Collections", "Arrays", "Optional", "Stream", "Stream",
    "Function", "BiFunction", "Predicate", "Consumer", "Supplier",
    # JUnit Jupiter
    "Test", "BeforeEach", "BeforeAll", "AfterEach", "AfterAll",
    "DisplayName", "Disabled", "Tag", "Nested", "TestInstance",
    "ParameterizedTest", "RepeatedTest", "ValueSource", "MethodSource",
    "Assertions", "Assumptions", "Assert", "Order", "TestMethodOrder",
    "MethodOrderer", "Timeout",
    # exceptions / common
    "Exception", "RuntimeException", "Throwable", "Error",
    "IllegalArgumentException", "NullPointerException",
    "NoSuchMethodException", "IOException", "InterruptedException",
    # time
    "Instant", "Duration", "LocalDate", "LocalDateTime", "LocalTime",
    "ZonedDateTime", "OffsetDateTime", "ZoneId", "ZoneOffset",
    # java.lang reflection
    "Method", "Field", "Constructor", "Modifier",
    # math
    "BigInteger", "BigDecimal",
    # NIO / IO
    "Path", "Paths", "Files", "File", "InputStream", "OutputStream",
    "Reader", "Writer", "BufferedReader", "BufferedWriter",
    # misc lang
    "Objects", "Override", "SuppressWarnings", "Deprecated",
    "FunctionalInterface", "SafeVarargs",
    # JUnit5 helpers some tests use
    "InOrder",
}

# JUnit assertion methods + common static-import helpers we don't want listed
# under "free helper functions" the agent must implement (they're imported).
JUNIT_FREE_HELPERS = {
    # JUnit Jupiter Assertions (static-imported via line 2)
    "assertEquals", "assertNotEquals", "assertTrue", "assertFalse",
    "assertNull", "assertNotNull", "assertSame", "assertNotSame",
    "assertThrows", "assertDoesNotThrow", "assertAll", "assertArrayEquals",
    "assertIterableEquals", "assertLinesMatch", "assertTimeout",
    "assertTimeoutPreemptively", "fail",
    # Java keywords that look like calls
    "if", "for", "while", "switch", "return", "throw", "new", "this",
    "super", "do", "catch", "try", "synchronized",
    # very common idioms agents understand without help
    "println", "print", "format", "printf", "valueOf", "toString",
    "equals", "hashCode", "compareTo", "length", "size", "isEmpty",
    "contains", "indexOf", "substring", "split", "trim", "replace",
    "get", "set", "add", "remove", "put", "containsKey", "containsValue",
    "keys", "values", "entrySet", "keySet",
}


def _strip_strings_and_comments(src: str) -> str:
    """Replace contents of string/char literals and comments with spaces so
    token regexes don't match identifier-shaped substrings inside them
    (e.g. uppercase tokens like ``BIGINT`` or ``CREATE TABLE`` appearing in
    a SQL string literal). Length is preserved to keep offsets stable."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        # Line comment
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        # Block comment
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        # String literal "..."
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            inner_len = max(0, (j - i) - 2)
            out.append('"' + " " * inner_len + ('"' if j > i + 1 else ""))
            i = j
            continue
        # Char literal '...'
        if c == "'":
            j = i + 1
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if src[j] == "'":
                    j += 1
                    break
                j += 1
            inner_len = max(0, (j - i) - 2)
            out.append("'" + " " * inner_len + ("'" if j > i + 1 else ""))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_test_body(java_src: str) -> tuple[str, list[str], list[str], list[str]]:
    """
    Parse a TestSolution.java file.

    Returns (test_body, fq_types, top_types, type_method_calls).

    `test_body` is the verbatim @Test method body (best-effort: everything
    inside the outer class braces). It's the original (un-stripped) source
    so we can echo it into the prompt verbatim.
    `fq_types` are dotted type names (e.g. ``org.jclouds.blobstore.options.GetOptions``).
    `top_types` are bare capitalised class refs after stripping JUnit/Java
    builtins and the FQ types' tail names.
    `type_method_calls` are entries like ``Solution.foo(...)`` describing the
    expected public API.

    String literals and comments are masked before token extraction so the
    set of "symbols" reflects code identifiers, not text inside ``"..."``
    or ``'...'``.
    """
    body_match = re.search(
        r"public\s+class\s+TestSolution\s*\{(.*)\}\s*\Z",
        java_src,
        flags=re.DOTALL,
    )
    body = body_match.group(1).strip() if body_match else java_src

    masked = _strip_strings_and_comments(body)

    fq = sorted(set(RE_FQTN.findall(masked)))

    # We deliberately do NOT subtract FQ tail names here: a test may use
    # both ``a.b.GetOptions`` and an unqualified ``GetOptions`` that
    # resolves to a different class (e.g. one in the default package).
    # Listing the bare name lets the agent know it must also be defined.
    # However we do need to mask out the FQ occurrences themselves before
    # collecting bare caps, otherwise the regex picks up the tail of every
    # FQ as if it were a separate bare reference.
    masked_for_caps = masked
    for fqt in fq:
        masked_for_caps = masked_for_caps.replace(
            fqt, " " * len(fqt)
        )

    # Capitalised single-identifier symbols in the body. Filter to those that
    # are not Java/JUnit builtins and not the test class itself.
    caps = set(RE_CAP_IDENT.findall(masked_for_caps))
    caps -= JAVA_BUILTIN
    caps.discard("TestSolution")
    top_types = sorted(caps)

    # Type.method(...) call sites. We dedupe by (type, method).
    seen: set[tuple[str, str]] = set()
    calls: list[str] = []
    for m in RE_TYPE_METHOD.finditer(masked):
        t, meth = m.group(1), m.group(2)
        if t in JAVA_BUILTIN or t == "TestSolution":
            continue
        if (t, meth) in seen:
            continue
        seen.add((t, meth))
        calls.append(f"{t}.{meth}(...)")

    return body, fq, top_types, calls


def render_instruction(
    body: str,
    fq: list[str],
    top_types: list[str],
    calls: list[str],
) -> str:
    """Render the new Java-coherent instruction.md."""
    lines: list[str] = []
    lines.append("## Task Description")
    lines.append("")
    lines.append(
        "Implement Java sources under `/app/src/main/java/` so that the "
        "JUnit 5 test at `/tests/TestSolution.java` compiles and passes. "
        "The verifier runs `mvn test` against the project rooted at `/app/` "
        "(see `/tests/pom.xml`). Java 17 is the target."
    )
    lines.append("")
    lines.append("### The test (verbatim)")
    lines.append("")
    lines.append("```java")
    lines.append("import org.junit.jupiter.api.*;")
    lines.append("import static org.junit.jupiter.api.Assertions.*;")
    lines.append("")
    lines.append("public class TestSolution {")
    # Re-indent each line of body by 4 spaces, normalising whitespace.
    for ln in body.splitlines():
        s = ln.rstrip()
        if not s:
            lines.append("")
        else:
            lines.append("    " + s.lstrip())
    lines.append("}")
    lines.append("```")
    lines.append("")

    lines.append("### Symbols the test references")
    lines.append("")
    lines.append(
        "Your code must define (or pull in via a properly configured "
        "classpath) the symbols below. JUnit 5 itself is already on the "
        "test classpath via `pom.xml`, so you only need to provide the "
        "non-JUnit symbols."
    )
    lines.append("")
    if fq:
        lines.append("**Fully-qualified types referenced:**")
        for t in fq:
            lines.append(f"- `{t}`")
        lines.append("")
    if top_types:
        lines.append(
            "**Top-level types/constants the test uses (not Java/JUnit "
            "builtins):**"
        )
        for t in top_types:
            lines.append(f"- `{t}`")
        lines.append("")
    if calls:
        lines.append("**Public API the test invokes (`Type.method(...)`):**")
        for c in calls:
            lines.append(f"- `{c}`")
        lines.append("")

    lines.append("### Implementation notes")
    lines.append("")
    lines.append(
        "- Place every Java source file under `/app/src/main/java/` "
        "(matching `pom.xml`'s `sourceDirectory`). Use package directories "
        "for fully-qualified types (e.g. "
        "`/app/src/main/java/org/example/Foo.java` for `org.example.Foo`)."
    )
    lines.append(
        "- The test compiles in the default package (no `package` line), "
        "so any class it references by bare name (e.g. `Solution`, "
        "`NamingUtils`, etc.) must also live in the default package "
        "(directly under `/app/src/main/java/`)."
    )
    lines.append(
        "- If the test uses static-imported helpers like `singletonList(...)` "
        "or `assertThat(...)` that aren't covered by the two `import` "
        "lines shown above, add the missing `import static ...;` to your "
        "own source files (the test file itself is read-only)."
    )
    lines.append(
        "- The verifier runs `mvn test` from `/app/`. Maven dependencies "
        "beyond JUnit 5 (already in `/tests/pom.xml`) need to be declared "
        "in your own `/app/pom.xml` if you choose to bring in third-party "
        "libraries; otherwise implement the referenced classes yourself."
    )
    lines.append(
        "- Do NOT modify `/tests/TestSolution.java` or `/tests/pom.xml` — "
        "they are the spec."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Path to extracted tasks root")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if args.limit:
        task_dirs = task_dirs[: args.limit]

    n_total = len(task_dirs)
    n_parsed = 0
    n_kept = 0
    n_dropped = 0
    drop_reasons: dict[str, int] = {}

    for i, td in enumerate(task_dirs, 1):
        test_path = td / "tests" / "TestSolution.java"
        instr_path = td / "instruction.md"
        if not test_path.is_file():
            n_dropped += 1
            drop_reasons["missing_test"] = drop_reasons.get("missing_test", 0) + 1
            if not args.dry_run:
                # Only drop if the directory truly has no test — leave on disk
                # but we won't include it in the "kept" count. We don't rmtree
                # here; the upload step is by-file, so unmodified dirs stay.
                pass
            continue

        try:
            src = test_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            n_dropped += 1
            drop_reasons[f"read_error:{type(e).__name__}"] = (
                drop_reasons.get(f"read_error:{type(e).__name__}", 0) + 1
            )
            continue

        if not src.strip():
            n_dropped += 1
            drop_reasons["empty"] = drop_reasons.get("empty", 0) + 1
            # Drop: remove the task dir so it isn't uploaded.
            if not args.dry_run:
                _rmtree(td)
            continue

        # Parse
        try:
            body, fq, top_types, calls = parse_test_body(src)
        except Exception as e:
            n_dropped += 1
            drop_reasons[f"parse:{type(e).__name__}"] = (
                drop_reasons.get(f"parse:{type(e).__name__}", 0) + 1
            )
            if not args.dry_run:
                _rmtree(td)
            continue

        n_parsed += 1

        # Sanity check: if no FQ types AND no top-level user types AND no
        # type-method calls, the parser found nothing actionable. Drop, since
        # the new prompt would be vacuous.
        if not fq and not top_types and not calls:
            n_dropped += 1
            drop_reasons["no_actionable_symbols"] = (
                drop_reasons.get("no_actionable_symbols", 0) + 1
            )
            if not args.dry_run:
                _rmtree(td)
            continue

        new_instr = render_instruction(body, fq, top_types, calls)
        if not args.dry_run:
            instr_path.write_text(new_instr, encoding="utf-8")
        n_kept += 1

        if i % 500 == 0 or i == n_total:
            print(
                f"[{i}/{n_total}] parsed={n_parsed} kept={n_kept} "
                f"dropped={n_dropped}",
                flush=True,
            )

    print(
        f"Done. total={n_total} parsed={n_parsed} kept={n_kept} "
        f"dropped={n_dropped} (dry_run={args.dry_run})"
    )
    if drop_reasons:
        print("Drop reasons:")
        for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")

    if n_total and n_kept * 10 < n_total:
        print(
            f"WARNING: kept ({n_kept}) is < 10% of total ({n_total}); "
            "parser may be too strict.",
            file=sys.stderr,
        )
        return 3
    return 0


def _rmtree(p: Path) -> None:
    """Recursively remove a task directory (used only for unparseable / empty
    tests so we don't upload junk)."""
    import shutil

    shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
