"""51 bug-type prompts ported from SERA's `sera/constants.py::ROLLOUT_ONE_PROMPTS`.

The upstream prompts target a specific function via `{{start_fn}}` in
`{{start_fn_file}}`. Our SERAlike variant operates over the whole codebase
(option a: no per-task function pointer) so we mechanically rewrote each
prompt to refer to "any function in this codebase" instead.

Each prompt describes ONE of the 51 bug categories (state mgmt, code clarity,
edge cases, ...). For every Harbor task we sample one of these 51 deterministically
keyed by the task's `path` field, so the rewrite is reproducible.

Source: https://github.com/allenai/SERA/blob/main/sera/constants.py @ L291-L343
"""

# Each entry is a (short_label, paragraph) pair. The label is for accounting; the
# paragraph is what gets stitched into the SERAlike instruction.md.
BUG_PROMPTS: list[tuple[str, str]] = [
    ("possible_bug",
     "There may be a bug somewhere in this codebase. When code in this library is exercised, behavior is not what is expected. The issue could be anywhere in the codebase."),
    ("resource_lifecycle",
     "There appears to be an issue with resource cleanup and lifecycle management somewhere in this codebase. Resources (such as file handles, database connections, network sockets, or memory allocations) are not being properly acquired, released, or managed. This may cause resource exhaustion, memory leaks, or dangling references."),
    ("data_validation",
     "There appears to be an issue with data validation and type safety somewhere in this codebase. Data is not being properly validated, type-checked, or sanitized. This may cause incorrect data to propagate through the system, runtime type errors, or unexpected behavior when invalid inputs are processed."),
    ("error_handling",
     "There appears to be an issue with error handling and exception management somewhere in this codebase. Errors are not being properly caught, handled, or propagated. This may cause silent failures, incorrect error messages, unhandled exceptions, or improper recovery from error conditions."),
    ("state_management",
     "There appears to be an issue with state management and data consistency somewhere in this codebase. State is not being properly initialized, updated, or synchronized. This may cause stale data, race conditions, inconsistent state across components, or incorrect behavior when state changes occur."),
    ("configuration_init",
     "There appears to be an issue with configuration and initialization logic somewhere in this codebase. The system is not being properly configured, initialized, or set up with the correct parameters and dependencies. This may cause incorrect default values, missing initialization steps, improper dependency injection, or configuration conflicts."),
    ("concurrency",
     "There appears to be an issue with concurrency and thread safety somewhere in this codebase. Operations are not being properly synchronized or protected from concurrent access. This may cause race conditions, deadlocks, data corruption, or unexpected behavior when multiple threads or processes access shared resources."),
    ("api_contracts",
     "There appears to be an issue with API contracts and interface compatibility somewhere in this codebase. Function signatures, return types, or expected behaviors do not match what callers expect or what documentation specifies. This may cause incorrect parameter passing, mismatched return values, breaking changes to public APIs, or violations of interface contracts."),
    ("performance",
     "There appears to be an issue with performance and efficiency somewhere in this codebase. Operations are not being performed optimally, leading to unnecessary computational overhead, redundant operations, inefficient algorithms, or excessive resource consumption. This may cause slow execution times, high memory usage, or poor scalability."),
    ("dependency_coupling",
     "There appears to be an issue with dependency management and module coupling somewhere in this codebase. Dependencies are not being properly managed, leading to tight coupling, circular dependencies, missing imports, or incorrect dependency injection. This may cause import errors, initialization failures, difficulty in testing, or violations of separation of concerns."),
    ("control_flow",
     "There appears to be an issue with control flow and logic correctness somewhere in this codebase. The execution path does not follow the intended logic, leading to incorrect branching, unreachable code, improper loop termination, or missing conditional checks. This may cause unexpected behavior, incorrect results, or logic that doesn't match the intended business requirements."),
    ("data_transform",
     "There appears to be an issue with data transformation and processing logic somewhere in this codebase. Data is not being correctly transformed, parsed, serialized, or converted between different formats or representations. This may cause data loss, incorrect mappings, malformed output, or failures when converting between types, structures, or encodings."),
    ("async_callbacks",
     "There appears to be an issue with asynchronous operations and callback handling somewhere in this codebase. Asynchronous operations are not being properly coordinated, awaited, or sequenced. This may cause timing issues, unhandled promises, callback hell, incorrect execution order, or failures in async/await patterns."),
    ("edge_cases",
     "There appears to be an issue with boundary conditions and edge case handling somewhere in this codebase. The code does not properly handle edge cases, boundary values, null/empty inputs, or extreme conditions. This may cause crashes, incorrect behavior with empty collections, off-by-one errors, null pointer exceptions, or failures when processing minimum/maximum values or unexpected input combinations."),
    ("maintainability_docs",
     "There appears to be an issue with code maintainability and documentation somewhere in this codebase. The code is difficult to understand, poorly documented, or contains misleading comments and naming. This may cause confusion about intended behavior, difficulty in debugging, incorrect assumptions by developers, or violations of coding standards and best practices."),
    ("security_input_san",
     "There appears to be an issue with security and input sanitization somewhere in this codebase. User inputs or external data are not being properly validated, sanitized, or escaped. This may cause security vulnerabilities such as injection attacks, unauthorized access, data exposure, or the execution of malicious code."),
    ("modularity_reuse",
     "There appears to be an issue with code modularity and reusability somewhere in this codebase. The code contains duplicated logic, hardcoded values, or tightly coupled components that make it difficult to extend, modify, or reuse. This may cause maintenance difficulties, inconsistent behavior across similar operations, or the need to change code in multiple places for a single logical change."),
    ("testing_observability",
     "There appears to be an issue with testing and observability somewhere in this codebase. The code lacks proper logging, debugging capabilities, or testability features. This may cause difficulty in diagnosing issues, insufficient visibility into runtime behavior, inability to trace execution flow, or challenges in writing effective unit tests."),
    ("compatibility_platform",
     "There appears to be an issue with compatibility and platform-specific behavior somewhere in this codebase. The code does not properly handle differences across operating systems, Python versions, or runtime environments. This may cause failures on specific platforms, incorrect path handling, encoding issues, or behavior that works in one environment but fails in another."),
    ("business_logic",
     "There appears to be an issue with code correctness and business logic implementation somewhere in this codebase. The implementation does not correctly fulfill the intended requirements or produces results that violate expected invariants, constraints, or business rules. This may cause incorrect calculations, violated assumptions, inconsistent domain logic, or outputs that don't match specifications."),
    ("robustness_recovery",
     "There appears to be an issue with code robustness and failure recovery somewhere in this codebase. The code does not gracefully handle unexpected conditions, partial failures, or degraded states. This may cause cascading failures, inability to recover from transient errors, lack of fallback mechanisms, or system instability when components fail or behave unexpectedly."),
    ("scalability_extensibility",
     "There appears to be an issue with code scalability and extensibility somewhere in this codebase. The code is not designed to handle growth in data volume, user load, or feature requirements. This may cause performance degradation at scale, inability to add new features without major refactoring, hardcoded limits that prevent expansion, or architectural constraints that limit future development."),
    ("consistency_conventions",
     "There appears to be an issue with code consistency and convention adherence somewhere in this codebase. The code does not follow established patterns, naming conventions, or style guidelines used throughout the rest of the codebase. This may cause inconsistent behavior, confusion about expected patterns, difficulty in code navigation, or violations of project-specific conventions and idioms."),
    ("clarity_complexity",
     "There appears to be an issue with code clarity and complexity management somewhere in this codebase. The code is overly complex, contains nested logic that is difficult to follow, or has functions that do too many things at once. This may cause difficulty in understanding the code's purpose, challenges in debugging and testing, increased likelihood of bugs, or violations of single responsibility principles."),
    ("integration_components",
     "There appears to be an issue with code integration and component interaction somewhere in this codebase. Components are not properly communicating, coordinating, or integrating with each other. This may cause mismatched expectations between modules, incorrect data flow between components, broken contracts between layers, or failures in how different parts of the system work together."),
    ("completeness_missing",
     "There appears to be an issue with code completeness and missing functionality somewhere in this codebase. The implementation is incomplete, contains placeholder code, or is missing critical functionality that should be present. This may cause NotImplementedError exceptions, stub functions that don't perform their intended operations, incomplete feature implementations, or gaps in the expected behavior."),
    ("reliability_determinism",
     "There appears to be an issue with code reliability and deterministic behavior somewhere in this codebase. The code produces inconsistent results across multiple executions with the same inputs, or depends on uncontrolled external factors. This may cause non-deterministic behavior, flaky operations, unreliable outputs, or dependencies on global state, random values, or timing that make the code unpredictable."),
    ("assumptions_preconditions",
     "There appears to be an issue with code assumptions and precondition validation somewhere in this codebase. The code makes implicit assumptions about the state of the system, the values of inputs, or the availability of resources without properly validating these preconditions. This may cause unexpected failures, incorrect behavior when assumptions are violated, or silent bugs when the code operates on data that doesn't meet its implicit requirements."),
    ("side_effects_purity",
     "There appears to be an issue with code side effects and function purity somewhere in this codebase. Functions or their dependencies produce unintended side effects, modify global state, or have hidden dependencies that make the code difficult to reason about and test. This may cause unexpected mutations, order-dependent behavior, difficulty in isolating functionality, or violations of functional programming principles where applicable."),
    ("abstraction_interface",
     "There appears to be an issue with code abstraction and interface design somewhere in this codebase. The code exposes implementation details, has leaky abstractions, or provides interfaces that are too rigid or too permissive. This may cause tight coupling to internal details, difficulty in changing implementations, violation of encapsulation principles, or APIs that don't properly hide complexity from callers."),
    ("duplication_redundancy",
     "There appears to be an issue with code duplication and redundancy somewhere in this codebase. The code contains repeated logic, duplicated code blocks, or redundant operations that could be consolidated. This may cause maintenance burden, inconsistent updates across duplicated sections, increased bug surface area, or violations of the DRY (Don't Repeat Yourself) principle."),
    ("ordering_init_seq",
     "There appears to be an issue with code ordering and initialization sequence somewhere in this codebase. Operations are not being executed in the correct order, dependencies are initialized after they're needed, or setup steps occur out of sequence. This may cause use-before-initialization errors, incorrect state at critical moments, dependency resolution failures, or operations that assume prerequisites that haven't been established yet."),
    ("algorithmic",
     "There appears to be an issue with code correctness and algorithmic implementation somewhere in this codebase. The underlying algorithm or computational logic produces incorrect results, uses the wrong approach for the problem domain, or contains mathematical or logical errors in its implementation. This may cause wrong outputs, inefficient solutions, incorrect calculations, or algorithms that don't properly solve the intended problem."),
    ("output_correctness",
     "There appears to be an issue with code behavior and output correctness somewhere in this codebase. Functions produce incorrect results, unexpected outputs, or behavior that doesn't match the documented or intended functionality. This may cause wrong return values, incorrect state changes, unexpected side effects, or outputs that violate the function's contract."),
    ("coupling_dependencies",
     "There appears to be an issue with code coupling and dependency relationships somewhere in this codebase. Components have inappropriate dependencies, circular references, or tight coupling that makes the code fragile and difficult to modify. This may cause changes in one area to unexpectedly break other areas, difficulty in testing components in isolation, or architectural issues where components know too much about each other's internals."),
    ("naming_semantics",
     "There appears to be an issue with code naming and semantic clarity somewhere in this codebase. Variable names, function names, or parameter names are misleading, ambiguous, or don't accurately reflect their purpose or the data they contain. This may cause confusion about what the code does, incorrect usage by developers, misunderstandings about data flow, or code that appears to do one thing but actually does another."),
    ("readability_maintain",
     "There appears to be an issue with code readability and maintainability somewhere in this codebase. The code structure, formatting, or organization makes it difficult to understand, modify, or maintain. This may cause confusion about control flow, difficulty tracking variable scope, poor separation of concerns, or code that is hard to navigate and comprehend."),
    ("testability_design",
     "There appears to be an issue with code testability and design for testing somewhere in this codebase. The code structure makes it difficult to write effective tests, mock dependencies, or verify behavior in isolation. This may cause hard-to-test code, tight coupling to external systems, inability to inject test doubles, or code that requires complex setup for testing."),
    ("backwards_compat",
     "There appears to be an issue with code backwards compatibility and version migration somewhere in this codebase. The code does not properly handle legacy data formats, deprecated APIs, or migration paths from older versions. This may cause breaking changes for existing users, failures when processing data from previous versions, or lack of graceful degradation when interfacing with older systems or data structures."),
    ("invariants_postconditions",
     "There appears to be an issue with code invariants and postcondition guarantees somewhere in this codebase. Functions do not maintain expected invariants, violate postconditions, or leave the system in an inconsistent state after execution. This may cause broken assumptions for subsequent operations, violated contracts about what the function guarantees, or state that doesn't satisfy the conditions that should hold after the function completes."),
    ("ownership_responsibility",
     "There appears to be an issue with code ownership and responsibility distribution somewhere in this codebase. Functions are doing too much or too little, responsibilities are not clearly assigned, or the code violates single responsibility principles. This may cause functions that are difficult to understand and test, unclear boundaries between components, or code where it's not obvious which component should handle specific tasks."),
    ("flexibility_params",
     "There appears to be an issue with code flexibility and parameterization somewhere in this codebase. The code uses hardcoded values, lacks configurable parameters, or doesn't provide sufficient flexibility for different use cases. This may cause inability to customize behavior, need to modify source code for simple changes, lack of extensibility for different scenarios, or rigid implementations that don't accommodate varying requirements."),
    ("transactions_atomicity",
     "There appears to be an issue with code transaction and atomicity guarantees somewhere in this codebase. Operations that should be atomic are not properly grouped, leading to partial updates, inconsistent state when operations fail midway, or lack of rollback mechanisms. This may cause data corruption, incomplete operations leaving the system in an invalid state, or the inability to recover from failures that occur during multi-step processes."),
    ("efficiency_resources",
     "There appears to be an issue with code efficiency and resource utilization somewhere in this codebase. The code performs unnecessary work, redundant computations, or inefficient operations that waste CPU cycles, memory, or other system resources. This may cause slow performance, high resource consumption, repeated calculations of the same values, or operations that could be optimized, cached, or eliminated entirely."),
    ("context_environment",
     "There appears to be an issue with code context and environment awareness somewhere in this codebase. The code does not properly account for its execution environment, runtime context, or deployment configuration. This may cause failures in different environments (development, staging, production), incorrect behavior based on missing or misconfigured environment variables, or assumptions about file paths, network availability, or system capabilities that don't hold in all contexts."),
    ("composability_interaction",
     "There appears to be an issue with code composability and function interaction somewhere in this codebase. Functions are not properly coordinating their inputs and outputs, leading to mismatched data formats, incompatible return types, or broken composition patterns. This may cause difficulty chaining operations, unexpected data transformations between function calls, or failures when combining multiple functions that should work together seamlessly."),
    ("lifecycle_temporal",
     "There appears to be an issue with code lifecycle and temporal behavior somewhere in this codebase. Operations are not properly handling time-dependent behavior, expiration, timeouts, or temporal ordering. This may cause operations that should timeout to hang indefinitely, expired data being treated as valid, incorrect handling of time zones or timestamps, or race conditions due to improper temporal sequencing."),
    ("idempotency",
     "There appears to be an issue with code idempotency and operation repeatability somewhere in this codebase. Repeated invocations with the same inputs produce different results, or operations that should be safely repeatable cause unintended side effects on subsequent calls. This may cause inconsistent behavior when retrying operations, inability to safely re-execute functions, state pollution across multiple calls, or operations that should be idempotent but modify system state in cumulative or unpredictable ways."),
    ("null_safety",
     "There appears to be an issue with code null safety and optional value handling somewhere in this codebase. The code does not properly handle None values, optional parameters, or nullable return types. This may cause NoneType errors, incorrect assumptions about value presence, missing null checks before dereferencing, or improper handling of optional data that may or may not be present."),
    ("memory_objects",
     "There appears to be an issue with code memory management and object lifecycle somewhere in this codebase. Objects are not being properly created, destroyed, or their references managed correctly. This may cause memory leaks, premature garbage collection, dangling references, circular references preventing cleanup, or incorrect object retention leading to excessive memory usage."),
    ("io_dataflow",
     "There appears to be an issue with code input/output handling and data flow somewhere in this codebase. Data is not being correctly received from inputs, passed between functions, or returned to callers. This may cause incorrect parameter usage, lost data during function calls, mismatched input/output expectations, or data that doesn't flow properly through the call chain."),
]

assert len(BUG_PROMPTS) == 51, f"expected 51 SERA prompts, got {len(BUG_PROMPTS)}"


def render_instruction(bug_label: str, bug_paragraph: str) -> str:
    """Build the full SERAlike instruction.md body from one of the 51 prompts."""
    return f"""{bug_paragraph}

Can you help me implement the necessary changes to this repository to fix the issue described above?
I've already taken care of all changes to any of the test files. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-test files in the repository to ensure the issue is fixed.

Follow these steps to resolve the issue:
1. As a first step, explore the repository structure and read code that may be relevant.
2. If you identify a real, justifiable change related to this prompt, edit the source code to make the fix and run any reproduction script you create.
3. Think about edge cases and verify your fix handles them.

If after exploring the codebase you cannot identify a real, justifiable change related to the prompt above, output `<abstain/>` and stop. Do not invent a change just to satisfy the prompt — abstaining is a valid and preferred answer when no real bug is present.

(Bug-type label for accounting: `{bug_label}`)
"""
