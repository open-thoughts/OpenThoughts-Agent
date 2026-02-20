# SkyRL-Agent SWE-Bench Workflow Instructions

Follow these steps to resolve the issue:

## 1. Explore the Codebase

First, explore the codebase to locate and understand the code relevant to the issue.

- Use efficient search commands to identify key files and functions.
- You should err on the side of caution and look at various relevant files and build your understanding of:
  - How the code works
  - What are the expected behaviors and edge cases
  - What are the potential root causes for the given issue

## 2. Assess Reproducibility

Assess whether you can reproduce the issue:

- Create a script at `{workspace}/reproduce_issue.py` that demonstrates the error.
- Execute this script to confirm the error behavior.
- You should reproduce the issue before fixing it.
- Your reproduction script should also assert the expected behavior for the fixed code.

## 3. Analyze Root Cause

Analyze the root cause:

- Identify the underlying problem based on your code exploration and reproduction results.
- Critically analyze different potential approaches to fix the issue.
- **You NEED to explicitly reason about multiple approaches to fix the issue.** Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
- You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

## 4. Implement Your Solution

Implement your solution:

- Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
- You should be thorough and methodical.

## 5. Verify Your Solution

Verify your solution:

- Rerun your reproduction script to confirm the error is fixed.
- If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

## 6. Run Unit Tests

Run unit tests:

- Find and run the relevant unit tests relevant to the performed fix.
- You should run the unit tests to ensure your solution is correct and does not cause any regressions.
- In cases where the unit tests do not pass, you should consider whether the unit tests do not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
- Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
  - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
  - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
  - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
- **RUN ALL relevant unit tests** to ensure your solution is correct and does not cause any regressions.

## 7. Test Edge Cases

Test edge cases:

- Identify potential edge cases that might challenge your solution.
- Create additional test cases in a separate file `{workspace}/edge_case_tests.py`.
- Execute these tests to verify your solution's robustness.
- You should run multiple rounds of edge cases. When creating edge cases:
  - Consider complex scenarios beyond the original issue description
  - Test for regressions to ensure existing functionality remains intact

## 8. Refine if Necessary

Refine if necessary:

- If edge case testing reveals issues, refine your solution accordingly.
- Ensure your final implementation handles all identified scenarios correctly.
- Document any assumptions or limitations of your solution.

## 9. Submit Your Solution

Submit your solution:

- Once you have verified your solution, submit your solution using the `finish` tool.

---

**A successful resolution means:**

- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled

---

**Task Details:**

