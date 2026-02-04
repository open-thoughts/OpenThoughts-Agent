#!/usr/bin/env python3
"""
Generate example tasks for all dataset types to verify task structure.

This script creates 5 example tasks for each dataset type using sample data,
without requiring external dataset downloads or LLM calls.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.commons import (
    DOCKERFILES,
    create_harbor_task_directory_generic,
    create_generic_test_sh,
    create_pytest_test_sh,
    create_io_test_sh,
    get_dockerfile,
)

# Base output directory
OUTPUT_BASE = Path(__file__).parent / "examples"


def create_example_tasks(
    dataset_name: str,
    samples: list,
    create_task_fn,
    num_examples: int = 5,
):
    """Create example tasks for a dataset."""
    output_dir = OUTPUT_BASE / f"example_{dataset_name}_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_examples} example tasks for {dataset_name}...")

    for i, sample in enumerate(samples[:num_examples]):
        create_task_fn(output_dir, i, sample, dataset_name)

    print(f"  -> Created in: {output_dir}")
    return output_dir


# =============================================================================
# UniTSyn Examples (Focal-Test Pairs)
# =============================================================================

UNITSYN_SAMPLES = [
    {
        "focal_function": '''def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b''',
        "test_function": '''def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0''',
        "language": "python",
        "repo": "example/math-utils",
    },
    {
        "focal_function": '''def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b''',
        "test_function": '''def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0''',
        "language": "python",
        "repo": "example/math-utils",
    },
    {
        "focal_function": '''def is_palindrome(s: str) -> bool:
    """Check if string is a palindrome."""
    return s == s[::-1]''',
        "test_function": '''def test_is_palindrome():
    assert is_palindrome("radar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True''',
        "language": "python",
        "repo": "example/string-utils",
    },
    {
        "focal_function": '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)''',
        "test_function": '''def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120''',
        "language": "python",
        "repo": "example/math-utils",
    },
    {
        "focal_function": '''def reverse_list(lst: list) -> list:
    """Reverse a list."""
    return lst[::-1]''',
        "test_function": '''def test_reverse_list():
    assert reverse_list([1, 2, 3]) == [3, 2, 1]
    assert reverse_list([]) == []
    assert reverse_list([1]) == [1]''',
        "language": "python",
        "repo": "example/list-utils",
    },
]


def create_unitsyn_task(output_dir, task_id, sample, prefix):
    instruction = f"""Implement a Python function based on the following specification:

{sample['focal_function'].split('"""')[1] if '"""' in sample['focal_function'] else 'Implement the function as specified.'}

The function should pass all provided test cases.

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_function']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "unitsyn", "repo": sample.get("repo", "")},
        solution_files={"solution.py": sample["focal_function"]},
    )


# =============================================================================
# TACO Examples (Competitive Programming)
# =============================================================================

TACO_SAMPLES = [
    {
        "problem": """Given an array of integers, find the sum of all elements.

Input: A list of integers
Output: The sum of all integers

Example:
Input: [1, 2, 3, 4, 5]
Output: 15""",
        "inputs": ["1 2 3 4 5", "10 20 30", "-1 1"],
        "outputs": ["15", "60", "0"],
    },
    {
        "problem": """Find the maximum element in an array.

Input: A list of integers
Output: The maximum integer

Example:
Input: [3, 1, 4, 1, 5, 9]
Output: 9""",
        "inputs": ["3 1 4 1 5 9", "1", "-5 -2 -8"],
        "outputs": ["9", "1", "-2"],
    },
    {
        "problem": """Count the number of even numbers in an array.

Input: A list of integers
Output: Count of even numbers

Example:
Input: [1, 2, 3, 4, 5, 6]
Output: 3""",
        "inputs": ["1 2 3 4 5 6", "1 3 5 7", "2 4 6 8"],
        "outputs": ["3", "0", "4"],
    },
    {
        "problem": """Check if a number is prime.

Input: A single integer n (n >= 2)
Output: "YES" if prime, "NO" otherwise

Example:
Input: 7
Output: YES""",
        "inputs": ["7", "4", "2", "17"],
        "outputs": ["YES", "NO", "YES", "YES"],
    },
    {
        "problem": """Reverse a string.

Input: A string
Output: The reversed string

Example:
Input: hello
Output: olleh""",
        "inputs": ["hello", "world", "a", "ab"],
        "outputs": ["olleh", "dlrow", "a", "ba"],
    },
]


def create_taco_task(output_dir, task_id, sample, prefix):
    instruction = f"""{sample['problem']}

Place your solution in /app/solution.py

Your solution should read from stdin and write to stdout."""

    test_files = {}
    for i, (inp, out) in enumerate(zip(sample["inputs"], sample["outputs"])):
        test_files[f"inputs/input_{i}.txt"] = inp
        test_files[f"outputs/output_{i}.txt"] = out

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_io_test_sh(),
        test_files=test_files,
        dataset_prefix=prefix,
        metadata={"source": "taco", "num_tests": len(sample["inputs"])},
    )


# =============================================================================
# Exercism Examples
# =============================================================================

EXERCISM_SAMPLES = [
    {
        "name": "two-fer",
        "instructions": """Create a function that returns a string like "One for X, one for me."

If no name is given, use "you" as the default.

Examples:
- two_fer() -> "One for you, one for me."
- two_fer("Alice") -> "One for Alice, one for me."
""",
        "test_code": '''def test_no_name_given():
    assert two_fer() == "One for you, one for me."

def test_a_name_given():
    assert two_fer("Alice") == "One for Alice, one for me."

def test_another_name_given():
    assert two_fer("Bob") == "One for Bob, one for me."''',
        "solution": '''def two_fer(name="you"):
    return f"One for {name}, one for me."''',
    },
    {
        "name": "leap",
        "instructions": """Determine if a year is a leap year.

A leap year occurs:
- On every year divisible by 4
- Except years divisible by 100
- Unless also divisible by 400
""",
        "test_code": '''def test_year_not_divisible_by_4():
    assert leap_year(2015) == False

def test_year_divisible_by_4_not_100():
    assert leap_year(2016) == True

def test_year_divisible_by_100_not_400():
    assert leap_year(1900) == False

def test_year_divisible_by_400():
    assert leap_year(2000) == True''',
        "solution": '''def leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)''',
    },
    {
        "name": "reverse-string",
        "instructions": """Reverse a string.

Example: "stressed" -> "desserts"
""",
        "test_code": '''def test_empty_string():
    assert reverse("") == ""

def test_a_word():
    assert reverse("robot") == "tobor"

def test_a_sentence():
    assert reverse("I'm hungry!") == "!yrgnuh m'I"''',
        "solution": '''def reverse(text):
    return text[::-1]''',
    },
    {
        "name": "isogram",
        "instructions": """Determine if a word is an isogram (no repeating letters).

The check should be case-insensitive and ignore hyphens and spaces.
""",
        "test_code": '''def test_empty_string():
    assert is_isogram("") == True

def test_isogram_with_only_lower_case():
    assert is_isogram("isogram") == True

def test_word_with_duplicated_letter():
    assert is_isogram("eleven") == False

def test_isogram_with_mixed_case():
    assert is_isogram("Alphabet") == False''',
        "solution": '''def is_isogram(string):
    cleaned = string.lower().replace("-", "").replace(" ", "")
    return len(cleaned) == len(set(cleaned))''',
    },
    {
        "name": "hamming",
        "instructions": """Calculate the Hamming distance between two DNA strands.

The Hamming distance is the number of positions at which the
corresponding symbols are different.
""",
        "test_code": '''def test_empty_strands():
    assert distance("", "") == 0

def test_single_letter_identical():
    assert distance("A", "A") == 0

def test_single_letter_different():
    assert distance("G", "T") == 1

def test_long_different_strands():
    assert distance("GGACG", "GGTCG") == 1''',
        "solution": '''def distance(strand_a, strand_b):
    if len(strand_a) != len(strand_b):
        raise ValueError("Strands must be of equal length.")
    return sum(a != b for a, b in zip(strand_a, strand_b))''',
    },
]


def create_exercism_task(output_dir, task_id, sample, prefix):
    instruction = f"""{sample['instructions']}

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "exercism", "exercise": sample["name"]},
        solution_files={"solution.py": sample["solution"]},
    )


# =============================================================================
# CodeNet Examples
# =============================================================================

CODENET_SAMPLES = [
    {
        "problem": """Problem: Sum of Two Numbers

Read two integers A and B, and output their sum.

Input: Two space-separated integers A and B (1 <= A, B <= 1000)
Output: A single integer, the sum of A and B

Example:
Input: 3 5
Output: 8""",
        "inputs": ["3 5", "100 200", "1 1"],
        "outputs": ["8", "300", "2"],
        "solution": """a, b = map(int, input().split())
print(a + b)""",
    },
    {
        "problem": """Problem: Product of Three Numbers

Read three integers and output their product.

Input: Three space-separated integers (1 <= each <= 100)
Output: The product

Example:
Input: 2 3 4
Output: 24""",
        "inputs": ["2 3 4", "1 1 1", "10 10 10"],
        "outputs": ["24", "1", "1000"],
        "solution": """nums = list(map(int, input().split()))
print(nums[0] * nums[1] * nums[2])""",
    },
    {
        "problem": """Problem: Even or Odd

Read an integer and output "Even" if it's even, "Odd" otherwise.

Input: A single integer N (1 <= N <= 10000)
Output: "Even" or "Odd"

Example:
Input: 4
Output: Even""",
        "inputs": ["4", "7", "100"],
        "outputs": ["Even", "Odd", "Even"],
        "solution": """n = int(input())
print("Even" if n % 2 == 0 else "Odd")""",
    },
    {
        "problem": """Problem: Absolute Value

Read an integer and output its absolute value.

Input: A single integer N (-10000 <= N <= 10000)
Output: |N|

Example:
Input: -5
Output: 5""",
        "inputs": ["-5", "3", "0"],
        "outputs": ["5", "3", "0"],
        "solution": """n = int(input())
print(abs(n))""",
    },
    {
        "problem": """Problem: String Length

Read a string and output its length.

Input: A single string (1 <= length <= 100)
Output: The length

Example:
Input: hello
Output: 5""",
        "inputs": ["hello", "a", "programming"],
        "outputs": ["5", "1", "11"],
        "solution": """s = input()
print(len(s))""",
    },
]


def create_codenet_task(output_dir, task_id, sample, prefix):
    instruction = f"""{sample['problem']}

Place your solution in /app/solution.py"""

    test_files = {}
    for i, (inp, out) in enumerate(zip(sample["inputs"], sample["outputs"])):
        test_files[f"inputs/input_{i}.txt"] = inp
        test_files[f"outputs/output_{i}.txt"] = out

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_io_test_sh(),
        test_files=test_files,
        dataset_prefix=prefix,
        metadata={"source": "codenet"},
        solution_files={"solution.py": sample["solution"]},
    )


# =============================================================================
# BugsInPy Examples
# =============================================================================

BUGSINPY_SAMPLES = [
    {
        "project": "example",
        "bug_id": "1",
        "buggy_code": '''def divide(a, b):
    return a / b  # Bug: no zero check''',
        "fixed_code": '''def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b''',
        "test": '''def test_divide_normal():
    assert divide(10, 2) == 5

def test_divide_by_zero():
    import pytest
    with pytest.raises(ValueError):
        divide(10, 0)''',
    },
    {
        "project": "example",
        "bug_id": "2",
        "buggy_code": '''def get_first(lst):
    return lst[0]  # Bug: no empty check''',
        "fixed_code": '''def get_first(lst):
    if not lst:
        return None
    return lst[0]''',
        "test": '''def test_get_first_normal():
    assert get_first([1, 2, 3]) == 1

def test_get_first_empty():
    assert get_first([]) is None''',
    },
    {
        "project": "example",
        "bug_id": "3",
        "buggy_code": '''def find_max(lst):
    max_val = 0  # Bug: wrong initial value
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val''',
        "fixed_code": '''def find_max(lst):
    if not lst:
        return None
    max_val = lst[0]
    for x in lst[1:]:
        if x > max_val:
            max_val = x
    return max_val''',
        "test": '''def test_find_max_positive():
    assert find_max([1, 5, 3]) == 5

def test_find_max_negative():
    assert find_max([-5, -2, -8]) == -2''',
    },
    {
        "project": "example",
        "bug_id": "4",
        "buggy_code": '''def concat_strings(a, b):
    return a + b  # Bug: no type check''',
        "fixed_code": '''def concat_strings(a, b):
    return str(a) + str(b)''',
        "test": '''def test_concat_strings():
    assert concat_strings("hello", "world") == "helloworld"

def test_concat_mixed():
    assert concat_strings("value:", 42) == "value:42"''',
    },
    {
        "project": "example",
        "bug_id": "5",
        "buggy_code": '''def is_even(n):
    return n % 2 == 1  # Bug: wrong comparison''',
        "fixed_code": '''def is_even(n):
    return n % 2 == 0''',
        "test": '''def test_is_even():
    assert is_even(4) == True
    assert is_even(3) == False
    assert is_even(0) == True''',
    },
]


def create_bugsinpy_task(output_dir, task_id, sample, prefix):
    instruction = f"""Fix the bug in the following Python code.

Project: {sample['project']}
Bug ID: {sample['bug_id']}

Buggy code:
```python
{sample['buggy_code']}
```

The code has a bug that causes some test cases to fail.
Fix the bug and place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "bugsinpy", "project": sample["project"], "bug_id": sample["bug_id"]},
        solution_files={"solution.py": sample["fixed_code"]},
    )


# =============================================================================
# QuixBugs Examples
# =============================================================================

QUIXBUGS_SAMPLES = [
    {
        "algorithm": "gcd",
        "buggy_code": '''def gcd(a, b):
    if b == 0:
        return a
    return gcd(a % b, b)''',
        "fixed_code": '''def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)''',
        "test_cases": [
            {"input": "(12, 8)", "expected": "4"},
            {"input": "(100, 25)", "expected": "25"},
            {"input": "(17, 13)", "expected": "1"},
        ],
    },
    {
        "algorithm": "bitcount",
        "buggy_code": '''def bitcount(n):
    count = 0
    while n:
        n ^= n - 1
        count += 1
    return count''',
        "fixed_code": '''def bitcount(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count''',
        "test_cases": [
            {"input": "127", "expected": "7"},
            {"input": "128", "expected": "1"},
            {"input": "0", "expected": "0"},
        ],
    },
    {
        "algorithm": "find_first_in_sorted",
        "buggy_code": '''def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr)
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return -1''',
        "fixed_code": '''def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(arr) and arr[lo] == x else -1''',
        "test_cases": [
            {"input": "([1, 2, 3, 4, 5], 3)", "expected": "2"},
            {"input": "([1, 2, 2, 2, 3], 2)", "expected": "1"},
            {"input": "([1, 2, 3], 4)", "expected": "-1"},
        ],
    },
    {
        "algorithm": "sqrt",
        "buggy_code": '''def sqrt(n, epsilon=0.01):
    approx = n / 2
    while abs(approx * approx - n) > epsilon:
        approx = 0.5 * (approx + n / approx)
    return approx''',
        "fixed_code": '''def sqrt(n, epsilon=0.01):
    if n < 0:
        raise ValueError("Cannot compute sqrt of negative number")
    if n == 0:
        return 0
    approx = n / 2
    while abs(approx * approx - n) > epsilon:
        approx = 0.5 * (approx + n / approx)
    return approx''',
        "test_cases": [
            {"input": "4", "expected": "~2.0"},
            {"input": "9", "expected": "~3.0"},
            {"input": "2", "expected": "~1.41"},
        ],
    },
    {
        "algorithm": "flatten",
        "buggy_code": '''def flatten(arr):
    result = []
    for x in arr:
        if isinstance(x, list):
            result.extend(x)
        else:
            result.append(x)
    return result''',
        "fixed_code": '''def flatten(arr):
    result = []
    for x in arr:
        if isinstance(x, list):
            result.extend(flatten(x))
        else:
            result.append(x)
    return result''',
        "test_cases": [
            {"input": "[[1, 2], [3, 4]]", "expected": "[1, 2, 3, 4]"},
            {"input": "[1, [2, [3, 4]]]", "expected": "[1, 2, 3, 4]"},
            {"input": "[]", "expected": "[]"},
        ],
    },
]


def create_quixbugs_task(output_dir, task_id, sample, prefix):
    instruction = f"""Fix the bug in this classic algorithm implementation.

Algorithm: {sample['algorithm']}

Buggy code:
```python
{sample['buggy_code']}
```

Test cases:
"""
    for tc in sample["test_cases"]:
        instruction += f"- Input: {tc['input']} -> Expected: {tc['expected']}\n"

    instruction += "\nPlace your fixed solution in /app/solution.py"

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import {sample['algorithm']}

"""
    for i, tc in enumerate(sample["test_cases"]):
        test_content += f"""
def test_case_{i}():
    result = {sample['algorithm']}{tc['input']}
    # Approximate check for floats
    expected = {tc['expected'].replace('~', '')}
    if isinstance(expected, float):
        assert abs(result - expected) < 0.1
    else:
        assert result == expected
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "quixbugs", "algorithm": sample["algorithm"]},
        solution_files={"solution.py": sample["fixed_code"]},
    )


# =============================================================================
# Methods2Test (Java) Examples
# =============================================================================

METHODS2TEST_SAMPLES = [
    {
        "focal": '''public static int add(int a, int b) {
    return a + b;
}''',
        "test": '''@Test
public void testAdd() {
    assertEquals(5, Calculator.add(2, 3));
    assertEquals(0, Calculator.add(-1, 1));
}''',
    },
    {
        "focal": '''public static String reverse(String s) {
    return new StringBuilder(s).reverse().toString();
}''',
        "test": '''@Test
public void testReverse() {
    assertEquals("olleh", StringUtils.reverse("hello"));
    assertEquals("", StringUtils.reverse(""));
}''',
    },
    {
        "focal": '''public static boolean isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}''',
        "test": '''@Test
public void testIsPrime() {
    assertTrue(MathUtils.isPrime(7));
    assertFalse(MathUtils.isPrime(4));
    assertFalse(MathUtils.isPrime(1));
}''',
    },
    {
        "focal": '''public static int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}''',
        "test": '''@Test
public void testFactorial() {
    assertEquals(120, MathUtils.factorial(5));
    assertEquals(1, MathUtils.factorial(0));
    assertEquals(1, MathUtils.factorial(1));
}''',
    },
    {
        "focal": '''public static int[] sort(int[] arr) {
    Arrays.sort(arr);
    return arr;
}''',
        "test": '''@Test
public void testSort() {
    assertArrayEquals(new int[]{1, 2, 3}, ArrayUtils.sort(new int[]{3, 1, 2}));
    assertArrayEquals(new int[]{}, ArrayUtils.sort(new int[]{}));
}''',
    },
]


def create_methods2test_task(output_dir, task_id, sample, prefix):
    # Extract method signature from focal code (first line typically)
    focal_code = sample['focal']
    lines = focal_code.strip().split('\n')
    signature = lines[0] if lines else "public void method()"

    instruction = f"""Implement a Java method with the following signature:

```java
{signature}
```

The method should pass all the provided test cases.

Place your solution in /app/src/main/java/Solution.java"""

    test_content = f"""import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

public class TestSolution {{
    {sample['test']}
}}
"""

    pom_xml = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.test</groupId>
    <artifactId>solution</artifactId>
    <version>1.0</version>
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""

    test_sh = '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cd /app
cp /tests/pom.xml . 2>/dev/null || true
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt
if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi
'''

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("java"),
        test_sh=test_sh,
        test_files={"TestSolution.java": test_content, "pom.xml": pom_xml},
        dataset_prefix=prefix,
        metadata={"source": "methods2test", "language": "java"},
    )


# =============================================================================
# pyMethods2Test Examples (Python focal-test pairs)
# =============================================================================

PYMETHODS2TEST_SAMPLES = [
    {
        "focal": '''def calculate_average(numbers: list) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)''',
        "test": '''def test_calculate_average():
    assert calculate_average([1, 2, 3, 4, 5]) == 3.0
    assert calculate_average([10]) == 10.0
    assert calculate_average([]) == 0.0''',
        "repo": "example/stats-utils",
    },
    {
        "focal": '''def find_duplicates(lst: list) -> list:
    """Find duplicate elements in a list."""
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)''',
        "test": '''def test_find_duplicates():
    assert set(find_duplicates([1, 2, 2, 3, 3, 3])) == {2, 3}
    assert find_duplicates([1, 2, 3]) == []
    assert find_duplicates([]) == []''',
        "repo": "example/list-utils",
    },
    {
        "focal": '''def merge_dicts(d1: dict, d2: dict) -> dict:
    """Merge two dictionaries."""
    result = d1.copy()
    result.update(d2)
    return result''',
        "test": '''def test_merge_dicts():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}
    assert merge_dicts({}, {}) == {}''',
        "repo": "example/dict-utils",
    },
    {
        "focal": '''def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())''',
        "test": '''def test_count_words():
    assert count_words("hello world") == 2
    assert count_words("one") == 1
    assert count_words("") == 0''',
        "repo": "example/text-utils",
    },
    {
        "focal": '''def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))''',
        "test": '''def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10''',
        "repo": "example/math-utils",
    },
]


def create_pymethods2test_task(output_dir, task_id, sample, prefix):
    # Extract function name and docstring from focal code
    focal_code = sample['focal']
    # Get function signature (first line)
    lines = focal_code.strip().split('\n')
    signature = lines[0] if lines else "def function():"
    # Get docstring if present
    docstring = ""
    if '"""' in focal_code:
        start = focal_code.find('"""') + 3
        end = focal_code.find('"""', start)
        if end > start:
            docstring = focal_code[start:end].strip()

    instruction = f"""Implement a Python function with the following signature:

```python
{signature}
```

{f"Description: {docstring}" if docstring else ""}

The function should pass all the provided test cases.

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "pymethods2test", "repo": sample.get("repo", "")},
        solution_files={"solution.py": sample["focal"]},
    )


# =============================================================================
# Defects4J Examples (Java bug fixes)
# =============================================================================

DEFECTS4J_SAMPLES = [
    {
        "project": "Math",
        "bug_id": "1",
        "buggy": '''public static double divide(double a, double b) {
    return a / b;  // Bug: no zero check
}''',
        "fixed": '''public static double divide(double a, double b) {
    if (b == 0) {
        throw new ArithmeticException("Division by zero");
    }
    return a / b;
}''',
        "test": '''@Test
public void testDivideNormal() {
    assertEquals(5.0, MathUtils.divide(10.0, 2.0), 0.001);
}

@Test(expected = ArithmeticException.class)
public void testDivideByZero() {
    MathUtils.divide(10.0, 0.0);
}''',
    },
    {
        "project": "Lang",
        "bug_id": "2",
        "buggy": '''public static boolean isEmpty(String str) {
    return str.length() == 0;  // Bug: no null check
}''',
        "fixed": '''public static boolean isEmpty(String str) {
    return str == null || str.length() == 0;
}''',
        "test": '''@Test
public void testIsEmptyNormal() {
    assertTrue(StringUtils.isEmpty(""));
    assertFalse(StringUtils.isEmpty("hello"));
}

@Test
public void testIsEmptyNull() {
    assertTrue(StringUtils.isEmpty(null));
}''',
    },
    {
        "project": "Collections",
        "bug_id": "3",
        "buggy": '''public static <T> T getFirst(List<T> list) {
    return list.get(0);  // Bug: no empty check
}''',
        "fixed": '''public static <T> T getFirst(List<T> list) {
    if (list == null || list.isEmpty()) {
        return null;
    }
    return list.get(0);
}''',
        "test": '''@Test
public void testGetFirstNormal() {
    List<String> list = Arrays.asList("a", "b", "c");
    assertEquals("a", CollectionUtils.getFirst(list));
}

@Test
public void testGetFirstEmpty() {
    assertNull(CollectionUtils.getFirst(new ArrayList<>()));
}''',
    },
    {
        "project": "Time",
        "bug_id": "4",
        "buggy": '''public static int daysBetween(Date d1, Date d2) {
    long diff = d2.getTime() - d1.getTime();
    return (int) (diff / (1000 * 60 * 60 * 24));  // Bug: rounding issues
}''',
        "fixed": '''public static int daysBetween(Date d1, Date d2) {
    long diff = Math.abs(d2.getTime() - d1.getTime());
    return (int) Math.round(diff / (1000.0 * 60 * 60 * 24));
}''',
        "test": '''@Test
public void testDaysBetween() {
    Calendar cal = Calendar.getInstance();
    Date d1 = cal.getTime();
    cal.add(Calendar.DAY_OF_MONTH, 5);
    Date d2 = cal.getTime();
    assertEquals(5, DateUtils.daysBetween(d1, d2));
}''',
    },
    {
        "project": "Closure",
        "bug_id": "5",
        "buggy": '''public static String capitalize(String str) {
    return str.substring(0, 1).toUpperCase() + str.substring(1);  // Bug: no empty check
}''',
        "fixed": '''public static String capitalize(String str) {
    if (str == null || str.isEmpty()) {
        return str;
    }
    return str.substring(0, 1).toUpperCase() + str.substring(1);
}''',
        "test": '''@Test
public void testCapitalizeNormal() {
    assertEquals("Hello", StringUtils.capitalize("hello"));
}

@Test
public void testCapitalizeEmpty() {
    assertEquals("", StringUtils.capitalize(""));
}''',
    },
]


def create_defects4j_task(output_dir, task_id, sample, prefix):
    instruction = f"""Fix the bug in the following Java code.

Project: {sample['project']}
Bug ID: {sample['bug_id']}

Buggy code:
```java
{sample['buggy']}
```

The code has a bug that causes test failures. Fix it and place your solution in /app/src/main/java/Solution.java"""

    test_content = f"""import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

public class TestSolution {{
    {sample['test']}
}}
"""

    pom_xml = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.test</groupId>
    <artifactId>solution</artifactId>
    <version>1.0</version>
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""

    test_sh = '''#!/bin/bash
set -e
mkdir -p /logs/verifier
cd /app
cp /tests/pom.xml . 2>/dev/null || true
mvn test -q 2>&1 | tee /logs/verifier/test_output.txt
if [ $? -eq 0 ]; then echo "1" > /logs/verifier/reward.txt; else echo "0" > /logs/verifier/reward.txt; fi
'''

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("java"),
        test_sh=test_sh,
        test_files={"TestSolution.java": test_content, "pom.xml": pom_xml},
        dataset_prefix=prefix,
        metadata={"source": "defects4j", "project": sample["project"], "bug_id": sample["bug_id"]},
        solution_files={"Solution.java": sample["fixed"]},
    )


# =============================================================================
# CoderEval Examples (Real-world coding tasks)
# =============================================================================

CODEREVAL_SAMPLES = [
    {
        "func_name": "parse_json_config",
        "docstring": "Parse a JSON configuration file and return a dictionary.",
        "code": '''def parse_json_config(filepath: str) -> dict:
    """Parse a JSON configuration file and return a dictionary."""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)''',
        "test": '''def test_parse_json_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"key": "value", "number": 42}')
    result = parse_json_config(str(config_file))
    assert result["key"] == "value"
    assert result["number"] == 42''',
    },
    {
        "func_name": "validate_email",
        "docstring": "Validate if a string is a valid email address.",
        "code": '''def validate_email(email: str) -> bool:
    """Validate if a string is a valid email address."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))''',
        "test": '''def test_validate_email():
    assert validate_email("test@example.com") == True
    assert validate_email("invalid") == False
    assert validate_email("user@domain.org") == True
    assert validate_email("@missing.com") == False''',
    },
    {
        "func_name": "retry_with_backoff",
        "docstring": "Retry a function with exponential backoff.",
        "code": '''def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry a function with exponential backoff."""
    import time
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(base_delay * (2 ** attempt))''',
        "test": '''def test_retry_with_backoff():
    counter = {"value": 0}
    def flaky_func():
        counter["value"] += 1
        if counter["value"] < 3:
            raise ValueError("Fail")
        return "success"
    result = retry_with_backoff(flaky_func, max_retries=5, base_delay=0.01)
    assert result == "success"
    assert counter["value"] == 3''',
    },
    {
        "func_name": "flatten_nested_dict",
        "docstring": "Flatten a nested dictionary with dot notation keys.",
        "code": '''def flatten_nested_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)''',
        "test": '''def test_flatten_nested_dict():
    nested = {"a": {"b": {"c": 1}}, "d": 2}
    result = flatten_nested_dict(nested)
    assert result == {"a.b.c": 1, "d": 2}''',
    },
    {
        "func_name": "chunk_list",
        "docstring": "Split a list into chunks of specified size.",
        "code": '''def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]''',
        "test": '''def test_chunk_list():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk_list([1, 2, 3], 3) == [[1, 2, 3]]
    assert chunk_list([], 2) == []''',
    },
]


def create_codereval_task(output_dir, task_id, sample, prefix):
    instruction = f"""Implement the function `{sample['func_name']}`.

Description: {sample['docstring']}

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "codereval", "func_name": sample["func_name"]},
        solution_files={"solution.py": sample["code"]},
    )


# =============================================================================
# TravisTorrent Examples (CI build mining)
# =============================================================================

TRAVISTORRENT_SAMPLES = [
    {
        "test_code": '''def test_user_registration():
    """Test user registration workflow."""
    user = register_user("testuser", "test@example.com", "password123")
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active == True

def test_user_login():
    """Test user login."""
    token = login("testuser", "password123")
    assert token is not None
    assert len(token) > 0''',
        "repo": "example/user-service",
        "framework": "pytest",
    },
    {
        "test_code": '''def test_cart_add_item():
    """Test adding items to shopping cart."""
    cart = ShoppingCart()
    cart.add_item("item1", 2, 10.99)
    assert len(cart.items) == 1
    assert cart.total == 21.98

def test_cart_remove_item():
    """Test removing items from cart."""
    cart = ShoppingCart()
    cart.add_item("item1", 1, 10.00)
    cart.remove_item("item1")
    assert len(cart.items) == 0''',
        "repo": "example/ecommerce",
        "framework": "pytest",
    },
    {
        "test_code": '''def test_api_get_users():
    """Test GET /api/users endpoint."""
    response = client.get("/api/users")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_create_user():
    """Test POST /api/users endpoint."""
    response = client.post("/api/users", json={"name": "John"})
    assert response.status_code == 201''',
        "repo": "example/rest-api",
        "framework": "pytest",
    },
    {
        "test_code": '''def test_database_connection():
    """Test database connection."""
    db = Database("test.db")
    assert db.is_connected() == True
    db.close()

def test_database_query():
    """Test database query."""
    db = Database("test.db")
    results = db.query("SELECT * FROM users")
    assert isinstance(results, list)''',
        "repo": "example/db-utils",
        "framework": "pytest",
    },
    {
        "test_code": '''def test_cache_set_get():
    """Test cache set and get."""
    cache = Cache()
    cache.set("key", "value", ttl=60)
    assert cache.get("key") == "value"

def test_cache_expiry():
    """Test cache expiry."""
    cache = Cache()
    cache.set("key", "value", ttl=0)
    assert cache.get("key") is None''',
        "repo": "example/cache-lib",
        "framework": "pytest",
    },
]


def create_travistorrent_task(output_dir, task_id, sample, prefix):
    instruction = f"""Implement the code to make the following tests pass.

Repository: {sample['repo']}
Test Framework: {sample['framework']}

```python
{sample['test_code']}
```

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "travistorrent", "repo": sample["repo"], "framework": sample["framework"]},
    )


# =============================================================================
# GitHub Actions Examples (CI workflow mining)
# =============================================================================

GHACTIONS_SAMPLES = [
    {
        "test_code": '''def test_markdown_parser():
    """Test markdown to HTML conversion."""
    html = parse_markdown("# Hello\n\nWorld")
    assert "<h1>Hello</h1>" in html
    assert "<p>World</p>" in html

def test_markdown_links():
    """Test markdown link parsing."""
    html = parse_markdown("[link](https://example.com)")
    assert '<a href="https://example.com">link</a>' in html''',
        "repo": "example/markdown-parser",
        "language": "python",
    },
    {
        "test_code": '''def test_logger_info():
    """Test logger info level."""
    logger = Logger()
    logger.info("Test message")
    assert "INFO" in logger.get_logs()

def test_logger_error():
    """Test logger error level."""
    logger = Logger()
    logger.error("Error message")
    assert "ERROR" in logger.get_logs()''',
        "repo": "example/logging-lib",
        "language": "python",
    },
    {
        "test_code": '''def test_config_loader():
    """Test configuration loading."""
    config = load_config("test.yaml")
    assert config is not None
    assert "settings" in config

def test_config_defaults():
    """Test default configuration."""
    config = load_config_with_defaults({})
    assert config["timeout"] == 30''',
        "repo": "example/config-utils",
        "language": "python",
    },
    {
        "test_code": '''def test_rate_limiter():
    """Test rate limiter."""
    limiter = RateLimiter(max_requests=5, window_seconds=1)
    for _ in range(5):
        assert limiter.allow() == True
    assert limiter.allow() == False

def test_rate_limiter_reset():
    """Test rate limiter reset."""
    limiter = RateLimiter(max_requests=1, window_seconds=0)
    limiter.allow()
    import time; time.sleep(0.1)
    assert limiter.allow() == True''',
        "repo": "example/rate-limiter",
        "language": "python",
    },
    {
        "test_code": '''def test_queue_push_pop():
    """Test queue operations."""
    q = Queue()
    q.push(1)
    q.push(2)
    assert q.pop() == 1
    assert q.pop() == 2

def test_queue_empty():
    """Test empty queue."""
    q = Queue()
    assert q.is_empty() == True
    q.push(1)
    assert q.is_empty() == False''',
        "repo": "example/data-structures",
        "language": "python",
    },
]


def create_ghactions_task(output_dir, task_id, sample, prefix):
    instruction = f"""Implement the code to make the following tests pass.

Repository: {sample['repo']}
Language: {sample['language']}

```python
{sample['test_code']}
```

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "ghactions", "repo": sample["repo"], "language": sample["language"]},
    )


# =============================================================================
# Software Heritage Examples (Archive mining)
# =============================================================================

SOFTWAREHERITAGE_SAMPLES = [
    {
        "test_code": '''def test_binary_search():
    """Test binary search implementation."""
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert binary_search(arr, 5) == 4
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 10) == 9
    assert binary_search(arr, 11) == -1''',
        "path": "algorithms/test_search.py",
        "repo": "example/algorithms",
    },
    {
        "test_code": '''def test_linked_list_append():
    """Test linked list append."""
    ll = LinkedList()
    ll.append(1)
    ll.append(2)
    ll.append(3)
    assert ll.to_list() == [1, 2, 3]

def test_linked_list_prepend():
    """Test linked list prepend."""
    ll = LinkedList()
    ll.prepend(1)
    ll.prepend(2)
    assert ll.to_list() == [2, 1]''',
        "path": "data_structures/test_linked_list.py",
        "repo": "example/data-structures",
    },
    {
        "test_code": '''def test_stack_operations():
    """Test stack push/pop."""
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    assert stack.pop() == 3
    assert stack.pop() == 2
    assert stack.peek() == 1''',
        "path": "data_structures/test_stack.py",
        "repo": "example/data-structures",
    },
    {
        "test_code": '''def test_tree_traversal():
    """Test binary tree traversal."""
    tree = BinaryTree()
    tree.insert(5)
    tree.insert(3)
    tree.insert(7)
    assert tree.inorder() == [3, 5, 7]
    assert tree.preorder() == [5, 3, 7]''',
        "path": "data_structures/test_tree.py",
        "repo": "example/data-structures",
    },
    {
        "test_code": '''def test_graph_bfs():
    """Test graph BFS."""
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    assert graph.bfs(0) == [0, 1, 2, 3]

def test_graph_dfs():
    """Test graph DFS."""
    graph = Graph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    assert graph.dfs(0) == [0, 1, 3, 2]''',
        "path": "data_structures/test_graph.py",
        "repo": "example/data-structures",
    },
]


def create_softwareheritage_task(output_dir, task_id, sample, prefix):
    instruction = f"""Implement the code to make the following tests pass.

Source: Software Heritage Archive
Original Path: {sample['path']}
Repository: {sample['repo']}

```python
{sample['test_code']}
```

Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "softwareheritage", "path": sample["path"], "repo": sample["repo"]},
    )


# =============================================================================
# SWE-bench Examples (Multi-file GitHub Issues) - HARD
# =============================================================================

SWEBENCH_SAMPLES = [
    {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "problem_statement": """QuerySet.update() doesn't work properly with F() expressions on related fields.

When using F() expressions in QuerySet.update() with related fields, the generated SQL
is incorrect and causes the update to fail silently or update wrong values.

Example:
```python
MyModel.objects.filter(pk=1).update(related_count=F('related_field__count'))
```

Expected: Updates related_count with the value from related_field.count
Actual: Generates invalid SQL or updates with wrong value
""",
        "test_patch": '''def test_update_with_related_f_expression():
    """Test F() expressions work with related fields in update()."""
    parent = Parent.objects.create(count=10)
    child = Child.objects.create(parent=parent, value=0)

    Child.objects.filter(pk=child.pk).update(value=F('parent__count'))
    child.refresh_from_db()
    assert child.value == 10
''',
        "patch": "# Fix for F() expression handling in update queries",
    },
    {
        "instance_id": "requests__requests-5678",
        "repo": "psf/requests",
        "problem_statement": """Session cookies not being sent with redirected requests.

When a request is redirected to a different path on the same domain, session cookies
are not being included in the redirected request.

Steps to reproduce:
1. Create a Session
2. Set a cookie for domain.com
3. Make request to domain.com/path1 which redirects to domain.com/path2
4. Observe that cookies are missing from the redirected request
""",
        "test_patch": '''def test_session_cookies_on_redirect():
    """Test cookies are sent with redirected requests."""
    s = Session()
    s.cookies.set('test', 'value', domain='httpbin.org')
    response = s.get('https://httpbin.org/redirect-to?url=/cookies')
    assert 'test' in response.json().get('cookies', {})
''',
        "patch": "# Fix for cookie handling in redirects",
    },
]


def create_swebench_task(output_dir, task_id, sample, prefix):
    instruction = f"""# GitHub Issue: {sample['repo']}

## Problem Statement

{sample['problem_statement']}

## Your Task

Fix this issue. The solution should:
1. Address all aspects of the issue description
2. Pass the provided tests
3. Not break any other functionality

Place your solution in /app/"""

    test_content = f"""import pytest

{sample['test_patch']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={
            "source": "swebench",
            "instance_id": sample["instance_id"],
            "repo": sample["repo"],
        },
        solution_files={"solution_hint.patch": sample["patch"]},
    )


# =============================================================================
# BigCodeBench Examples (Multi-library Tasks) - HARD
# =============================================================================

BIGCODEBENCH_SAMPLES = [
    {
        "task_id": "BigCodeBench/1",
        "instruct_prompt": """Write a function that reads a CSV file, filters rows where the 'age' column
is greater than a given threshold, and returns a pandas DataFrame with the filtered results.
The function should handle missing values by dropping rows with NaN in the 'age' column.""",
        "test": '''def test_filter_csv_by_age():
    import pandas as pd
    import tempfile
    import os

    # Create test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("name,age\\nAlice,25\\nBob,30\\nCharlie,\\nDiana,35\\n")
        temp_path = f.name

    try:
        result = filter_csv_by_age(temp_path, 28)
        assert len(result) == 2
        assert list(result['name']) == ['Bob', 'Diana']
    finally:
        os.unlink(temp_path)
''',
        "canonical_solution": '''import pandas as pd

def filter_csv_by_age(filepath: str, threshold: int) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['age'])
    return df[df['age'] > threshold]
''',
        "libs": ["pandas"],
    },
    {
        "task_id": "BigCodeBench/2",
        "instruct_prompt": """Create a function that downloads an image from a URL, resizes it to the
specified dimensions using PIL, and saves it to the given output path.""",
        "test": '''def test_download_and_resize_image():
    import os
    import tempfile
    from PIL import Image

    # Use a small test image
    url = "https://httpbin.org/image/png"
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "resized.png")
        download_and_resize_image(url, output_path, 50, 50)

        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.size == (50, 50)
''',
        "canonical_solution": '''import requests
from PIL import Image
from io import BytesIO

def download_and_resize_image(url: str, output_path: str, width: int, height: int):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((width, height))
    img.save(output_path)
''',
        "libs": ["requests", "pillow"],
    },
]


def create_bigcodebench_task(output_dir, task_id, sample, prefix):
    libs = sample.get("libs", [])
    instruction = f"""# Task

{sample['instruct_prompt']}

## Required Libraries

This task requires: {', '.join(libs) if libs else 'Standard library only'}

## Instructions

1. Place your solution in `/app/solution.py`
2. Export the required function(s)
3. Your solution should pass all test cases
"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "bigcodebench", "task_id": sample["task_id"], "libs": libs},
        solution_files={"solution.py": sample["canonical_solution"]},
    )


# =============================================================================
# CrossCodeEval Examples (Cross-file Context) - HARD
# =============================================================================

CROSSCODEEVAL_SAMPLES = [
    {
        "task_id": "cross-001",
        "prompt": '''class UserService:
    def __init__(self, db: Database):
        self.db = db

    def get_user(self, user_id: int):
        # TODO: Implement using self.db methods
        pass
''',
        "cross_context": [
            '''class Database:
    def query(self, table: str, conditions: dict) -> list:
        """Query a table with given conditions."""
        pass

    def get_by_id(self, table: str, id: int) -> dict:
        """Get a single record by ID."""
        pass
''',
        ],
        "groundtruth": '''class UserService:
    def __init__(self, db: Database):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.get_by_id('users', user_id)
''',
        "test": '''def test_get_user():
    class MockDB:
        def get_by_id(self, table, id):
            if table == 'users' and id == 1:
                return {'id': 1, 'name': 'Alice'}
            return None

    service = UserService(MockDB())
    user = service.get_user(1)
    assert user['name'] == 'Alice'
''',
    },
    {
        "task_id": "cross-002",
        "prompt": '''class OrderProcessor:
    def __init__(self, inventory: InventoryManager, payment: PaymentGateway):
        self.inventory = inventory
        self.payment = payment

    def process_order(self, order: dict) -> bool:
        # TODO: Check inventory, process payment, update stock
        pass
''',
        "cross_context": [
            '''class InventoryManager:
    def check_stock(self, item_id: str) -> int:
        """Returns available quantity."""
        pass

    def reserve_stock(self, item_id: str, quantity: int) -> bool:
        """Reserve stock for an order."""
        pass
''',
            '''class PaymentGateway:
    def charge(self, amount: float, card_token: str) -> bool:
        """Process payment. Returns True if successful."""
        pass
''',
        ],
        "groundtruth": '''class OrderProcessor:
    def __init__(self, inventory: InventoryManager, payment: PaymentGateway):
        self.inventory = inventory
        self.payment = payment

    def process_order(self, order: dict) -> bool:
        item_id = order['item_id']
        quantity = order['quantity']

        if self.inventory.check_stock(item_id) < quantity:
            return False

        if not self.payment.charge(order['total'], order['card_token']):
            return False

        return self.inventory.reserve_stock(item_id, quantity)
''',
        "test": '''def test_process_order():
    class MockInventory:
        def check_stock(self, item_id): return 10
        def reserve_stock(self, item_id, qty): return True

    class MockPayment:
        def charge(self, amount, token): return True

    processor = OrderProcessor(MockInventory(), MockPayment())
    result = processor.process_order({
        'item_id': 'SKU001',
        'quantity': 2,
        'total': 29.99,
        'card_token': 'tok_test'
    })
    assert result == True
''',
    },
]


def create_crosscodeeval_task(output_dir, task_id, sample, prefix):
    cross_context = sample.get("cross_context", [])

    instruction = f"""# Cross-File Code Completion Task

## Context (Code to Complete)

```python
{sample['prompt']}
```

"""
    if cross_context:
        instruction += "## Cross-File Context\n\n"
        for i, ctx in enumerate(cross_context):
            instruction += f"### Related File {i + 1}\n\n```python\n{ctx}\n```\n\n"

    instruction += """## Your Task

Complete the code above using the cross-file context provided.
Place your solution in `/app/solution.py`
"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={"source": "crosscodeeval", "task_id": sample["task_id"]},
        solution_files={"solution.py": sample["groundtruth"]},
    )


# =============================================================================
# CodeElo Examples (Competition-Level Algorithms) - VERY HARD
# =============================================================================

CODEELO_SAMPLES = [
    {
        "problem_id": "CF-1234A",
        "description": """Two Sum with Constraints

Given an array of n integers and a target sum S, find two distinct indices i and j
such that a[i] + a[j] = S and i < j. If multiple solutions exist, output the one
with the smallest i, then smallest j.

Input:
- Line 1: Two integers n (2 <= n <= 10^5) and S (-10^9 <= S <= 10^9)
- Line 2: n space-separated integers (-10^9 <= a[i] <= 10^9)

Output:
- Two space-separated integers i and j (1-indexed)
- If no solution exists, output "-1 -1"

Example:
Input:
5 8
2 7 11 15 1
Output:
1 2
""",
        "inputs": ["5 8\n2 7 11 15 1", "4 10\n1 2 3 4", "3 100\n1 2 3"],
        "outputs": ["1 2", "-1 -1", "-1 -1"],
        "difficulty": 1200,
        "tags": ["two pointers", "hash map"],
    },
    {
        "problem_id": "CF-5678B",
        "description": """Longest Increasing Subsequence Length

Given an array of n integers, find the length of the longest strictly increasing
subsequence.

Input:
- Line 1: Integer n (1 <= n <= 10^5)
- Line 2: n space-separated integers

Output:
- A single integer, the length of the LIS

Example:
Input:
6
10 9 2 5 3 7
Output:
4
""",
        "inputs": ["6\n10 9 2 5 3 7", "5\n1 2 3 4 5", "4\n4 3 2 1"],
        "outputs": ["4", "5", "1"],
        "difficulty": 1600,
        "tags": ["dp", "binary search"],
    },
]


def create_codeelo_task(output_dir, task_id, sample, prefix):
    instruction = f"""# Problem: {sample['problem_id']}

## Difficulty: {sample['difficulty']} (Elo rating)

## Tags: {', '.join(sample.get('tags', []))}

## Problem Statement

{sample['description']}

## Your Task

Implement a solution that reads from stdin and writes to stdout.
Place your solution in `/app/solution.py`
"""

    test_files = {}
    for i, (inp, out) in enumerate(zip(sample["inputs"], sample["outputs"])):
        test_files[f"inputs/input_{i}.txt"] = inp
        test_files[f"outputs/output_{i}.txt"] = out

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_io_test_sh(),
        test_files=test_files,
        dataset_prefix=prefix,
        metadata={
            "source": "codeelo",
            "problem_id": sample["problem_id"],
            "difficulty": sample["difficulty"],
            "tags": sample.get("tags", []),
        },
    )


# =============================================================================
# BugsInPy-MF Examples (Multi-Fault Bugs) - HARD
# =============================================================================

BUGSINPY_MF_SAMPLES = [
    {
        "bug_id": "mf-001",
        "project": "calculator",
        "buggy_code": '''def add(a, b):
    return a - b  # Bug 1: wrong operator

def multiply(a, b):
    return a + b  # Bug 2: wrong operator

def divide(a, b):
    return a * b  # Bug 3: wrong operator
''',
        "fixed_code": '''def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
''',
        "test_code": '''def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(2, 3) == 6

def test_divide():
    assert divide(6, 2) == 3
''',
        "fault_count": 3,
    },
    {
        "bug_id": "mf-002",
        "project": "string-utils",
        "buggy_code": '''def reverse_string(s):
    return s  # Bug 1: not reversing

def is_palindrome(s):
    return s == s  # Bug 2: comparing with self
''',
        "fixed_code": '''def reverse_string(s):
    return s[::-1]

def is_palindrome(s):
    return s == s[::-1]
''',
        "test_code": '''def test_reverse():
    assert reverse_string("hello") == "olleh"

def test_palindrome():
    assert is_palindrome("radar") == True
    assert is_palindrome("hello") == False
''',
        "fault_count": 2,
    },
]


def create_bugsinpy_mf_task(output_dir, task_id, sample, prefix):
    instruction = f"""# Multi-Fault Bug Fix Task

## Project: {sample['project']}
## Bug ID: {sample['bug_id']}
## Fault Count: {sample['fault_count']} bugs to fix

## Buggy Code

```python
{sample['buggy_code']}
```

## Your Task

This code contains **{sample['fault_count']} interacting bugs**.
1. Identify all bugs
2. Fix each bug
3. Ensure all tests pass

Place your fixed code in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={
            "source": "bugsinpy-mf",
            "bug_id": sample["bug_id"],
            "fault_count": sample["fault_count"],
        },
        solution_files={"solution.py": sample["fixed_code"]},
    )


# =============================================================================
# BugSwarm Examples (CI Environment Pairs) - HARD
# =============================================================================

BUGSWARM_SAMPLES = [
    {
        "artifact_id": "example-python-12345",
        "repo": "example/api-client",
        "language": "python",
        "failing_code": '''def make_request(url, timeout=30):
    import requests
    response = requests.get(url, timeout=timeout)
    return response.json()  # Bug: no status check
''',
        "passing_code": '''def make_request(url, timeout=30):
    import requests
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()
''',
        "test_code": '''def test_make_request_success():
    result = make_request("https://httpbin.org/json")
    assert isinstance(result, dict)

def test_make_request_error():
    import pytest
    with pytest.raises(Exception):
        make_request("https://httpbin.org/status/404")
''',
    },
    {
        "artifact_id": "example-python-67890",
        "repo": "example/data-utils",
        "language": "python",
        "failing_code": '''def safe_divide(a, b):
    return a / b  # Bug: no zero check
''',
        "passing_code": '''def safe_divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
''',
        "test_code": '''def test_safe_divide():
    assert safe_divide(10, 2) == 5

def test_safe_divide_by_zero():
    import pytest
    with pytest.raises(ZeroDivisionError):
        safe_divide(10, 0)
''',
    },
]


def create_bugswarm_task(output_dir, task_id, sample, prefix):
    instruction = f"""# CI Bug Fix Task (BugSwarm)

## Repository: {sample['repo']}
## Artifact ID: {sample['artifact_id']}
## Language: {sample['language']}

## Failing Code

```python
{sample['failing_code']}
```

## Your Task

Fix the bug to make the CI tests pass.
Place your solution in /app/solution.py"""

    test_content = f"""import pytest
import sys
sys.path.insert(0, '/app')

from solution import *

{sample['test_code']}
"""

    create_harbor_task_directory_generic(
        output_dir=output_dir,
        task_id=task_id,
        instruction=instruction,
        dockerfile=get_dockerfile("python"),
        test_sh=create_pytest_test_sh(),
        test_files={"test_solution.py": test_content},
        dataset_prefix=prefix,
        metadata={
            "source": "bugswarm",
            "artifact_id": sample["artifact_id"],
            "repo": sample["repo"],
        },
        solution_files={"solution.py": sample["passing_code"]},
    )


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate example tasks for all datasets."""
    print("=" * 60)
    print("Generating example tasks for all datasets")
    print("=" * 60)
    print()

    # Create examples directory
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Generate examples for each dataset
    datasets = [
        # Focal-Test Pairs (HIGH PRIORITY)
        ("unitsyn-python", UNITSYN_SAMPLES, create_unitsyn_task),
        ("pymethods2test", PYMETHODS2TEST_SAMPLES, create_pymethods2test_task),
        ("methods2test", METHODS2TEST_SAMPLES, create_methods2test_task),
        # Competitive Programming
        ("codenet-python", CODENET_SAMPLES, create_codenet_task),
        ("taco", TACO_SAMPLES, create_taco_task),
        ("exercism-python", EXERCISM_SAMPLES, create_exercism_task),
        # Bug Benchmarks
        ("defects4j", DEFECTS4J_SAMPLES, create_defects4j_task),
        ("bugsinpy", BUGSINPY_SAMPLES, create_bugsinpy_task),
        ("quixbugs-python", QUIXBUGS_SAMPLES, create_quixbugs_task),
        ("codereval-python", CODEREVAL_SAMPLES, create_codereval_task),
        # Raw Mining
        ("travistorrent", TRAVISTORRENT_SAMPLES, create_travistorrent_task),
        ("ghactions", GHACTIONS_SAMPLES, create_ghactions_task),
        ("softwareheritage", SOFTWAREHERITAGE_SAMPLES, create_softwareheritage_task),
        # Advanced/Complex Datasets (HARD)
        ("swebench", SWEBENCH_SAMPLES, create_swebench_task),
        ("bigcodebench", BIGCODEBENCH_SAMPLES, create_bigcodebench_task),
        ("crosscodeeval", CROSSCODEEVAL_SAMPLES, create_crosscodeeval_task),
        ("codeelo", CODEELO_SAMPLES, create_codeelo_task),
        ("bugsinpy-mf", BUGSINPY_MF_SAMPLES, create_bugsinpy_mf_task),
        ("bugswarm", BUGSWARM_SAMPLES, create_bugswarm_task),
    ]

    created_dirs = []
    for name, samples, create_fn in datasets:
        output_dir = create_example_tasks(name, samples, create_fn)
        created_dirs.append(output_dir)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    for d in created_dirs:
        task_count = len(list(d.iterdir()))
        print(f"  {d.name}: {task_count} tasks")

    print()
    print(f"All examples created in: {OUTPUT_BASE}")
    print()

    # Show structure of first task
    first_task = list(created_dirs[0].iterdir())[0]
    print("Example task structure:")
    for item in sorted(first_task.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(first_task)
            print(f"  {rel_path}")


if __name__ == "__main__":
    main()
