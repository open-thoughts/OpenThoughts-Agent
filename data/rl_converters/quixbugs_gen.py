#!/usr/bin/env python3
"""Generate per-task ``tests/test_solution.py`` + gold ``solution/solution.py``
for the 40 QuixBugs-Python algorithms.

Expected outputs are computed at build time from the textbook-correct
references in :mod:`quixbugs_correct` and baked as literals, so the verifier
only needs the agent's ``/app/solution.py`` plus (for graph/linked tasks) a
``Node`` fixture helper.
"""
from __future__ import annotations

from pathlib import Path

from . import quixbugs_correct as qc


# Algorithm order matches the exp_rpt_quixbugs-python parquet exactly.
ALGOS = [
    "bitcount", "breadth_first_search", "bucketsort", "depth_first_search",
    "detect_cycle", "find_first_in_sorted", "find_in_sorted", "flatten",
    "gcd", "get_factors", "hanoi", "is_valid_parenthesization",
    "kheapsort", "knapsack", "kth", "lcs_length", "levenshtein", "lis",
    "longest_common_subsequence", "max_sublist_sum", "mergesort",
    "minimum_spanning_tree", "next_palindrome", "next_permutation", "pascal",
    "possible_change", "powerset", "quicksort", "reverse_linked_list",
    "rpn_eval", "shortest_path_length", "shortest_path_lengths",
    "shortest_paths", "shunting_yard", "sieve", "sqrt", "subsequences",
    "to_base", "topological_ordering", "wrap",
]


NODE_HELPER = '''\
class Node:
    """Minimal node: graph algorithms read .successors / .successor /
    .incoming_nodes / .outgoing_nodes and carry an optional .label."""
    def __init__(self, label=None, successors=None, successor=None,
                 incoming_nodes=None, outgoing_nodes=None):
        self.label = label
        self.successors = successors if successors is not None else []
        self.successor = successor
        self.incoming_nodes = incoming_nodes if incoming_nodes is not None else set()
        self.outgoing_nodes = outgoing_nodes if outgoing_nodes is not None else []
'''


# Pure (non-graph) inputs: algo -> list of arg tuples.
PURE_INPUTS = {
    "bitcount": [(0,), (1,), (7,), (127,), (255,), (13,), (3005,), (1023,)],
    "bucketsort": [([3, 1, 4, 1, 5, 9, 2, 6], 10), ([5, 5, 5], 6), ([], 3), ([0, 1, 2], 3)],
    "find_first_in_sorted": [([1, 2, 3, 4, 5], 3), ([1, 1, 2, 2, 3, 3], 2), ([1, 2, 3, 4], 5), ([2, 4, 6, 8], 2)],
    "find_in_sorted": [([1, 2, 3, 4, 5, 6, 7], 5), ([1, 2, 3, 4, 6, 7, 8], 5), ([2, 4, 6, 8, 10], 8), ([], 1)],
    "flatten": [([1, [2, [3, 4], 5], 6],), ([[1, 2], [3, 4]],), ([],), ([1, 2, 3],)],
    "gcd": [(17, 0), (13, 13), (37, 600), (20, 100), (624129, 2061517), (3, 12)],
    "get_factors": [(1,), (2,), (12,), (100,), (17,), (64,)],
    "hanoi": [(0,), (1,), (2,), (3,)],
    "is_valid_parenthesization": [("(()())",), ("())(",), ("((()))",), ("",), ("(",)],
    "kheapsort": [([3, 1, 4, 1, 5, 9, 2, 6], 3), ([1, 2, 3], 1), ([5, 4, 3, 2, 1], 2)],
    "knapsack": [(5, [(1, 2), (2, 3), (3, 4)]), (0, [(1, 1)]), (10, [(5, 10), (4, 7), (6, 8)])],
    "kth": [([1, 2, 3, 4, 5], 0), ([3, 1, 4, 1, 5, 9], 2), ([5, 4, 3, 2, 1], 3), ([2, 1], 1)],
    "lcs_length": [("abcde", "ace"), ("abc", "abc"), ("", "abc"), ("aaaa", "aa")],
    "levenshtein": [("kitten", "sitting"), ("", "abc"), ("abc", "abc"), ("flaw", "lawn")],
    "lis": [([1, 2, 3, 4, 5],), ([5, 4, 3, 2, 1],), ([3, 1, 4, 1, 5, 9, 2, 6],), ([],)],
    "longest_common_subsequence": [("abcde", "ace"), ("abc", "def"), ("aaaa", "aa")],
    "max_sublist_sum": [([-2, 1, -3, 4, -1, 2, 1, -5, 4],), ([1, 2, 3, 4],), ([-1, -2, -3],)],
    "mergesort": [([3, 1, 4, 1, 5, 9, 2, 6],), ([],), ([1],), ([5, 4, 3, 2, 1],)],
    "next_palindrome": [([1, 2, 3],), ([9, 9, 9],), ([1, 2, 9],), ([8, 9, 9, 9],), ([1],)],
    "next_permutation": [([1, 2, 3],), ([3, 2, 1],), ([1, 1, 5],), ([1, 3, 2],)],
    "pascal": [(1,), (2,), (3,), (5,)],
    "possible_change": [([1, 5, 10], 5), ([1, 2, 5], 10), ([5], 0), ([], 3)],
    "powerset": [([1, 2, 3],), ([],), ([1],)],
    "quicksort": [([3, 1, 4, 1, 5, 9, 2, 6],), ([],), ([1],), ([5, 4, 3, 2, 1],)],
    "rpn_eval": [([3.0, 4.0, "+"],), ([7.0, 2.0, "-", 3.0, "+"],), ([5.0, 1.0, 2.0, "+", "*"],), ([3.0, 4.0, "/"],)],
    "shortest_path_lengths": [
        (3, {(0, 1): 1, (1, 2): 2}),
        (4, {(0, 1): 5, (1, 2): 3, (0, 3): 10, (2, 3): 1}),
    ],
    "shortest_paths": [
        ("a", {("a", "b"): 1, ("b", "c"): 2}),
        ("x", {("x", "y"): 5, ("y", "z"): 3, ("x", "z"): 20}),
    ],
    "minimum_spanning_tree": [
        ({("a", "b"): 1, ("b", "c"): 2, ("a", "c"): 3},),
        ({("a", "b"): 4, ("c", "d"): 1, ("b", "c"): 2},),
    ],
    "shunting_yard": [([3, "+", 4],), ([3, "+", 4, "*", 2],), ([2, "*", 3, "+", 4],)],
    "sieve": [(2,), (10,), (20,), (30,)],
    "sqrt": [(2, 0.01), (4, 0.2), (27, 0.01), (170, 0.03)],
    "subsequences": [(1, 3, 2), (0, 4, 2), (1, 5, 3)],
    "to_base": [(10, 2), (255, 16), (100, 8), (31, 2)],
    "wrap": [("hello world foo bar", 10), ("abcdefgh", 3), ("a b c d", 2)],
}


def _lit(v) -> str:
    """Render a value as a Python literal for embedding in the test file."""
    return repr(v)


def _dump_cases(cases) -> str:
    """repr() a case list, rendering float('inf')/'-inf'/'nan' portably."""
    import re
    text = repr(cases)
    text = re.sub(r"(?<![A-Za-z0-9_.'\)])inf(?![A-Za-z0-9_])", "float('inf')", text)
    text = re.sub(r"(?<![A-Za-z0-9_.])nan(?![A-Za-z0-9_])", "float('nan')", text)
    return text


def _normalize(name, value):
    """Convert a reference output to a comparable literal form."""
    if name in ("kheapsort", "flatten"):
        return list(value)
    if name == "shortest_path_lengths":
        return {tuple(k): v for k, v in value.items()}
    if name == "shortest_paths":
        return dict(value)
    if name == "minimum_spanning_tree":
        return set(tuple(e) for e in value)
    return value


def _normalize_expr(name, var):
    """Python expression to normalize the agent's raw output before compare."""
    if name in ("kheapsort", "flatten"):
        return f"list({var})"
    if name == "shortest_path_lengths":
        return f"{{tuple(k): v for k, v in {var}.items()}}"
    if name == "shortest_paths":
        return f"dict({var})"
    if name == "minimum_spanning_tree":
        return f"set(tuple(e) for e in {var})"
    return var


def _gold_source(name: str) -> str:
    """The gold solution = the correct reference, written as solution.py."""
    import inspect
    func = getattr(qc, name)
    src = inspect.getsource(func)
    # Drop leading "def " indentation artifacts from module layout.
    return src.strip() + "\n"


def _build_pure_test(name: str) -> str:
    ref = getattr(qc, name)
    cases = []
    for args in PURE_INPUTS[name]:
        raw = ref(*args)
        expected = _normalize(name, raw)
        cases.append((args, expected))
    norm_call = _normalize_expr(name, f"{name}(*args)")
    body = [
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        "",
        "",
        f"from solution import {name}",
        "",
        "",
        f"_CASES = {_dump_cases(cases)}",
        "",
        "",
        "@pytest.mark.parametrize('args,expected', _CASES)",
        "def test_case(args, expected):",
        f"    result = {norm_call}",
        f"    assert result == expected, f'{name} failed for {{args}}: got {{result}}, want {{expected}}'",
        "",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------- #
# Graph / linked-list algorithm tests
# --------------------------------------------------------------------------- #

def _build_graph_test(name: str) -> str:
    if name == "breadth_first_search":
        return _bfs_dfs_test(name)
    if name == "depth_first_search":
        return _bfs_dfs_test(name)
    if name == "detect_cycle":
        return _detect_cycle_test()
    if name == "reverse_linked_list":
        return _reverse_linked_list_test()
    if name == "shortest_path_length":
        return _shortest_path_length_test()
    if name == "topological_ordering":
        return _topological_test()
    raise ValueError(name)


def _bfs_dfs_test(name: str) -> str:
    # Build a fixed graph and compute expected bools from the reference.
    def make_graph():
        n0, n1, n2, n3 = (qc.Node(label=i) for i in range(4))
        n0.successors = [n1, n2]
        n1.successors = [n3]
        n2.successors = [n3]
        n3.successors = []
        return n0, n3

    ref = getattr(qc, name)
    start, goal = make_graph()
    reach = ref(start, goal)
    start2, _ = make_graph()
    outside = qc.Node(label=99)
    noreach = ref(start2, outside)
    cases = [("reachable", reach), ("unreachable", noreach)]
    return "\n".join([
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        NODE_HELPER,
        "",
        "",
        "def _graph():",
        "    n0, n1, n2, n3 = (Node(label=i) for i in range(4))",
        "    n0.successors = [n1, n2]",
        "    n1.successors = [n3]",
        "    n2.successors = [n3]",
        "    n3.successors = []",
        "    return n0, n3",
        "",
        "",
        f"from solution import {name}",
        "",
        "",
        f"_EXPECTED = {repr(cases)}",
        "",
        "",
        "@pytest.mark.parametrize('label,expected', _EXPECTED)",
        "def test_case(label, expected):",
        "    start, goal = _graph()",
        "    if label == 'reachable':",
        f"        assert {name}(start, goal) == expected",
        "    else:",
        "        outside = Node(label=99)",
        f"        assert {name}(start, outside) == expected",
        "",
    ])


def _detect_cycle_test() -> str:
    # No-cycle list: a->b->c->None.  Cycle list: x->y->z->y.
    def make_nocycle():
        a, b, c = (qc.Node(label=x) for x in "abc")
        a.successor = b
        b.successor = c
        c.successor = None
        return a

    def make_cycle():
        x, y, z = (qc.Node(label=x) for x in "xyz")
        x.successor = y
        y.successor = z
        z.successor = y
        return x

    nocyc = qc.detect_cycle(make_nocycle())
    cyc = qc.detect_cycle(make_cycle())
    return "\n".join([
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        NODE_HELPER,
        "",
        "",
        "def _no_cycle():",
        "    a, b, c = (Node(label=x) for x in 'abc')",
        "    a.successor = b; b.successor = c; c.successor = None",
        "    return a",
        "",
        "",
        "def _cycle():",
        "    x, y, z = (Node(label=x) for x in 'xyz')",
        "    x.successor = y; y.successor = z; z.successor = y",
        "    return x",
        "",
        "",
        "from solution import detect_cycle",
        "",
        "",
        "@pytest.mark.parametrize('builder,expected', "
        f"[(_no_cycle, {repr(nocyc)}), (_cycle, {repr(cyc)})])",
        "def test_case(builder, expected):",
        "    assert detect_cycle(builder()) == expected",
        "",
    ])


def _reverse_linked_list_test() -> str:
    # a->b->c->None ; reversed order labels: c, b, a
    def make():
        a, b, c = (qc.Node(label=x) for x in "abc")
        a.successor = b
        b.successor = c
        c.successor = None
        return a

    head = make()
    res = qc.reverse_linked_list(head)
    order = []
    cur = res
    while cur is not None:
        order.append(cur.label)
        cur = cur.successor
    return "\n".join([
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        NODE_HELPER,
        "",
        "",
        "def _list():",
        "    a, b, c = (Node(label=x) for x in 'abc')",
        "    a.successor = b; b.successor = c; c.successor = None",
        "    return a",
        "",
        "",
        "from solution import reverse_linked_list",
        "",
        "",
        f"_EXPECTED = {repr(order)}",
        "",
        "",
        "def test_case():",
        "    head = reverse_linked_list(_list())",
        "    labels = []",
        "    cur = head",
        "    while cur is not None:",
        "        labels.append(cur.label)",
        "        cur = cur.successor",
        "    assert labels == _EXPECTED, f'reversed order {{labels}} != {{_EXPECTED}}'",
        "",
    ])


def _shortest_path_length_test() -> str:
    def make():
        A, B, C, D = (qc.Node(label=x) for x in "ABCD")
        A.successors = [B, C]
        B.successors = [C]
        C.successors = [D]
        D.successors = []
        lbe = {(A, B): 1, (A, C): 4, (B, C): 1, (C, D): 1}
        return A, D, lbe

    A, D, lbe = make()
    dist = qc.shortest_path_length(lbe, A, D)
    return "\n".join([
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        NODE_HELPER,
        "",
        "",
        "def _graph():",
        "    A, B, C, D = (Node(label=x) for x in 'ABCD')",
        "    A.successors = [B, C]; B.successors = [C]; C.successors = [D]; D.successors = []",
        "    lbe = {(A, B): 1, (A, C): 4, (B, C): 1, (C, D): 1}",
        "    return A, D, lbe",
        "",
        "",
        "from solution import shortest_path_length",
        "",
        "",
        f"_EXPECTED = {repr(dist)}",
        "",
        "",
        "def test_case():",
        "    start, goal, lbe = _graph()",
        "    assert shortest_path_length(lbe, start, goal) == _EXPECTED",
        "",
    ])


def _topological_test() -> str:
    def make():
        X = qc.Node(label="X", incoming_nodes=set(), outgoing_nodes=[])
        Y = qc.Node(label="Y", incoming_nodes={X}, outgoing_nodes=[])
        Z = qc.Node(label="Z", incoming_nodes={X}, outgoing_nodes=[])
        W = qc.Node(label="W", incoming_nodes={Y, Z}, outgoing_nodes=[])
        X.outgoing_nodes = [Y, Z]
        Y.outgoing_nodes = [W]
        Z.outgoing_nodes = [W]
        return [X, Y, Z, W]

    nodes = make()
    order = [n.label for n in qc.topological_ordering(nodes)]
    return "\n".join([
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        "import pytest",
        NODE_HELPER,
        "",
        "",
        "def _dag():",
        "    X = Node(label='X', incoming_nodes=set(), outgoing_nodes=[])",
        "    Y = Node(label='Y', incoming_nodes={X}, outgoing_nodes=[])",
        "    Z = Node(label='Z', incoming_nodes={X}, outgoing_nodes=[])",
        "    W = Node(label='W', incoming_nodes={Y, Z}, outgoing_nodes=[])",
        "    X.outgoing_nodes = [Y, Z]; Y.outgoing_nodes = [W]; Z.outgoing_nodes = [W]",
        "    return [X, Y, Z, W]",
        "",
        "",
        "from solution import topological_ordering",
        "",
        "",
        f"_EXPECTED = {repr(order)}",
        "",
        "",
        "def test_case():",
        "    result = topological_ordering(_dag())",
        "    labels = [n.label for n in result]",
        "    assert labels == _EXPECTED, f'topo order {{labels}} != {{_EXPECTED}}'",
        "",
    ])


def build_test_solution(name: str) -> str:
    if name in PURE_INPUTS:
        return _build_pure_test(name)
    return _build_graph_test(name)


def build_gold_solution(name: str) -> str:
    return _gold_source(name)


if __name__ == "__main__":
    # Self-test: every algorithm produces a syntactically valid test + gold
    # whose gold passes its own test.
    import subprocess
    import sys
    tmp = Path("/tmp/qb_selftest")
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n_ok = 0
    for i, name in enumerate(ALGOS):
        gold = build_gold_solution(name)
        test = build_test_solution(name)
        (tmp / "solution.py").write_text(gold)
        (tmp / "test_solution.py").write_text(test)
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(tmp / "test_solution.py"), "-q"],
            capture_output=True, text=True,
        )
        ok = r.returncode == 0
        n_ok += ok
        status = "OK" if ok else "FAIL"
        print(f"[{i:02d}] {name:30} {status}")
        if not ok:
            print("    stdout:", r.stdout[-400:])
            print("    stderr:", r.stderr[-400:])
    print(f"\n{n_ok}/{len(ALGOS)} algorithms self-test green")
