#!/usr/bin/env python3
"""Correct reference implementations + canonical inputs for the 40 QuixBugs
Python algorithms, used to (a) seed a known-correct gold solution and
(b) emit a real ``tests/test_solution.py`` with hardcoded expected outputs.

These are the standard, textbook-correct versions of each algorithm.  Expected
outputs are computed at build time by executing these references, then baked as
literals into the emitted pytest module so the verifier needs no reference at
runtime -- only the ``Node`` fixture helper for graph/linked-list tasks.
"""
from __future__ import annotations


class Node:
    """Minimal graph / linked-list node used by the graph algorithms.

    Equality is identity (no ``__eq__`` override) which matches the way the
    fixtures are constructed -- the goal/start node is always a real reference
    into the graph, so ``is`` and ``==`` agree.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# --------------------------------------------------------------------------- #
# Correct reference implementations
# --------------------------------------------------------------------------- #

def bitcount(n):
    c = 0
    while n:
        n &= n - 1
        c += 1
    return c


def bucketsort(arr, k):
    counts = [0] * k
    for x in arr:
        counts[x] += 1
    out = []
    for i, c in enumerate(counts):
        out.extend([i] * c)
    return out


def find_first_in_sorted(arr, x):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid
        elif x <= arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return -1


def find_in_sorted(arr, x):
    def binsearch(start, end):
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        return mid
    return binsearch(0, len(arr))


def flatten(arr):
    for x in arr:
        if isinstance(x, list):
            yield from flatten(x)
        else:
            yield x


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def get_factors(n):
    if n == 1:
        return []
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return [i] + get_factors(n // i)
    return [n]


def hanoi(height, start=1, end=3):
    steps = []
    if height > 0:
        helper = ({1, 2, 3} - {start} - {end}).pop()
        steps.extend(hanoi(height - 1, start, helper))
        steps.append((start, end))
        steps.extend(hanoi(height - 1, helper, end))
    return steps


def is_valid_parenthesization(parens):
    depth = 0
    for paren in parens:
        if paren == "(":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def kheapsort(arr, k):
    import heapq
    heap = arr[:k]
    heapq.heapify(heap)
    for x in arr[k:]:
        yield heapq.heappushpop(heap, x)
    while heap:
        yield heapq.heappop(heap)


def knapsack(capacity, items):
    from collections import defaultdict
    memo = defaultdict(int)
    for i in range(1, len(items) + 1):
        weight, value = items[i - 1]
        for j in range(1, capacity + 1):
            memo[i, j] = memo[i - 1, j]
            if weight <= j:
                memo[i, j] = max(memo[i, j], value + memo[i - 1, j - weight])
    return memo[len(items), capacity]


def kth(arr, k):
    pivot = arr[0]
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]
    num_less = len(below)
    num_lessoreq = len(arr) - len(above)
    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k - num_lessoreq)
    return pivot


def lcs_length(s, t):
    from collections import Counter
    dp = Counter()
    for i in range(len(s)):
        for j in range(len(t)):
            if s[i] == t[j]:
                dp[i, j] = dp[i - 1, j - 1] + 1
    return max(dp.values()) if dp else 0


def levenshtein(source, target):
    if source == "" or target == "":
        return len(source) or len(target)
    elif source[0] == target[0]:
        return levenshtein(source[1:], target[1:])
    return 1 + min(
        levenshtein(source, target[1:]),
        levenshtein(source[1:], target[1:]),
        levenshtein(source[1:], target),
    )


def lis(arr):
    ends = {}
    longest = 0
    for i, val in enumerate(arr):
        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends[j]] < val]
        length = max(prefix_lengths) if prefix_lengths else 0
        if length == longest or val < arr[ends[length + 1]]:
            ends[length + 1] = i
            longest = max(longest, length + 1)
    return longest


def longest_common_subsequence(a, b):
    if not a or not b:
        return ""
    elif a[0] == b[0]:
        return a[0] + longest_common_subsequence(a[1:], b[1:])
    return max(
        longest_common_subsequence(a, b[1:]),
        longest_common_subsequence(a[1:], b),
        key=len,
    )


def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0
    for x in arr:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far


def mergesort(arr):
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i]); i += 1  # noqa: E702
            else:
                result.append(right[j]); j += 1  # noqa: E702
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    if len(arr) <= 1:
        return arr
    middle = len(arr) // 2
    return merge(mergesort(arr[:middle]), mergesort(arr[middle:]))


def next_palindrome(digit_list):
    digits = list(digit_list)
    high_mid = len(digits) // 2
    low_mid = (len(digits) - 1) // 2
    while high_mid < len(digits) and low_mid >= 0:
        if digits[high_mid] == 9:
            digits[high_mid] = 0
            digits[low_mid] = 0
            high_mid += 1
            low_mid -= 1
        else:
            digits[high_mid] += 1
            if low_mid != high_mid:
                digits[low_mid] += 1
            return digits
    return [1] + (len(digits) - 1) * [0] + [1]


def next_permutation(perm):
    for i in range(len(perm) - 2, -1, -1):
        if perm[i] < perm[i + 1]:
            for j in range(len(perm) - 1, i, -1):
                if perm[i] < perm[j]:
                    nxt = list(perm)
                    nxt[i], nxt[j] = perm[j], perm[i]
                    nxt[i + 1:] = reversed(nxt[i + 1:])
                    return nxt
    return list(perm)


def pascal(n):
    rows = [[1]]
    for r in range(1, n):
        row = []
        for c in range(r + 1):
            upleft = rows[r - 1][c - 1] if c > 0 else 0
            upright = rows[r - 1][c] if c < r else 0
            row.append(upleft + upright)
        rows.append(row)
    return rows


def possible_change(coins, total):
    if total == 0:
        return 1
    if total < 0 or not coins:
        return 0
    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)


def powerset(arr):
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return rest_subsets + [[first] + s for s in rest_subsets]
    return [[]]


def quicksort(arr):
    if not arr:
        return []
    pivot = arr[0]
    lesser = quicksort([x for x in arr[1:] if x < pivot])
    greater = quicksort([x for x in arr[1:] if x >= pivot])
    return lesser + [pivot] + greater


def rpn_eval(tokens):
    def op(symbol, a, b):
        return {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
                "*": lambda a, b: a * b, "/": lambda a, b: a / b}[symbol](a, b)
    stack = []
    for token in tokens:
        if isinstance(token, (int, float)):
            stack.append(token)
        else:
            a = stack.pop()
            b = stack.pop()
            stack.append(op(token, b, a))
    return stack.pop()


def shunting_yard(tokens):
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    rpntokens = []
    opstack = []
    for token in tokens:
        if isinstance(token, int):
            rpntokens.append(token)
        else:
            while opstack and precedence[token] <= precedence[opstack[-1]]:
                rpntokens.append(opstack.pop())
            opstack.append(token)
    while opstack:
        rpntokens.append(opstack.pop())
    return rpntokens


def sieve(max):
    primes = []
    for n in range(2, max + 1):
        if all(n % p > 0 for p in primes):
            primes.append(n)
    return primes


def sqrt(x, epsilon):
    approx = x / 2
    while abs(x - approx ** 2) > epsilon:
        approx = 0.5 * (approx + x / approx)
    return approx


def subsequences(a, b, k):
    if k == 0:
        return [[]]
    ret = []
    for i in range(a, b + 1 - k):
        ret.extend([i] + rest for rest in subsequences(i + 1, b, k - 1))
    return ret


def to_base(num, b):
    result = ""
    alphabet = __import__("string").digits + __import__("string").ascii_uppercase
    while num > 0:
        i = num % b
        num //= b
        result = alphabet[i] + result
    return result


def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = text.rfind(" ", 0, cols + 1)
        if end == -1:
            end = cols
        line, text = text[:end], text[end:]
        lines.append(line)
    lines.append(text)
    return lines


def shortest_path_lengths(n, length_by_edge):
    from collections import defaultdict
    dist = defaultdict(lambda: float("inf"))
    dist.update({(i, i): 0 for i in range(n)})
    dist.update(length_by_edge)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    return dict(dist)


def shortest_paths(source, weight_by_edge):
    weight_by_node = {v: float("inf") for u, v in weight_by_edge}
    weight_by_node[source] = 0
    for _ in range(len(weight_by_node) - 1):
        for (u, v), weight in weight_by_edge.items():
            weight_by_node[v] = min(weight_by_node[u] + weight, weight_by_node[v])
    return weight_by_node


# --- graph / linked-list references (take Node fixtures) -------------------- #

def breadth_first_search(startnode, goalnode):
    from collections import deque
    queue = deque([startnode])
    seen = {startnode}
    while queue:
        node = queue.popleft()
        if node is goalnode:
            return True
        for nxt in getattr(node, "successors", []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def depth_first_search(startnode, goalnode):
    seen = set()

    def visit(node):
        if node in seen:
            return False
        if node is goalnode:
            return True
        seen.add(node)
        return any(visit(n) for n in getattr(node, "successors", []))

    return visit(startnode)


def detect_cycle(node):
    hare = tortoise = node
    while True:
        if hare is None or getattr(hare, "successor", None) is None:
            return False
        tortoise = tortoise.successor
        hare = hare.successor.successor
        if hare is tortoise:
            return True


def reverse_linked_list(node):
    prevnode = None
    while node:
        nxt = node.successor
        node.successor = prevnode
        prevnode = node
        node = nxt
    return prevnode


def shortest_path_length(length_by_edge, startnode, goalnode):
    from heapq import heappush, heappop
    unvisited = [(0, startnode)]
    visited = set()
    while unvisited:
        distance, node = heappop(unvisited)
        if node is goalnode:
            return distance
        if node in visited:
            continue
        visited.add(node)
        for nxt in getattr(node, "successors", []):
            if nxt not in visited:
                heappush(unvisited, (distance + length_by_edge[node, nxt], nxt))
    return float("inf")


def topological_ordering(nodes):
    ordered = [n for n in nodes if not n.incoming_nodes]
    for node in ordered:
        for nxt in node.outgoing_nodes:
            if set(ordered).issuperset(nxt.incoming_nodes) and nxt not in ordered:
                ordered.append(nxt)
    return ordered


def minimum_spanning_tree(weight_by_edge):
    group_by_node = {}
    mst = set()
    for edge in sorted(weight_by_edge, key=weight_by_edge.__getitem__):
        u, v = edge
        if group_by_node.setdefault(u, {u}) != group_by_node.setdefault(v, {v}):
            mst.add(edge)
            group_by_node[u].update(group_by_node[v])
            for node in group_by_node[v]:
                group_by_node[node] = group_by_node[u]
    return mst
