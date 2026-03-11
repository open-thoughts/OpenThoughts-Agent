#!/usr/bin/env python3
"""
Tests for multi-round negotiation task generation and verifier logic.

Run with:
  python -m data.negotiation.test_generate
  pytest data/negotiation/test_generate.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Lazy import of generate to avoid pulling in data.commons (pyarrow) until needed
_generate_module = None

def _get_generate():
    global _generate_module
    if _generate_module is None:
        try:
            import data.negotiation.generate as m
            _generate_module = m
        except Exception as e:
            _generate_module = e
    if isinstance(_generate_module, Exception):
        raise _generate_module
    return _generate_module


# ---------------------------------------------------------------------------
# Verifier reward formula (inline for test independence)
# ---------------------------------------------------------------------------

def compute_reward(price: float, r_s: float, r_b: float, role: str, eps: float = 1e-9) -> float:
    """Verifier reward formula: A * (u_agent / (TS + eps))."""
    A = 1.0 if (r_s <= price <= r_b) else 0.0
    if A == 0:
        return 0.0
    u_s = price - r_s
    u_b = r_b - price
    TS = r_b - r_s
    u_agent = u_s if role == "seller" else u_b
    return A * (u_agent / (TS + eps))


class TestVerifierReward(unittest.TestCase):
    """Test the verifier reward formula."""

    def test_no_agreement_below_zopa(self):
        self.assertEqual(compute_reward(80, 90, 110, "seller"), 0.0)
        self.assertEqual(compute_reward(80, 90, 110, "buyer"), 0.0)

    def test_no_agreement_above_zopa(self):
        self.assertEqual(compute_reward(120, 90, 110, "seller"), 0.0)
        self.assertEqual(compute_reward(120, 90, 110, "buyer"), 0.0)

    def test_agreement_seller_captures_all(self):
        reward = compute_reward(110, 90, 110, "seller")
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_agreement_buyer_captures_all(self):
        reward = compute_reward(90, 90, 110, "buyer")
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_agreement_half_surplus(self):
        self.assertAlmostEqual(compute_reward(100, 90, 110, "seller"), 0.5, places=6)
        self.assertAlmostEqual(compute_reward(100, 90, 110, "buyer"), 0.5, places=6)


class TestLoadItems(unittest.TestCase):
    """Test CSV and default item loading."""

    def test_default_items(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        DEFAULT_ITEMS = gen.DEFAULT_ITEMS
        self.assertGreater(len(DEFAULT_ITEMS), 0)
        for item in DEFAULT_ITEMS:
            self.assertIn("r_s", item)
            self.assertIn("r_b", item)
            self.assertLessEqual(item["r_s"], item["r_b"])

    def test_load_from_craigslist_csv(self):
        csv_path = ROOT / "data" / "negotiation" / "craigslist_bargains" / "train.csv"
        if not csv_path.exists():
            self.skipTest(f"Craigslist CSV not found: {csv_path}")
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        items = gen.load_items_from_csv(csv_path)
        if len(items) == 0:
            self.skipTest("CSV has no rows with ZOPA; column semantics may differ")
        for item in items[:5]:
            self.assertIn("r_s", item)
            self.assertIn("r_b", item)
            self.assertLessEqual(item["r_s"], item["r_b"])

    def test_load_from_craigslist_csv_resample_mode(self):
        csv_path = ROOT / "data" / "negotiation" / "craigslist_bargains" / "train.csv"
        if not csv_path.exists():
            self.skipTest(f"Craigslist CSV not found: {csv_path}")
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        import random
        rng = random.Random(42)
        items = gen.load_items_from_csv(csv_path, zopa_mode="resample", rng=rng)
        self.assertGreater(len(items), 0, "resample mode should yield items even without ZOPA rows")
        for item in items[:10]:
            self.assertLessEqual(item["r_s"], item["r_b"], "resample must ensure r_s <= r_b")


class TestDeriveTaskParams(unittest.TestCase):
    """Test task parameter derivation from item + role."""

    def test_seller_derived_params(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "X", "description": "", "list_price": 100.0, "category": "cat",
                "r_s": 80.0, "r_b": 120.0}
        params = gen._derive_task_params(item, "seller")
        self.assertEqual(params["K"], 10)
        self.assertAlmostEqual(params["p_min"], 50.0)
        self.assertAlmostEqual(params["p_max"], 200.0)
        # Counterpart is buyer with r_b=120; opening = 120 * 0.80 = 96
        self.assertAlmostEqual(params["counterpart_opening"], 96.0)

    def test_buyer_derived_params(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "X", "description": "", "list_price": 100.0, "category": "cat",
                "r_s": 80.0, "r_b": 120.0}
        params = gen._derive_task_params(item, "buyer")
        # Counterpart is seller with r_s=80; opening = min(200, 80*1.20) = 96
        self.assertAlmostEqual(params["counterpart_opening"], 96.0)

    def test_p_min_not_negative(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "X", "description": "", "list_price": 1.0, "category": "cat",
                "r_s": 0.5, "r_b": 1.5}
        params = gen._derive_task_params(item, "seller")
        self.assertGreaterEqual(params["p_min"], 0.0)


class TestBuildScenarioAndInstruction(unittest.TestCase):
    """Test scenario and instruction builders."""

    def test_build_scenario_has_multi_round_fields(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "X", "description": "Y", "list_price": 50.0, "category": "cat",
                "r_s": 40.0, "r_b": 60.0}
        scenario = gen.build_scenario(item, "seller", seed=1)
        self.assertEqual(scenario["r_s"], 40.0)
        self.assertEqual(scenario["r_b"], 60.0)
        self.assertEqual(scenario["role"], "seller")
        self.assertEqual(scenario["seed"], 1)
        # Multi-round fields must be present
        self.assertIn("K", scenario)
        self.assertIn("p_min", scenario)
        self.assertIn("p_max", scenario)
        self.assertIn("delta_max", scenario)
        self.assertIn("counterpart_opening", scenario)
        self.assertEqual(scenario["K"], 10)

    def test_build_instruction_multi_round(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "Test Item", "description": "Desc", "list_price": 50.0,
                "category": "electronics", "r_s": 40.0, "r_b": 60.0}
        scenario = gen.build_scenario(item, "buyer", seed=99)
        text = gen.build_instruction("buyer", item, scenario)
        self.assertIn("buyer", text)
        self.assertIn("Test Item", text)
        self.assertIn("counterpart.py offer", text)
        self.assertIn("counterpart.py accept", text)
        self.assertIn("counterpart.py reject", text)
        # Agent sees their own reservation (r_b for buyer)
        self.assertIn("60.00", text)
        # Instruction should NOT contain the counterpart's reservation directly labelled
        self.assertNotIn("r_s", text)

    def test_build_instruction_seller(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        item = {"title": "Chair", "description": "", "list_price": 100.0,
                "category": "furniture", "r_s": 80.0, "r_b": 120.0}
        scenario = gen.build_scenario(item, "seller", seed=5)
        text = gen.build_instruction("seller", item, scenario)
        self.assertIn("seller", text)
        # Seller sees their own reservation (r_s)
        self.assertIn("80.00", text)
        self.assertIn("higher", text)


class TestExampleTask(unittest.TestCase):
    """Test example task (specimen) layout and verifier."""

    def test_example_task_files_exist(self):
        example_dir = ROOT / "data" / "negotiation" / "example_task"
        self.assertTrue((example_dir / "instruction.md").exists(), "instruction.md")
        self.assertTrue((example_dir / "data" / "scenario.json").exists(), "data/scenario.json")
        self.assertTrue((example_dir / "tests" / "test.sh").exists(), "tests/test.sh")
        self.assertTrue((example_dir / "task.toml").exists(), "task.toml")
        self.assertTrue((example_dir / "environment" / "Dockerfile").exists(), "environment/Dockerfile")
        self.assertTrue((example_dir / "environment" / "counterpart.py").exists(), "environment/counterpart.py")

    def test_example_scenario_valid(self):
        path = ROOT / "data" / "negotiation" / "example_task" / "data" / "scenario.json"
        with open(path) as f:
            scenario = json.load(f)
        self.assertEqual(scenario["role"], "seller")
        self.assertLessEqual(scenario["r_s"], scenario["r_b"])
        # Multi-round fields
        self.assertIn("K", scenario)
        self.assertIn("counterpart_opening", scenario)
        self.assertGreater(scenario["K"], 0)

    def test_example_task_toml_timeout(self):
        path = ROOT / "data" / "negotiation" / "example_task" / "task.toml"
        content = path.read_text()
        # Agent timeout should be >= 1800 for multi-round
        import re
        m = re.search(r"timeout_sec\s*=\s*(\d+\.?\d*)", content)
        self.assertIsNotNone(m, "timeout_sec not found in task.toml")
        self.assertGreaterEqual(float(m.group(1)), 1800.0)

    def test_example_dockerfile_has_openai(self):
        path = ROOT / "data" / "negotiation" / "example_task" / "environment" / "Dockerfile"
        content = path.read_text()
        self.assertIn("openai", content)
        self.assertIn("counterpart.py", content)

    def test_example_verifier_reward_simulation(self):
        """Simulate verifier: deal at 100, r_s=85 r_b=110 seller → reward = (100-85)/(110-85) = 0.6."""
        path = ROOT / "data" / "negotiation" / "example_task" / "data" / "scenario.json"
        with open(path) as f:
            scenario = json.load(f)
        reward = compute_reward(100, scenario["r_s"], scenario["r_b"], scenario["role"])
        self.assertAlmostEqual(reward, 0.6, places=6)

    def test_example_instruction_has_commands(self):
        path = ROOT / "data" / "negotiation" / "example_task" / "instruction.md"
        content = path.read_text()
        self.assertIn("counterpart.py offer", content)
        self.assertIn("counterpart.py accept", content)
        self.assertIn("counterpart.py reject", content)


class TestCounterpartModule(unittest.TestCase):
    """Test counterpart.py logic (without calling the LLM API)."""

    def _get_counterpart(self):
        cp_path = ROOT / "data" / "negotiation" / "counterpart.py"
        if not cp_path.exists():
            self.skipTest("counterpart.py not found")
        import importlib.util
        spec = importlib.util.spec_from_file_location("counterpart", cp_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_compute_reward_agreement(self):
        mod = self._get_counterpart()
        reward = mod.compute_reward(100.0, 90.0, 110.0, "seller")
        self.assertAlmostEqual(reward, 0.5, places=6)

    def test_compute_reward_no_deal(self):
        mod = self._get_counterpart()
        self.assertEqual(mod.compute_reward(80.0, 90.0, 110.0, "seller"), 0.0)

    def test_rule_based_response_accept(self):
        """Rule-based: buyer accepts when offer <= reservation."""
        mod = self._get_counterpart()
        resp = mod._rule_based_response(
            counterpart_role="buyer",
            counterpart_reservation=110.0,
            agent_offer=105.0,  # below buyer reservation
            history=[],
            rounds_remaining=5,
        )
        self.assertEqual(resp["action"], "accept")

    def test_rule_based_response_counter(self):
        """Rule-based: buyer counters when offer > reservation."""
        mod = self._get_counterpart()
        resp = mod._rule_based_response(
            counterpart_role="buyer",
            counterpart_reservation=110.0,
            agent_offer=120.0,  # above buyer reservation -> should counter
            history=[{"turn": "counterpart", "action": "offer", "price": 95.0, "message": ""}],
            rounds_remaining=5,
        )
        self.assertEqual(resp["action"], "offer")
        self.assertIsNotNone(resp["price"])
        self.assertLess(resp["price"], 120.0)

    def test_counterpart_opening_seller(self):
        """Seller counterpart opens above their reservation."""
        mod = self._get_counterpart()
        opening = mod._counterpart_opening("seller", 80.0, 0.0, 200.0)
        self.assertGreater(opening, 80.0)
        self.assertLessEqual(opening, 200.0)

    def test_counterpart_opening_buyer(self):
        """Buyer counterpart opens below their reservation."""
        mod = self._get_counterpart()
        opening = mod._counterpart_opening("buyer", 110.0, 0.0, 200.0)
        self.assertLess(opening, 110.0)
        self.assertGreaterEqual(opening, 0.0)

    def test_init_state_seller_scenario(self):
        """State initialized correctly for seller agent (buyer counterpart)."""
        mod = self._get_counterpart()
        scenario = {
            "r_s": 90.0, "r_b": 110.0, "role": "seller", "seed": 42,
            "K": 10, "p_min": 12.5, "p_max": 50.0, "delta_max": 3.75,
            "counterpart_opening": 88.0,
            "item_context": {"title": "Mouse", "description": "", "list_price": 25.0, "category": "electronics"},
        }
        state = mod.init_state(scenario)
        self.assertEqual(state["counterpart_role"], "buyer")
        self.assertEqual(state["counterpart_reservation"], 110.0)
        self.assertEqual(state["counterpart_last_offer"], 88.0)
        self.assertFalse(state["done"])
        self.assertEqual(state["K"], 10)


class TestFullGeneration(unittest.TestCase):
    """Test full task generation (requires data.commons)."""

    def test_generate_task_has_multi_round_structure(self):
        try:
            gen = _get_generate()
        except Exception as e:
            self.skipTest(f"generate module unavailable: {e}")
        items = gen.load_items(None, gen.DEFAULT_ITEMS)
        self.assertGreater(len(items), 0)
        item = items[0]
        role = "seller"
        scenario = gen.build_scenario(item, role, seed=99)
        instruction = gen.build_instruction(role, item, scenario)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                task_dir = gen.create_negotiation_task(out, 0, scenario, instruction)
                # Standard Harbor files
                self.assertTrue((task_dir / "instruction.md").exists())
                self.assertTrue((task_dir / "data" / "scenario.json").exists())
                self.assertTrue((task_dir / "tests" / "test.sh").exists())
                self.assertTrue((task_dir / "task.toml").exists())
                self.assertTrue((task_dir / "environment" / "Dockerfile").exists())
                # Multi-round additions
                self.assertTrue((task_dir / "environment" / "counterpart.py").exists(),
                                "counterpart.py must be in environment/")
                # Scenario has multi-round fields
                with open(task_dir / "data" / "scenario.json") as f:
                    loaded = json.load(f)
                self.assertIn("K", loaded)
                self.assertIn("counterpart_opening", loaded)
                self.assertEqual(loaded["r_s"], scenario["r_s"])
                self.assertEqual(loaded["role"], role)
                # Dockerfile references openai and counterpart.py
                dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
                self.assertIn("openai", dockerfile)
                self.assertIn("counterpart.py", dockerfile)
                # Verifier reads negotiation_log.json not contract.json
                verifier = (task_dir / "tests" / "test.sh").read_text()
                self.assertIn("negotiation_log.json", verifier)
                self.assertNotIn("contract.json", verifier)
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"data.commons (pyarrow/numpy) unavailable: {e}")


def run_standalone():
    """Run tests without pytest."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestVerifierReward,
        TestLoadItems,
        TestDeriveTaskParams,
        TestBuildScenarioAndInstruction,
        TestExampleTask,
        TestCounterpartModule,
        TestFullGeneration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_standalone())
