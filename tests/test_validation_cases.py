"""Preserve the 5+5 validation contract and separation from final test tracks."""
import json
from pathlib import Path
import unittest


class ValidationSuiteTests(unittest.TestCase):
    def test_five_random_and_five_randomized_tracks_with_reserved_seeds(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        validation = json.loads((examples/"validation_cases.json").read_text())["cases"]
        testing = json.loads((examples/"evaluation_cases.json").read_text())["cases"]
        self.assertEqual(len(validation), 10)
        self.assertEqual(len({case["id"] for case in validation}), 10)
        self.assertEqual(len({case["seed"] for case in validation}), 10)
        self.assertFalse({case["seed"] for case in validation} & {case["seed"] for case in testing})
        for group, randomized in [("normal", False), ("randomized", True)]:
            cases = [case for case in validation if case["group"] == group]
            self.assertEqual(len(cases), 5)
            for case in cases:
                self.assertEqual(case["max_steps"], 1000)
                self.assertEqual(case["env_kwargs"]["domain_randomize"], randomized)
                self.assertEqual(case["env_kwargs"]["n_static_obstacles"], 0)
                self.assertEqual(case["env_kwargs"]["n_dynamic_obstacles"], 0)
