import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
from src.mda_config import resolve_rank, validate_parameter_options


ROOT = Path(__file__).resolve().parents[1]


class MDAConfigTests(unittest.TestCase):
    def compose(self, name="default", overrides=None):
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            return compose(config_name=name, overrides=overrides or [])

    def test_all_entry_points_have_mda_defaults(self):
        for name in ("default", "clip_generalization_exp1", "clip_generalization_exp2", "flan_t5_default"):
            with self.subTest(config=name):
                cfg = self.compose(name)
                self.assertEqual(validate_parameter_options(cfg.mda), (2400, 1.0))
                self.assertEqual(resolve_rank((768, 768), cfg.mda, 8), 96)

    def test_cli_overrides_reach_rank_and_etf_dimension(self):
        cfg = self.compose(overrides=["mda.k=0.0625", "mda.dimension=1200", "mda.parameter_alpha=0.5"])
        self.assertEqual(resolve_rank((768, 512), cfg.mda, 8), 32)
        self.assertEqual(validate_parameter_options(cfg.mda), (1200, 0.5))

    def test_rank_fraction_and_minimum(self):
        options = self.compose(overrides=["mda.k=0.01"]).mda
        self.assertEqual(resolve_rank((768, 512), options, 8), 5)
        self.assertEqual(resolve_rank((2, 3), options, 8), 1)

    def test_default_ratio_tracks_actual_merged_tasks(self):
        options = self.compose().mda
        for count in (1, 6, 8, 14, 20):
            with self.subTest(tasks=count):
                self.assertEqual(resolve_rank((768, 512), options, count), 512 // count)

    def test_bad_fraction_and_task_count_rejected(self):
        for fraction in (0, -0.1, 1.1, 32, float("nan"), float("inf"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                resolve_rank((768, 512), SimpleNamespace(k=fraction), 8)
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                resolve_rank((768, 512), SimpleNamespace(k=None), count)

    def test_invalid_etf_options_rejected(self):
        for dim in (0, 1, -1, 2.5, True):
            with self.subTest(dimension=dim), self.assertRaises(ValueError):
                validate_parameter_options(SimpleNamespace(dimension=dim, parameter_alpha=1.0))
        for alpha in (-0.1, 1.1, float("nan"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                validate_parameter_options(SimpleNamespace(dimension=2400, parameter_alpha=alpha))

    def test_experiment_specific_defaults_preserved(self):
        clip = self.compose().mda
        generalization = self.compose("clip_generalization_exp1").mda
        flan = self.compose("flan_t5_default").mda
        self.assertEqual((clip.lr, clip.eval_interval, clip.alignment_warmup_epochs, clip.rotation_weight), (0.001, 500, 0, 0.0))
        self.assertEqual((generalization.eval_interval, generalization.alignment_warmup_epochs, generalization.rotation_weight), (200, 300, 0.2))
        self.assertEqual(flan.lr, 0.0001)

    def test_soft_assignment_temperature_config(self):
        self.assertEqual(self.compose().mda.assignment_temperature, 1.0)
        self.assertEqual(self.compose(overrides=["mda.assignment_temperature=0.2"]).mda.assignment_temperature, 0.2)
        self.assertEqual(self.compose("clip_generalization_exp1").mda.assignment_temperature, 1.0)

    def test_flan_merge_forwards_config_to_mda(self):
        # Exercise the actual adapter without importing GPU/model dependencies.
        tree = ast.parse((ROOT / "src/task_wise_fusion_TSVM.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fuse_weights_TSVM")
        function.args.args[0].annotation = None
        function.args.args[1].annotation = None
        function.returns = None
        received = []
        namespace = {"types": __import__("types"), "TSVM_utils": SimpleNamespace(compute_mda=lambda vectors, config: received.append((vectors, config)) or "merged")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "adapter", "exec"), namespace)
        cfg = self.compose("flan_t5_default", ["mda.k=0.125", "mda.dimension=800"])
        self.assertEqual(namespace["fuse_weights_TSVM"](["task"], "cpu", cfg), "merged")
        vectors, actual = received[0]
        self.assertEqual(vectors, ["task"])
        self.assertEqual((actual.device, actual.mda.k, actual.mda.dimension), ("cpu", 0.125, 800))


if __name__ == "__main__":
    unittest.main()
