"""Regression coverage for the QR branch's [classes, feature_dim] contract."""

import ast
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from src.mda_config import resolve_rank, validate_parameter_options


ROOT = Path(__file__).resolve().parents[1]


class ParameterQRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        source = (ROOT / "src/TSVM_utils.py").read_text(encoding="utf-8")
        # Execute the actual math definitions without CLIP/dataset dependencies.
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {"compute_mda", "compute_and_sum_svd_mem_reduction", "compute_and_sum_svd_mem_reduction_subspace_avg", "build_etf_svd", "build_etf_qr"}]
        self.namespace = dict(torch=torch, F=F, np=SimpleNamespace(sqrt=math.sqrt),
                              resolve_rank=resolve_rank, validate_parameter_options=validate_parameter_options)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "parameter-merge", "exec"), self.namespace)

    def merge(self, vectors, dimension, alpha=1.0):
        cfg = SimpleNamespace(device="cpu", mda=SimpleNamespace(k=None, dimension=dimension, parameter_alpha=alpha))
        with redirect_stdout(io.StringIO()):
            return self.namespace["compute_mda"](vectors, cfg)

    def test_qr_builder_returns_classes_by_features(self):
        etf = self.namespace["build_etf_qr"](d=8, c=4)
        self.assertEqual(etf.shape, (4, 8))
        torch.testing.assert_close(etf @ etf.T, torch.eye(4), atol=1e-6, rtol=1e-6)

    def test_rectangular_qr_branch_preserves_layer_shapes(self):
        vectors = [{"weight": torch.randn(12, 8), "bias": torch.randn(12),
                    "text_projection": torch.randn(8, 8)} for _ in range(2)]
        qr = Mock(wraps=self.namespace["build_etf_qr"])
        svd = Mock(wraps=self.namespace["build_etf_svd"])
        self.namespace.update(build_etf_qr=qr, build_etf_svd=svd)
        result = self.merge(vectors, dimension=4)
        qr.assert_called_once_with(d=8, c=4)
        svd.assert_not_called()
        for key, value in result.items():
            self.assertEqual(value.shape, vectors[0][key].shape)
            self.assertTrue(torch.isfinite(value).all())
        # These non-QR paths must keep the original averaging behavior.
        for key in ("bias", "text_projection"):
            torch.testing.assert_close(result[key], torch.stack([v[key] for v in vectors]).mean(0))

    def test_square_boundary_uses_qr(self):
        vectors = [{"weight": torch.randn(6, 4)} for _ in range(2)]
        qr = Mock(wraps=self.namespace["build_etf_qr"])
        self.namespace["build_etf_qr"] = qr
        result = self.merge(vectors, dimension=4)
        qr.assert_called_once_with(d=4, c=4)
        self.assertEqual(result["weight"].shape, (6, 4))
        self.assertTrue(torch.isfinite(result["weight"]).all())

    def test_svd_branch_matches_pre_fix_baseline(self):
        a = torch.tensor([[1., 2., 0., -1.], [0., 3., 1., 2.], [2., -1., 4., 0.], [1., 0., 2., 5.]])
        b = torch.tensor([[3., 0., 1., 2.], [-1., 1., 2., 0.], [0., 2., 1., 4.], [2., 3., -1., 1.]])
        # Captured from the unmodified SVD branch before the QR fix.
        expected = torch.tensor([
            [1.92407906, 5.22838640, -0.76143861, -0.77278125],
            [-2.15324569, 1.38487387, 0.07458901, 3.20803380],
            [1.57334328, 1.15379143, 4.26165485, 0.68824798],
            [2.67441201, -1.39149594, 0.50796515, 4.83715916],
        ])
        qr = Mock(wraps=self.namespace["build_etf_qr"])
        self.namespace["build_etf_qr"] = qr
        result = self.merge([{"weight": a}, {"weight": b}], dimension=6)
        qr.assert_not_called()
        torch.testing.assert_close(result["weight"], expected, atol=2e-5, rtol=2e-5)

    def test_qr_projection_and_alpha_blending_are_unchanged(self):
        vectors = [{"weight": torch.randn(6, 8)} for _ in range(2)]
        # A fixed orthonormal reference makes the expected projection explicit.
        etf = torch.cat([torch.eye(4), torch.zeros(4, 4)], dim=1)
        self.namespace["build_etf_qr"] = Mock(return_value=etf)
        reconstructed = self.merge(vectors, dimension=4, alpha=0.0)["weight"]
        aligned = self.merge(vectors, dimension=4, alpha=1.0)["weight"]
        blended = self.merge(vectors, dimension=4, alpha=0.3)["weight"]
        torch.testing.assert_close(aligned[:, :4], reconstructed[:, :4])
        torch.testing.assert_close(aligned[:, 4:], torch.zeros_like(aligned[:, 4:]))
        torch.testing.assert_close(blended, 0.7 * reconstructed + 0.3 * aligned)

    def test_default_entry_point_uses_concat(self):
        cfg = SimpleNamespace(device="cpu", mda=SimpleNamespace(k=None, dimension=4, parameter_alpha=1.0))
        vectors = [{"weight": torch.randn(6, 8)} for _ in range(2)]
        with redirect_stdout(io.StringIO()):
            torch.manual_seed(10)
            expected = self.namespace["compute_and_sum_svd_mem_reduction"](vectors, cfg)
            torch.manual_seed(10)
            actual = self.namespace["compute_mda"](vectors, cfg)
        torch.testing.assert_close(actual["weight"], expected["weight"])

    def test_subspace_average_matches_low_rank_mean_projection(self):
        cfg = SimpleNamespace(device="cpu", mda=SimpleNamespace(k=None, dimension=4, parameter_alpha=1.0))
        vectors = [{"weight": torch.randn(6, 8), "bias": torch.randn(6)} for _ in range(3)]
        etf = torch.cat([torch.eye(4), torch.zeros(4, 4)], dim=1)
        self.namespace["build_etf_qr"] = Mock(return_value=etf)
        low_rank = []
        for vector in vectors:
            u, s, vh = torch.linalg.svd(vector["weight"], full_matrices=False)
            low_rank.append((u[:, :2] * s[:2]) @ vh[:2])
        expected = torch.stack(low_rank).mean(0)
        expected[:, 4:] = 0
        with redirect_stdout(io.StringIO()):
            actual = self.namespace["compute_and_sum_svd_mem_reduction_subspace_avg"](vectors, cfg)
        torch.testing.assert_close(actual["weight"], expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual["bias"], torch.stack([v["bias"] for v in vectors]).mean(0))


if __name__ == "__main__":
    unittest.main()
