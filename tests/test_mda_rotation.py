import ast
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import itertools
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch._dynamo  # Load lazy optimizer dependencies before patching IO modules.
from torch import nn
import torch.nn.functional as F

from src.mda_rotation import (
    TaskRotations, proper_polar, optimal_rotation, generate_etf, alignment_loss,
    procrustes_penalty, save_alignment_checkpoint, load_alignment_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def extract_definitions(path, names, namespace):
    """Run production definitions while avoiding dataset/CLIP download imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


class RotationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def assert_proper(self, matrix):
        torch.testing.assert_close(matrix.T @ matrix, torch.eye(matrix.shape[0], dtype=matrix.dtype), atol=2e-5, rtol=2e-5)
        self.assertAlmostEqual(torch.linalg.det(matrix).item(), 1.0, places=5)

    def test_procrustes_matches_linear_convention(self):
        source = torch.randn(20, 3, dtype=torch.float64)
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        target = F.linear(source, expected)
        result = optimal_rotation(source, target)
        torch.testing.assert_close(F.linear(source, result), target)
        self.assert_proper(result)

    def test_reflection_is_excluded(self):
        reflection = torch.diag(torch.tensor([-1., 1., 1.]))
        self.assert_proper(proper_polar(reflection))

    def test_projected_adam_learns_rotation(self):
        rotations = TaskRotations(["seen"], 3)
        source = torch.randn(24, 3)
        expected = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = F.linear(source, expected)
        optimizer = torch.optim.Adam(rotations.parameters(), lr=0.05)
        initial = F.mse_loss(rotations(source, "seen"), target).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = F.mse_loss(rotations(source, "seen"), target)
            loss.backward()
            optimizer.step()
            rotations.project_()
            self.assert_proper(rotations.matrix_for("seen"))
        self.assertLess(loss.item(), initial * 0.01)

    def test_targets_cover_both_dimension_regimes(self):
        for dimension, classes in [(3, 4), (8, 3), (2, 6)]:
            with self.subTest(dimension=dimension, classes=classes):
                targets = generate_etf(dimension, classes)
                self.assertEqual(targets.shape, (classes, dimension))
                torch.testing.assert_close(targets.norm(dim=1), torch.ones(classes))
                if dimension >= classes - 1:
                    expected = (torch.eye(classes) - torch.ones(classes, classes) / classes) * classes / (classes - 1)
                    torch.testing.assert_close(targets @ targets.T, expected, atol=2e-6, rtol=2e-6)

    def test_alignment_and_regularizer_have_finite_gradients(self):
        rotations = TaskRotations(["task"], 3)
        raw = torch.randn(9, 3, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
        targets = generate_etf(3, 5)  # Several classes absent from this batch.
        loss = alignment_loss(rotations(raw, "task"), labels, targets)
        loss += 0.2 * procrustes_penalty(raw, labels, targets, rotations.matrix_for("task"))
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(rotations.matrix_for("task").grad.norm().item(), 0)

    def test_unseen_tasks_are_identity(self):
        rotations = TaskRotations(["seen"], 3)
        features = torch.randn(5, 3)
        self.assertIsNone(rotations.matrix_for("unseen"))
        torch.testing.assert_close(rotations(features, "unseen"), features)

    def test_real_evaluator_uses_the_same_rotation_convention(self):
        namespace = {"torch": torch, "F": F}
        extract_definitions(ROOT / "src/modeling.py", {"ImageClassifier"}, namespace)

        class Progress:
            def __init__(self, values, **unused):
                self.values = values

            def __iter__(self):
                return iter(self.values)

            def set_description(self, value):
                pass

        namespace.update(tqdm=SimpleNamespace(tqdm=Progress),
                         maybe_dictionarize=lambda batch: batch,
                         log=logging.getLogger("test"))
        extract_definitions(ROOT / "src/clip_eval.py", {"eval_single_dataset_preprocess_head"}, namespace)
        encoder = nn.Identity()
        encoder.train_preprocess = encoder.val_preprocess = None
        head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.eye(2))
        batch = [{"images": torch.eye(2), "labels": torch.ones(2, dtype=torch.long)}]
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        cfg = SimpleNamespace(device="cpu", fast_dev_run=False)
        evaluate = namespace["eval_single_dataset_preprocess_head"]
        with redirect_stdout(io.StringIO()):
            unrotated = evaluate(encoder, head, "task", cfg, dataloader=batch)
            rotated = evaluate(encoder, head, "task", cfg, rotation=rotation, dataloader=batch)
        self.assertEqual(unrotated["top1"], 0.5)
        self.assertEqual(rotated["top1"], 1.0)

    def test_checkpoint_restores_predictions_and_next_step(self):
        rotations = TaskRotations(["task"], 3)
        coefficient = nn.Parameter(torch.tensor([0.8]))
        optimizer = torch.optim.Adam([coefficient] + list(rotations.parameters()), lr=0.03)
        source = torch.randn(12, 3)
        targets = generate_etf(3, 4)
        labels = torch.arange(12) % 4

        def step(coef, rots, optim, refs):
            optim.zero_grad()
            loss = alignment_loss(rots(source * coef, "task"), labels, refs) + (coef - 0.4).square().sum()
            loss.backward()
            optim.step()
            rots.project_()

        step(coefficient, rotations, optimizer, targets)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "alignment.pt"
            save_alignment_checkpoint(path, coefficient, rotations, optimizer, 1, targets)
            restored = TaskRotations(["task"], 3)
            new_coef = nn.Parameter(torch.zeros(1))
            new_optimizer = torch.optim.Adam([new_coef] + list(restored.parameters()), lr=0.9)
            new_targets = torch.zeros_like(targets)
            epoch = load_alignment_checkpoint(path, new_coef, restored, new_optimizer, new_targets)
            self.assertEqual(epoch, 1)
            torch.testing.assert_close(restored(source * new_coef, "task"), rotations(source * coefficient, "task"))
            torch.testing.assert_close(new_targets, targets)
            step(coefficient, rotations, optimizer, targets)
            step(new_coef, restored, new_optimizer, new_targets)
            torch.testing.assert_close(new_coef, coefficient)
            torch.testing.assert_close(restored.matrix_for("task"), rotations.matrix_for("task"))
            wrong = TaskRotations(["another-task"], 3)
            with self.assertRaisesRegex(ValueError, "task names"):
                load_alignment_checkpoint(path, new_coef, wrong, new_optimizer, new_targets)
            legacy = torch.load(path, weights_only=True)
            del legacy["adaptation_method"]
            torch.save(legacy, path)
            with self.assertRaisesRegex(ValueError, "argmax pseudo-label"):
                load_alignment_checkpoint(path, new_coef, restored, new_optimizer, new_targets)

    def test_both_training_entry_points_end_to_end_on_cpu(self):
        """Run real main loops with tiny fixed models and in-memory datasets."""
        for filename in ("MDA.py", "MDA_generalization.py"):
            for weight in (0.0, 0.2):
                with self.subTest(script=filename, rotation_weight=weight), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                    self.run_entrypoint(filename, Path(folder), weight)

    def test_pseudo_label_training_ignores_ground_truth(self):
        for filename in ("MDA.py", "MDA_generalization.py"):
            states = []
            for mode in ("missing", "original", "permuted"):
                with self.subTest(script=filename, labels=mode), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                    torch.manual_seed(19)
                    states.append(self.run_entrypoint(filename, Path(folder), 0.2, mode))
            for state in states[1:]:
                torch.testing.assert_close(state["lambdas"], states[0]["lambdas"], rtol=0, atol=0)
                for name, matrix in state["rotations"].items():
                    torch.testing.assert_close(matrix, states[0]["rotations"][name], rtol=0, atol=0)

    def run_entrypoint(self, filename, folder, rotation_weight, label_mode="missing"):
        # Fake only external model/data IO; production training, loss, optimizer,
        # projection, checkpoint and evaluation forwarding execute unchanged.
        images = torch.randn(8, 3)
        labels = torch.arange(8) % 2
        base = nn.Linear(3, 3, bias=False)
        base.val_preprocess = None
        task_names = ["first", "second"]
        heads = {name: nn.Linear(3, 2, bias=False) for name in task_names + ["unseen"]}

        class Wrapper(nn.Module):
            def __init__(self, model, unused=None):
                super().__init__()
                self.model = model

            def forward(self, inputs):
                return self.model(inputs)

        def functionalize(model):
            params, names = [], []
            for name, param in list(model.named_parameters()):
                params.append(param)
                names.append(name)
                parent, _, leaf = name.rpartition(".")
                delattr(model.get_submodule(parent), leaf)
            return params, names

        def load_weights(model, names, params):
            for name, param in zip(names, params):
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent), leaf, param)

        class Vector:
            def __init__(self, vector=None, **unused):
                self.vector = vector if vector is not None else {"weight": torch.eye(3) * 0.1}

        evaluations = []
        predictions = []
        alignment_labels = []

        def entropy(logits):
            predictions.append(logits.detach().argmax(1))
            return -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)

        def check_alignment(features, pseudo_labels, targets):
            expected = predictions[-1] + 2 * (len(alignment_labels) % 2)
            torch.testing.assert_close(pseudo_labels, expected)
            alignment_labels.append(pseudo_labels.clone())
            return alignment_loss(features, pseudo_labels, targets)

        def check_procrustes(features, pseudo_labels, targets, matrix):
            torch.testing.assert_close(pseudo_labels, alignment_labels[-1])
            return procrustes_penalty(features, pseudo_labels, targets, matrix)

        def evaluate(encoder, head, name, cfg, rotation=None):
            evaluations.append((name, None if rotation is None else rotation.detach().clone()))
            with torch.no_grad():
                features = encoder(images)
                if rotation is not None:
                    features = F.linear(features, rotation)
                predictions = head(features).argmax(1)
            return {"top1": (predictions == labels).float().mean().item()}

        namespace = dict(torch=torch, nn=nn, F=F, logging=logging, log=logging.getLogger("test"),
                         get_classification_head=lambda cfg, name: heads[name],
                         load_weights=load_weights, make_functional=functionalize,
                         ModelWrapper=Wrapper, TaskVector=Vector,
                         CHECKPOINT_DIR=folder, DATA_DIR=folder, RESULTS_DIR=folder,
                         pretrained_model_path=lambda model: folder / "base.pt",
                         finetuned_model_path=lambda model, dataset: dataset,
                         check_parameterNamesMatch=lambda vectors: None,
                         compute_mda=lambda vectors, cfg: vectors[0],
                         timeit_context=nullcontext, TaskRotations=TaskRotations,
                         generate_etf=generate_etf, alignment_loss=check_alignment,
                         procrustes_penalty=check_procrustes,
                         save_alignment_checkpoint=save_alignment_checkpoint,
                         load_alignment_checkpoint=load_alignment_checkpoint,
                         eval_single_dataset_preprocess_head=evaluate,
                         softmax_entropy=entropy,
                         os=os, Path=Path, itertools=itertools, tqdm=lambda values: values,
                         pd=SimpleNamespace(DataFrame=lambda results: SimpleNamespace(to_csv=lambda *args, **kwargs: None)))
        extract_definitions(ROOT / "scripts" / filename, {"AdaMerging", "main"}, namespace)
        # The production main imports these two IO functions locally.
        common = SimpleNamespace(maybe_dictionarize=lambda batch: batch)
        training_batch = {"images": images}
        if label_mode != "missing":
            training_batch["labels"] = labels if label_mode == "original" else labels.flip(0)
        registry = SimpleNamespace(get_dataset=lambda *args, **kwargs: SimpleNamespace(
            classnames=["a", "b"], test_loader_shuffle=[training_batch]))
        cfg = SimpleNamespace(model="toy", device="cpu", sam_retraining=False,
                              datasets=task_names, seen_datasets=task_names, unseen_datasets=["unseen"],
                              num_workers=0, batch_size=8,
                              mda=SimpleNamespace(lambda_init=1.0, lr=0.02, betas=[0.9, 0.999],
                                                  weight_decay=0., epochs=3, batch_size=8, prototype_batch_size=4,
                                                  eval_interval=2, alignment_warmup_epochs=0,
                                                  alignment_alpha=1., rotation_weight=rotation_weight, resume=None))
        real_load = torch.load

        def model_load(path, *args, **kwargs):
            return deepcopy(base) if str(path).endswith("base.pt") else real_load(path, *args, **kwargs)

        with patch.dict("sys.modules", {"src.datasets.common": common, "src.datasets.registry": registry}), patch.object(torch, "load", model_load), redirect_stdout(io.StringIO()):
            namespace["main"](cfg)
        checkpoint_paths = list(folder.rglob("alignment.pt"))
        self.assertEqual(len(checkpoint_paths), 1)
        state = real_load(checkpoint_paths[0], weights_only=True)
        self.assertEqual(state["epoch"], 3)  # Final step saved even off interval.
        self.assertGreater((state["lambdas"] - 1).norm().item(), 1e-5)
        for matrix in state["rotations"].values():
            self.assert_proper(matrix)
            self.assertGreater((matrix - torch.eye(3)).norm().item(), 1e-4)
        for name, rotation in evaluations:
            if name == "unseen":
                self.assertIsNone(rotation)
            else:
                self.assertIsNotNone(rotation)
        self.assertTrue(any(rotation is not None and (rotation - torch.eye(3)).norm() > 1e-4 for _, rotation in evaluations))
        self.assertTrue(all(not param.requires_grad for head in heads.values() if head is not heads["unseen"] for param in head.parameters()))
        cfg.mda.resume = str(checkpoint_paths[0])
        cfg.mda.epochs = 5
        with patch.dict("sys.modules", {"src.datasets.common": common, "src.datasets.registry": registry}), patch.object(torch, "load", model_load), redirect_stdout(io.StringIO()):
            namespace["main"](cfg)
        resumed = real_load(checkpoint_paths[0], weights_only=True)
        self.assertEqual(resumed["epoch"], 5)
        for matrix in resumed["rotations"].values():
            self.assert_proper(matrix)
        self.assertEqual(len(alignment_labels), 10)
        return resumed


if __name__ == "__main__":
    unittest.main()
