import numpy as np
import sys
import types
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from eval_abcd_nurd import (
    fit_class_transform, _linear_whiten, checkpoint_reference_indices)


def test_weighted_pca_biases_mean_toward_higher_weight():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    weights = np.asarray([1.0, 3.0, 100.0])  # third point excluded by mask
    mu, W = fit_class_transform(
        embeddings, mask, n_pca=None, class_name="QCD", weights=weights)

    # unweighted mean of the masked points is [1, 0]; 3x weight on [2, 0]
    # should pull mu[0] above 1.0.
    assert mu[0] > 1.0

    z = _linear_whiten(mu, W, embeddings[mask])
    assert z.shape == (2, 2)
    assert np.isfinite(z).all()


def test_unweighted_pca_whitens_reference_class():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    mu, W = fit_class_transform(embeddings, mask, n_pca=None, class_name="QCD")
    z = _linear_whiten(mu, W, embeddings)
    assert z.shape == (3, 2)
    assert np.isfinite(z).all()


def test_checkpoint_indices_are_explicitly_moved_to_cpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = {
        "preprocessing": {
            "data_signature": {"n_events": 5, "token": "sample"},
            "weighting": {"generator": {
                "effective_physics_weight_sha256": "weights",
            }},
            "split": {
                "train_indices": torch.tensor([0, 2, 4], device=device),
                "validation_indices": torch.tensor([1, 3], device=device),
            },
        },
    }
    fit, selection = checkpoint_reference_indices(
        checkpoint,
        {"n_events": 5, "token": "sample"},
        {"effective_physics_weight_sha256": "weights"},
    )
    assert fit.dtype == np.int64
    assert selection.dtype == np.int64
    assert fit.tolist() == [0, 2, 4]
    assert selection.tolist() == [1, 3]
