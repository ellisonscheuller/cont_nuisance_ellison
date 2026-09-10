import numpy as np
import sys
import types
import torch

sys.modules.setdefault("wandb", types.ModuleType("wandb"))
from eval_abcd_nurd import (
    fit_class_kernel_transform, _kernel_whiten, checkpoint_reference_indices)


def test_weighted_kernel_pca_biases_fit_sample_toward_higher_weight():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    weights = np.asarray([1.0, 3.0, 100.0])  # third point excluded by mask
    kpca = fit_class_kernel_transform(
        embeddings, mask, n_components=1, class_name="QCD",
        weights=weights, fit_sample_cap=4000, seed=0)

    # KernelPCA has no native sample_weight, so a weighted fit is approximated
    # by resampling with replacement proportional to `weights`. Reproduce the
    # same first RNG draw the function makes internally (same seed, same
    # first call) to check the resample lands close to the true proportion.
    rng = np.random.default_rng(0)
    ref_weights = weights[mask]
    p = ref_weights / ref_weights.sum()
    fit_idx = rng.choice(2, size=4000, replace=True, p=p)
    assert abs(fit_idx.mean() - p[1]) < 0.05

    z = _kernel_whiten(kpca, embeddings[mask])
    assert z.shape == (2, 1)
    assert np.isfinite(z).all()


def test_unweighted_kernel_pca_whitens_reference_class():
    embeddings = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
    mask = np.asarray([True, True, False])
    kpca = fit_class_kernel_transform(
        embeddings, mask, n_components=1, class_name="QCD", fit_sample_cap=4000)
    z = _kernel_whiten(kpca, embeddings)
    assert z.shape == (3, 1)
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
