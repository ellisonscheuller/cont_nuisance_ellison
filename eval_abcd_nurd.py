"""
ABCD eval for the NURD contrastive checkpoint (hlt_nurd_con).

Axis 1: AE reco loss (HLTAutoencoder, loaded from separate ae_ckpt)
Axis 2: Mahalanobis distance in a kernel-PCA-whitened NURD latent space
        (default) OR 1-P(QCD) classifier logit score (--axis2_logit)

All PCA-embedding plots (pca_kde_embeddings.png, pca_corner.png,
hist2d_pca_md_kde.png) are KDE contours rather than per-point scatter, and
use KernelPCA (--pca_kernel/--pca_gamma) rather than linear PCA throughout.

Usage
-----
python eval_abcd_nurd.py \
    --ckpt      /eos/user/e/escheull/ssl_checkpoints/hlt/hlt/hlt_nurd_run_epoch_critic/checkpoint_main.pth.tar \
    --ae_ckpt   /eos/user/e/escheull/ssl_checkpoints/hlt/hlt/ae_pretrain/checkpoint_ae.pth \
    --test_pt   /eos/user/e/escheull/smcocktail_1M_noZB/hlt_smcocktail_test.pt \
    [--class_names DY,QCD,WJetsToLNu,...]  # ordered by label index \
    [--signal_pt /eos/user/e/escheull/signal_pt/hlt_signal.pt] \
    [--signal_class_names ttH,HHbbgg,tttt]  # ordered by label index in signal_pt \
    [--baseline_ckpt checkpoint_main_nodecorr.pth.tar]  # lambda_info=0 comparison \
    [--n_pca 6] \
    [--axis2_logit]  # use 1-P(QCD) instead of MD
    [--outdir /eos/user/e/escheull/abcd_outputs] \
    [--wandb_run_name nurd_abcd_v1]
"""
import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, gaussian_kde
from sklearn.decomposition import KernelPCA
from matplotlib.lines import Line2D

from models.hlt_con import HLTContrastiveModel
from models.hlt_autoencoder import HLTAutoencoder
from utils.hlt_weights import file_sha256, load_generator_weights, sample_signature


# ── ABCD helpers (identical to eval_abcd.py) ─────────────────────────────────

def weighted_quantile(values, q, weights):
    """Weighted quantile: threshold where cumulative weight reaches q * total_weight."""
    sorter = np.argsort(values)
    sv = values[sorter]
    sw = weights[sorter]
    cumw = np.cumsum(sw)
    idx = np.searchsorted(cumw, q * cumw[-1])
    return sv[np.clip(idx, 0, len(sv) - 1)]


ABCD_REGIONS = ("A", "B", "C", "D")


def abcd_region_statistics_at_thresholds(
        loss_1, loss_2, thresh_1, thresh_2, weights=None):
    """Return yield, sumw2, raw count and effective count in each region.

    For unit-weight legacy samples, ``effective_count`` is exactly the raw
    event count.  For weighted samples it is ``(sum w)^2 / sum(w^2)``, which is
    the count relevant to statistical precision.
    """
    loss_1 = np.asarray(loss_1)
    loss_2 = np.asarray(loss_2)
    if weights is None:
        event_weights = np.ones(len(loss_1), dtype=np.float64)
    else:
        event_weights = np.asarray(weights, dtype=np.float64)
    if loss_1.shape != loss_2.shape or event_weights.shape != loss_1.shape:
        raise ValueError("ABCD scores and weights must have identical shapes.")

    high_1 = loss_1 > thresh_1
    high_2 = loss_2 > thresh_2
    masks = {
        "A": high_1 & high_2,
        "B": high_1 & ~high_2,
        "C": ~high_1 & high_2,
        "D": ~high_1 & ~high_2,
    }
    statistics = {}
    for region, mask in masks.items():
        selected = event_weights[mask]
        event_yield = float(selected.sum())
        sumw2 = float(np.square(selected).sum())
        effective_count = (
            event_yield * event_yield / sumw2 if sumw2 > 0.0 else 0.0)
        statistics[region] = {
            "yield": event_yield,
            "sumw2": sumw2,
            "raw_count": int(mask.sum()),
            "effective_count": float(effective_count),
        }
    return statistics


def abcd_statistics(loss_1, loss_2, percent_1, percent_2, weights=None):
    """Select percentile thresholds and return per-region statistics."""
    if weights is not None:
        thresh_1 = weighted_quantile(loss_1, percent_1, weights)
        thresh_2 = weighted_quantile(loss_2, percent_2, weights)
    else:
        thresh_1 = np.quantile(loss_1, percent_1)
        thresh_2 = np.quantile(loss_2, percent_2)
    statistics = abcd_region_statistics_at_thresholds(
        loss_1, loss_2, thresh_1, thresh_2, weights=weights)
    return thresh_1, thresh_2, statistics


def abcd_yields(statistics):
    return tuple(statistics[region]["yield"] for region in ABCD_REGIONS)


def statistically_valid_regions(statistics, minimums):
    """Require adequate effective statistics independently in A, B, C and D."""
    return all(
        statistics[region]["effective_count"] >= float(minimums[region])
        for region in ABCD_REGIONS)


def closure_ratio_and_uncertainty(statistics):
    """ABCD predicted/observed ratio with weighted-Poisson sumw2 error."""
    A, B, C, D = abcd_yields(statistics)
    if min(A, B, C, D) <= 0.0:
        return np.nan, np.nan
    ratio = (B * C) / (D * A)
    relative_variance = sum(
        statistics[region]["sumw2"]
        / (statistics[region]["yield"] ** 2)
        for region in ABCD_REGIONS)
    return ratio, abs(ratio) * np.sqrt(relative_variance)


def abcd_counts(loss_1, loss_2, percent_1, percent_2, weights=None):
    thresh_1, thresh_2, statistics = abcd_statistics(
        loss_1, loss_2, percent_1, percent_2, weights=weights)
    A, B, C, D = abcd_yields(statistics)
    if weights is None:
        A, B, C, D = (int(A), int(B), int(C), int(D))
    return thresh_1, thresh_2, A, B, C, D


def abcd_counts_at_thresholds(loss_1, loss_2, thresh_1, thresh_2, weights=None):
    """ABCD yields at fixed thresholds selected on another sample."""
    statistics = abcd_region_statistics_at_thresholds(
        loss_1, loss_2, thresh_1, thresh_2, weights=weights)
    return abcd_yields(statistics)


def nonclosure_A(A, B, C, D, eps=1e-8):
    A_hat = (B * C) / max(D, eps)
    if A_hat <= 0:
        return np.inf, A_hat
    return (A - A_hat) / A_hat, A_hat


def profile_plot(ax, x, y, nbins=30, logx=False, min_per_bin=20, label="mean ± SE"):
    x, y = np.asarray(x), np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if logx:
        m &= (x > 0)
    x, y = x[m], y[m]
    xu = np.log10(x) if logx else x
    lo, hi = float(xu.min()), float(xu.max())
    if lo == hi:
        hi = np.nextafter(hi, np.inf)
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean, _, _ = binned_statistic(xu, y, statistic="mean", bins=edges)
    std,  _, _ = binned_statistic(xu, y, statistic="std",  bins=edges)
    cnt,  _, _ = binned_statistic(xu, y, statistic="count",bins=edges)
    sem = std / np.sqrt(np.maximum(cnt, 1))
    good = cnt >= min_per_bin
    xc = centers[good]
    xplot = (10.0 ** xc) if logx else xc
    if logx:
        ax.set_xscale("log")
    ax.errorbar(xplot, mean[good], yerr=sem[good],
                fmt="o", ms=3, lw=1, capsize=2, label=label)
    ax.grid(alpha=0.3)
    return {"x": xplot, "mean": mean[good], "sem": sem[good], "count": cnt[good]}


# ── Class names / colors / KDE plotting ─────────────────────────────────────

# Matches double_disco_hlt's data_collide2v_ptrawv1.yaml / _100k.yaml samples:
# order (11 classes, QCD kept at index 1) -- NOT the old 4-class DY/QCD/TT/
# WJets grouping. Callers should still pass --class_names explicitly so the
# legend always reflects the actual data config used; this is the fallback
# for when it's omitted.
_DEFAULT_CLASS_NAMES = {
    0: "DY", 1: "QCD", 2: "WJetsToLNu", 3: "WJetsToQQ", 4: "ZJetsToQQ",
    5: "ZJetsTobb", 6: "ZJetsTocc", 7: "ZJetsTovv", 8: "TTHadronic",
    9: "TTLeptonic", 10: "TTSemiLeptonic",
}


def resolve_class_names(names_arg, default_names=None):
    """Ordered, comma-separated --class_names into a {label: name} dict.

    Falls back to `default_names` (the current 11-class scheme) when
    omitted.
    """
    if names_arg:
        names = [n.strip() for n in names_arg.split(",") if n.strip()]
        return {i: name for i, name in enumerate(names)}
    return dict(default_names or _DEFAULT_CLASS_NAMES)


def build_class_colors(class_names, cmap_name=None):
    """Deterministic label -> color map sized to the number of classes."""
    labels = sorted(class_names)
    if cmap_name is None:
        cmap_name = "tab10" if len(labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    return {label: cmap(i % cmap.N) for i, label in enumerate(labels)}


def _kernel_whiten(kpca, embeddings, chunk_size=20_000):
    """Project through a fitted KernelPCA and whiten by sqrt(eigenvalue).

    This is the kernel-PCA generalization of linear PCA whitening
    (z = (x - mu) @ V / sqrt(L)): dividing each kernel-PCA component by the
    sqrt of its eigenvalue gives unit-variance coordinates in which the
    squared norm is the kernel-space analogue of Mahalanobis distance.

    KernelPCA.transform materializes a dense (n_embeddings, n_fit_samples)
    kernel matrix; at 1M-event scale that would be tens of GB in one call, so
    this chunks over `embeddings` (transform cost is linear in n_embeddings).
    """
    scale = np.sqrt(np.clip(kpca.eigenvalues_, 1e-12, None))
    n = embeddings.shape[0]
    if n <= chunk_size:
        return kpca.transform(embeddings) / scale
    parts = [
        kpca.transform(embeddings[start:start + chunk_size])
        for start in range(0, n, chunk_size)
    ]
    return np.concatenate(parts, axis=0) / scale


def fit_class_kernel_transform(embeddings, mask, n_components, class_name,
                                kernel="rbf", gamma=None, fit_sample_cap=5000,
                                weights=None, seed=42):
    """Fit a KernelPCA whitening transform on embeddings[mask].

    KernelPCA fit/transform cost is quadratic in the fit-set size, so the
    reference class is subsampled to `fit_sample_cap` points before fitting
    (transform() is then applied to the full sample, which is linear in N).
    KernelPCA has no native sample_weight support; a weighted fit is
    approximated by resampling the reference points with replacement
    proportional to `weights` before fitting -- exact for the unweighted case
    used by the legacy (unit-weight) eval jobs.
    """
    ref = embeddings[mask]
    n_ref = ref.shape[0]
    rng = np.random.default_rng(seed)

    if weights is None:
        fit_n = min(n_ref, int(fit_sample_cap))
        fit_idx = rng.choice(n_ref, size=fit_n, replace=False)
    else:
        ref_weights = np.asarray(weights, dtype=np.float64)[mask]
        if not np.isfinite(ref_weights).all() or np.any(ref_weights < 0):
            raise ValueError("Reference weights must be finite and non-negative.")
        weight_sum = ref_weights.sum()
        if weight_sum <= 0:
            raise ValueError(f"Reference class {class_name} has zero total weight.")
        # Not capped by n_ref: sampling with replacement is exactly how a
        # small/rare reference class still gets a full fit_sample_cap-sized
        # weighted fit set.
        fit_n = int(fit_sample_cap)
        fit_idx = rng.choice(n_ref, size=fit_n, replace=True, p=ref_weights / weight_sum)
    fit_points = ref[fit_idx]

    requested = int(n_components) if n_components else ref.shape[1]
    n_components_eff = max(1, min(requested, fit_points.shape[0] - 1))
    print(f"  Fitting KernelPCA ({kernel}) whitening on {n_ref} {class_name} "
          f"events (fit sample={fit_n}, components={n_components_eff})...", flush=True)

    kpca = KernelPCA(n_components=n_components_eff, kernel=kernel, gamma=gamma,
                      fit_inverse_transform=False, random_state=seed)
    kpca.fit(fit_points)
    return kpca


def plot_kde_contours(ax, groups, log_x=False, log_y=False,
                       fit_sample_cap=20_000, seed=42,
                       level_fracs=(0.05, 0.15, 0.3, 0.5, 0.7, 0.88),
                       min_events=50):
    """Draw one KDE contour set per group on a shared grid.

    groups: list of (name, color, x, y). Returns legend handles so callers
    can build a legend without per-point scatter artists.
    """
    rng = np.random.default_rng(seed)
    prepared = []
    for name, color, x, y in groups:
        x = np.asarray(x); y = np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y)
        if log_x:
            valid &= x > 0
        if log_y:
            valid &= y > 0
        tx = np.log10(x[valid]) if log_x else x[valid]
        ty = np.log10(y[valid]) if log_y else y[valid]
        if tx.shape[0] >= min_events:
            prepared.append((name, color, tx, ty))
    if not prepared:
        return []

    all_x = np.concatenate([tx for _, _, tx, _ in prepared])
    all_y = np.concatenate([ty for _, _, _, ty in prepared])
    xi, yi = np.mgrid[all_x.min():all_x.max():200j, all_y.min():all_y.max():200j]
    xplot = 10 ** xi if log_x else xi
    yplot = 10 ** yi if log_y else yi

    handles = []
    for name, color, tx, ty in prepared:
        if tx.shape[0] > fit_sample_cap:
            idx = rng.choice(tx.shape[0], fit_sample_cap, replace=False)
            tx, ty = tx[idx], ty[idx]
        kde = gaussian_kde(np.vstack([tx, ty]))
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        levels = zi.max() * np.array(level_fracs)
        ax.contour(xplot, yplot, zi, levels=levels, colors=color,
                   alpha=0.7, linewidths=1.5)
        handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=name))
    return handles


# ── Model loading ─────────────────────────────────────────────────────────────

def load_nurd_model(ckpt_path, device):
    """Load HLTContrastiveModel from NURD main checkpoint."""
    # Keep checkpoint metadata (especially saved split indices) on CPU. Model
    # parameters are copied to ``device`` by load_state_dict below.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    # num_tokens: infer from state dict (linear attn projection shape)
    sd = ckpt["state_dict_model"]
    e_proj_key = next((k for k in sd if k.endswith(".attn.e.weight")), None)
    if e_proj_key is not None:
        num_tokens = sd[e_proj_key].shape[1] - 1   # -1 for CLS token
    else:
        num_tokens = cfg.get("linear_dim", 100)     # fallback

    # num_classes: infer from classifier weight
    num_classes = sd["classifier.weight"].shape[0]

    model = HLTContrastiveModel(
        num_classes=num_classes,
        embed_size=cfg["embed_size"],
        latent_dim=cfg["latent_dim"],
        proj_dim=cfg["proj_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        linear_dim=cfg["linear_dim"],
        num_tokens=num_tokens,
        dropout=cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded HLTContrastiveModel: latent_dim={cfg['latent_dim']}, "
          f"num_layers={cfg['num_layers']}, num_tokens={num_tokens}", flush=True)
    return model, ckpt


def load_ae(ae_ckpt_path, ae_scaler, device):
    """Load HLTAutoencoder from its own checkpoint."""
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_cfg  = ae_ckpt.get("ae_config", {
        "features": None, "latent_dim": 16,
        "encoder_config": {"nodes": [512, 256]},
        "decoder_config": {"nodes": [256, 512, None]},
        "alpha": 1.0,
    })
    if ae_cfg["features"] is None:
        first_w = ae_ckpt["ae"][next(iter(ae_ckpt["ae"]))]
        ae_cfg["features"] = first_w.shape[1]
    ae = HLTAutoencoder(ae_cfg).to(device)
    ae.load_state_dict(ae_ckpt["ae"])
    ae.eval()
    print(f"Loaded HLTAutoencoder: features={ae_cfg['features']}, "
          f"latent={ae_cfg['latent_dim']}", flush=True)
    return ae


# ── Inference ─────────────────────────────────────────────────────────────────

def compute_ae_scores(ae, ae_scaler, pt_path, device, batch_size=4096):
    """AE reco loss (MSE) per event using obj features from pt_path."""
    mu  = ae_scaler["mu"].cpu().numpy()
    std = ae_scaler["std"].cpu().numpy()

    raw = torch.load(pt_path, map_location="cpu")
    obj = torch.nan_to_num(
        raw["obj"][:, :, :4].reshape(raw["obj"].shape[0], -1).float(),
        nan=0.0, posinf=0.0, neginf=0.0).numpy()
    obj_norm = torch.from_numpy(((obj - mu) / (std + 1e-8)).astype(np.float32))
    N = obj_norm.shape[0]
    print(f"  AE inference on {N} events...", flush=True)

    scores = []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = obj_norm[i0:i0 + batch_size].to(device)
            recon, _ = ae(xb)
            mse = ((recon - xb) ** 2).mean(dim=1)
            scores.append(mse.cpu())
    return torch.cat(scores).numpy().astype(np.float32)


def embed_pf(model, pt_path, device, batch_size=512, return_logits=False):
    """Run NURD encoder on PF candidates; return (latents [N,D], labels [N]).
    If return_logits=True, returns (latents, logits [N,C], labels)."""
    raw    = torch.load(pt_path, map_location="cpu")
    pf     = torch.nan_to_num(raw["pf"], nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].numpy()
    N = pf.shape[0]
    print(f"  Encoder inference on {N} events from {pt_path}...", flush=True)

    latents, logits_list = [], []
    with torch.no_grad():
        for i0 in range(0, N, batch_size):
            xb = pf[i0:i0 + batch_size].to(device)
            latent, logit = model(xb)
            latents.append(latent.cpu())
            if return_logits:
                logits_list.append(logit.cpu())
    if return_logits:
        return (torch.cat(latents, dim=0).numpy(),
                torch.cat(logits_list, dim=0).numpy(),
                labels)
    return torch.cat(latents, dim=0).numpy(), labels


def compute_logit_axis2(logits, qcd_label=1):
    """1 - P(QCD) from classifier logits — higher means more anomalous."""
    probs = F.softmax(torch.from_numpy(logits.astype(np.float32)), dim=1).numpy()
    return (1.0 - probs[:, qcd_label]).astype(np.float32)


def compute_md_scores(model, pt_path, device, batch_size=512, n_pca=None,
                      bkg_labels=None, reference_pt=None,
                      reference_weights=None, reference_fit_indices=None,
                      class_names=None, pca_kernel="rbf", pca_gamma=None,
                      pca_kernel_fit_samples=5000):
    """
    Embed all events, fit KernelPCA whitening per background class, return MD scores.

    bkg_labels: list of class labels to use as reference.
      [1]       (default) → QCD-only MD.
      [0, 1, 3] → min-MD across three classes (element-wise minimum).

    Returns (md [N], labels [N], kpca_qcd, latents [N,D], class_transforms).
    class_transforms is a list of (label, kpca) — reuse for signal inference.
    """
    class_names = class_names or {}
    if bkg_labels is None:
        bkg_labels = [1]

    latents, labels = embed_pf(model, pt_path, device, batch_size)
    if reference_pt:
        print(f"  Fitting MD reference on independent sample: {reference_pt}", flush=True)
        reference_latents, reference_labels = embed_pf(
            model, reference_pt, device, batch_size)
    else:
        reference_latents, reference_labels = latents, labels

    if reference_weights is not None:
        reference_weights = np.asarray(reference_weights, dtype=np.float64)
        if reference_weights.shape[0] != reference_labels.shape[0]:
            raise ValueError(
                "Reference-weight length does not match the reference sample.")

    fit_mask = np.ones(reference_labels.shape[0], dtype=bool)
    if reference_fit_indices is not None:
        fit_mask[:] = False
        fit_mask[np.asarray(reference_fit_indices, dtype=np.int64)] = True

    class_transforms = []
    for cls in bkg_labels:
        mask = (reference_labels == cls) & fit_mask
        if mask.sum() < 10:
            print(f"  WARNING: class {cls} has only {mask.sum()} events — skipping", flush=True)
            continue
        kpca = fit_class_kernel_transform(
            reference_latents, mask, n_pca,
            class_names.get(cls, str(cls)),
            kernel=pca_kernel, gamma=pca_gamma,
            fit_sample_cap=pca_kernel_fit_samples,
            weights=reference_weights)
        class_transforms.append((cls, kpca))

    if not class_transforms:
        raise RuntimeError("No background classes with enough events.")

    md_per_class = []
    for cls, kpca_c in class_transforms:
        z_c = _kernel_whiten(kpca_c, latents)
        md_per_class.append((z_c * z_c).sum(axis=1))
    md = np.stack(md_per_class, axis=0).min(axis=0).astype(np.float32)

    if len(class_transforms) > 1:
        print(f"  Min-MD across classes {[c for c, _ in class_transforms]}", flush=True)

    qcd_entry = next((t for t in class_transforms if t[0] == 1), class_transforms[0])
    kpca_qcd = qcd_entry[1]

    return (md, labels, kpca_qcd, latents, class_transforms,
            reference_latents, reference_labels)


def load_physics_weights(weight_path, pt_path, qcd_label):
    """Load evaluation weights through the same validated training contract."""
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    labels = raw["label"].long().reshape(-1)
    weights, metadata = load_generator_weights(
        weight_path, labels, qcd_label=qcd_label, sample=raw)
    signature = sample_signature(raw)
    return weights.numpy().astype(np.float64), metadata, signature


def checkpoint_reference_indices(checkpoint, signature, weight_metadata):
    """Validate and recover disjoint MD-fit and threshold-selection roles."""
    preprocessing = checkpoint.get("preprocessing", {})
    if preprocessing.get("data_signature") != signature:
        raise ValueError(
            "Held-out reference sample does not match the training checkpoint.")
    expected_checksum = preprocessing.get(
        "weighting", {}).get("generator", {}).get(
            "effective_physics_weight_sha256")
    actual_checksum = weight_metadata.get("effective_physics_weight_sha256")
    if not expected_checksum or expected_checksum != actual_checksum:
        raise ValueError(
            "Held-out reference weights do not match the training checkpoint.")
    split = preprocessing.get("split", {})
    fit_indices = torch.as_tensor(
        split.get("train_indices", []), dtype=torch.long
    ).detach().cpu().numpy().astype(np.int64, copy=False)
    selection_indices = torch.as_tensor(
        split.get("validation_indices", []), dtype=torch.long
    ).detach().cpu().numpy().astype(np.int64, copy=False)
    combined = np.concatenate([fit_indices, selection_indices])
    if (combined.size != signature["n_events"]
            or np.unique(combined).size != combined.size
            or combined.min(initial=0) < 0
            or combined.max(initial=-1) >= signature["n_events"]):
        raise ValueError(
            "Checkpoint train/validation indices do not partition the reference sample.")
    return fit_indices, selection_indices


# ── Main ──────────────────────────────────────────────────────────────────────

def ABCD(config):
    if os.environ.get("WANDB_MODE", "").lower() != "offline":
        print("Logging in to wandb...", flush=True)
        wandb.login()
    resume_id = config.get("resume_run_id", None)
    wandb.init(project=config.get("wandb_project", "AE vs. Contrastive ABCD"),
               name=config.get("wandb_run_name", None),
               id=resume_id,
               resume="allow" if resume_id else None,
               settings=wandb.Settings(_disable_stats=True),
               config=config)
    run_name = wandb.run.name
    print(f"Run name: {run_name}", flush=True)

    outdir   = config.get("outdir", "outputs_abcd")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load models ───────────────────────────────────────────────────────────
    model, main_ckpt = load_nurd_model(config["ckpt"], device)
    expected_ae_sha256 = main_ckpt.get("ae_checkpoint_sha256")
    if not expected_ae_sha256 or file_sha256(
            config["ae_ckpt"]) != expected_ae_sha256:
        raise ValueError(
            "AE checkpoint does not match the exact AE used for NURD training.")
    ae_scaler = main_ckpt["ae_scaler"]
    ae = load_ae(config["ae_ckpt"], ae_scaler, device)
    qcd_label = int(config.get("qcd_label", 1))
    class_names  = resolve_class_names(config.get("class_names"))
    class_colors = build_class_colors(class_names)
    sig_class_names = (
        resolve_class_names(config.get("signal_class_names"))
        if config.get("signal_class_names") else None)

    reference_pt = config.get("reference_pt")
    reference_physics = reference_signature = None
    reference_fit_indices = selection_indices = None
    if reference_pt:
        if not config.get("reference_weight_path"):
            raise ValueError(
                "Held-out evaluation requires --reference_weight_path.")
        if os.path.realpath(reference_pt) == os.path.realpath(config["test_pt"]):
            raise ValueError(
                "Held-out reference and report files must be different samples.")
        reference_physics, reference_metadata, reference_signature = (
            load_physics_weights(
                config["reference_weight_path"], reference_pt, qcd_label))
        reference_fit_indices, selection_indices = checkpoint_reference_indices(
            main_ckpt, reference_signature, reference_metadata)

    test_physics = None
    if config.get("gen_weight_path"):
        test_physics, _test_metadata, _test_signature = load_physics_weights(
            config["gen_weight_path"], config["test_pt"], qcd_label)

    # ── AE scores ─────────────────────────────────────────────────────────────
    print("Computing AE scores (bkg)...", flush=True)
    ae_bkg = compute_ae_scores(ae, ae_scaler, config["test_pt"], device)
    ae_reference = None
    if reference_pt:
        print("Computing AE scores (threshold-selection reference)...", flush=True)
        ae_reference = compute_ae_scores(
            ae, ae_scaler, reference_pt, device)

    # free AE GPU memory before running encoder
    del ae
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── axis 2 scores (MD or logit) ───────────────────────────────────────────
    use_logit  = config.get("axis2_logit", False)
    axis2_label = "1-P(QCD)" if use_logit else "NURD Contrastive score (MD)"
    axis2_log_scale = not use_logit   # MD → log y; logit ∈ [0,1] → linear y

    pca_kernel = config.get("pca_kernel", "rbf")
    pca_gamma = config.get("pca_gamma")
    pca_kernel_fit_samples = int(config.get("pca_kernel_fit_samples", 5000))

    # always need latents for PCA embedding plots; get logits too when requested
    if use_logit:
        print("Axis 2 = 1-P(QCD) logit mode.", flush=True)
        latents_all, logits_all, labels = embed_pf(
            model, config["test_pt"], device, return_logits=True)
        con_bkg = compute_logit_axis2(logits_all, qcd_label=qcd_label)
        class_transforms = []   # not used in logit mode
        md_kpca = None
        if reference_pt:
            reference_latents, reference_logits, reference_labels = embed_pf(
                model, reference_pt, device, return_logits=True)
            reference_axis2 = compute_logit_axis2(
                reference_logits, qcd_label=qcd_label)
    else:
        bkg_labels = [0, 1, 3] if config.get("min_md") else [1]
        if config.get("min_md"):
            print("Min-MD mode: axis 2 = min-MD across labels "
                  f"{bkg_labels}", flush=True)
        print("Computing contrastive MD scores (bkg)...", flush=True)
        (con_bkg, labels, md_kpca, latents_all, class_transforms,
         reference_latents, reference_labels) = compute_md_scores(
            model, config["test_pt"], device,
            n_pca=config.get("n_pca"),
            bkg_labels=bkg_labels,
            reference_pt=reference_pt,
            reference_weights=reference_physics,
            reference_fit_indices=reference_fit_indices,
            class_names=class_names,
            pca_kernel=pca_kernel,
            pca_gamma=pca_gamma,
            pca_kernel_fit_samples=pca_kernel_fit_samples,
        )
        if reference_pt:
            reference_md = []
            for _cls, kpca_c in class_transforms:
                transformed = _kernel_whiten(kpca_c, reference_latents)
                reference_md.append((transformed * transformed).sum(axis=1))
            reference_axis2 = np.stack(reference_md, axis=0).min(axis=0)

    if len(con_bkg) != len(ae_bkg):
        raise ValueError(f"Length mismatch: contrastive {len(con_bkg)} vs AE {len(ae_bkg)}")

    # ── mask ──────────────────────────────────────────────────────────────────
    mask = np.isfinite(ae_bkg) & np.isfinite(con_bkg) & (ae_bkg > 0)
    axis1_bkg = ae_bkg[mask]
    axis2_bkg = con_bkg[mask]
    labels_masked  = labels[mask]
    latents_masked = latents_all[mask]
    print(f"Events after masking: {mask.sum()}", flush=True)

    if not use_logit:
        emb_pca   = _kernel_whiten(md_kpca, latents_masked)
        n_pca     = emb_pca.shape[1]
        axis2_pca = axis2_bkg
    else:
        # still compute a KernelPCA of the latent for the embedding plots
        _pca_fit = fit_class_kernel_transform(
            latents_masked, labels_masked == qcd_label,
            min(6, latents_masked.shape[1]), class_names.get(qcd_label, "QCD"),
            kernel=pca_kernel, gamma=pca_gamma,
            fit_sample_cap=pca_kernel_fit_samples)
        emb_pca   = _kernel_whiten(_pca_fit, latents_masked)
        n_pca     = emb_pca.shape[1]
        axis2_pca = axis2_bkg   # same axis in logit mode

    qcd_only  = labels_masked == qcd_label
    axis1_qcd = axis1_bkg[qcd_only]
    axis2_qcd = axis2_bkg[qcd_only]
    print(f"QCD events for ABCD: {qcd_only.sum()}", flush=True)

    # ── gen weights (QCD only, for weighted ABCD) ─────────────────────────────
    gen_weights_qcd = None
    if test_physics is not None:
        gen_weights_qcd = test_physics[mask][qcd_only]
        print(f"Gen weights loaded: {len(gen_weights_qcd)} QCD weights "
              f"(min={gen_weights_qcd.min():.3e}, max={gen_weights_qcd.max():.3e})", flush=True)

    selection_axis1 = selection_axis2 = selection_weights = None
    if reference_pt:
        selection_mask = np.zeros(len(reference_axis2), dtype=bool)
        selection_mask[selection_indices] = True
        selection_mask &= (
            np.isfinite(ae_reference)
            & np.isfinite(reference_axis2)
            & (ae_reference > 0)
            & (reference_labels == qcd_label)
        )
        selection_axis1 = ae_reference[selection_mask]
        selection_axis2 = reference_axis2[selection_mask]
        selection_weights = reference_physics[selection_mask]
        print(
            "Held-out protocol: MD fit on "
            f"{len(reference_fit_indices)} saved training rows; thresholds selected "
            f"on {selection_mask.sum()} saved validation QCD rows; independent test "
            "is report-only.", flush=True)

    # ── signal (optional) ─────────────────────────────────────────────────────
    # signal_pt may bundle several signal processes (e.g. ttH/HH/tttt) as
    # distinct label values in the same .pt file; sig_labels_masked keeps
    # those apart so each is plotted as its own named/colored group instead
    # of one pooled blob.
    sig_axis1 = sig_axis2 = sig_axis2_pca = None
    sig_latents_masked = sig_emb_pca = sig_labels_masked = None
    sig_unique_labels = []
    sig_class_colors = {}
    if config.get("signal_pt"):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("Running signal inference...", flush=True)

        ae_sig = load_ae(config["ae_ckpt"], ae_scaler, device)
        sig_ae = compute_ae_scores(ae_sig, ae_scaler, config["signal_pt"], device)
        del ae_sig

        if use_logit:
            sig_latents, sig_logits, sig_raw_labels = embed_pf(
                model, config["signal_pt"], device, return_logits=True)
            sig_con = compute_logit_axis2(sig_logits, qcd_label=config.get("qcd_label", 1))
        else:
            sig_latents, sig_raw_labels = embed_pf(model, config["signal_pt"], device)
            sig_mds = []
            for cls, kpca_c in class_transforms:
                z_c = _kernel_whiten(kpca_c, sig_latents)
                sig_mds.append((z_c * z_c).sum(axis=1))
            sig_con = np.stack(sig_mds, axis=0).min(axis=0).astype(np.float32)

        sig_mask = np.isfinite(sig_ae) & np.isfinite(sig_con) & (sig_ae > 0)
        sig_axis1          = sig_ae[sig_mask]
        sig_axis2          = sig_con[sig_mask]
        sig_latents_masked = sig_latents[sig_mask]
        sig_labels_masked  = sig_raw_labels[sig_mask]
        if not use_logit:
            sig_emb_pca   = _kernel_whiten(md_kpca, sig_latents_masked)
            sig_axis2_pca = (sig_emb_pca * sig_emb_pca).sum(axis=1).astype(np.float32)
        else:
            sig_emb_pca   = _kernel_whiten(_pca_fit, sig_latents_masked)
            sig_axis2_pca = sig_axis2
        print(f"Signal events after masking: {sig_mask.sum()}", flush=True)

        sig_unique_labels = sorted(int(v) for v in np.unique(sig_labels_masked).tolist())
        if sig_class_names is None:
            sig_class_names = {lbl: f"Signal{lbl}" for lbl in sig_unique_labels}
        else:
            for lbl in sig_unique_labels:
                sig_class_names.setdefault(lbl, f"Signal{lbl}")
        sig_class_colors = build_class_colors(
            {lbl: sig_class_names[lbl] for lbl in sig_unique_labels}, cmap_name="Dark2")

    # ── ABCD scan ─────────────────────────────────────────────────────────────
    percent = np.linspace(0.50, 0.98, 48)
    min_A   = int(config.get("min_A", 50))
    min_B   = int(config.get("min_B", 50))
    min_C   = int(config.get("min_C", 50))
    min_D   = int(config.get("min_D", 500))
    statistically_valid_closure = bool(
        config.get("statistically_valid_closure", False))
    region_minimums = {
        "A": min_A, "B": min_B, "C": min_C, "D": min_D,
    }
    if statistically_valid_closure:
        count_basis = (
            "effective event counts (sumw)^2/sumw2"
            if gen_weights_qcd is not None else "raw event counts")
        print(
            "Statistically valid closure enabled: requiring "
            f"A>={min_A}, B>={min_B}, C>={min_C}, D>={min_D} using "
            f"{count_basis}.", flush=True)
    nc_grid = np.full((len(percent), len(percent)), np.nan)
    grid_rejected_points = 0

    oracle_best = {"nonclosure": np.inf}
    for i, p1 in enumerate(percent):
        for j, p2 in enumerate(percent):
            t1, t2, statistics = abcd_statistics(
                axis1_qcd, axis2_qcd, p1, p2,
                weights=gen_weights_qcd)
            A, B, C, D = abcd_yields(statistics)
            valid = (
                statistically_valid_regions(statistics, region_minimums)
                if statistically_valid_closure
                else A >= min_A and D >= min_D)
            if not valid:
                grid_rejected_points += 1
                continue
            nc, A_hat = nonclosure_A(A, B, C, D)
            nc_grid[i, j] = nc
            if np.isfinite(nc) and abs(nc) < abs(oracle_best["nonclosure"]):
                oracle_best.update(dict(
                    p1=p1, p2=p2, t1=t1, t2=t2,
                    A=A, B=B, C=C, D=D, A_hat=A_hat, nonclosure=nc,
                    region_statistics=statistics))

    if selection_axis1 is not None:
        selection_best = {"nonclosure": np.inf}
        for p1 in percent:
            for p2 in percent:
                t1, t2, statistics = abcd_statistics(
                    selection_axis1, selection_axis2, p1, p2,
                    weights=selection_weights)
                A, B, C, D = abcd_yields(statistics)
                valid = (
                    statistically_valid_regions(statistics, region_minimums)
                    if statistically_valid_closure
                    else A >= min_A and D >= min_D)
                if not valid:
                    continue
                nc, A_hat = nonclosure_A(A, B, C, D)
                if np.isfinite(nc) and abs(nc) < abs(
                        selection_best["nonclosure"]):
                    selection_best.update(dict(
                        p1=p1, p2=p2, t1=t1, t2=t2,
                        A=A, B=B, C=C, D=D, A_hat=A_hat,
                        nonclosure=nc, region_statistics=statistics))
        if "t1" not in selection_best:
            raise RuntimeError(
                "No validation ABCD working point found. Try lowering the "
                "minimum ABCD region counts.")
        t1_opt, t2_opt = selection_best["t1"], selection_best["t2"]
        report_statistics = abcd_region_statistics_at_thresholds(
            axis1_qcd, axis2_qcd, t1_opt, t2_opt,
            weights=gen_weights_qcd)
        A, B, C, D = abcd_yields(report_statistics)
        report_nc, report_A_hat = nonclosure_A(A, B, C, D)
        best = dict(selection_best)
        best.update({
            "selection_nonclosure": float(selection_best["nonclosure"]),
            "A": A, "B": B, "C": C, "D": D,
            "A_hat": report_A_hat,
            "nonclosure": report_nc,
            "region_statistics": report_statistics,
            "selection_source": "saved_training_validation_split",
        })
        report_weights = (
            np.ones(len(axis1_qcd), dtype=np.float64)
            if gen_weights_qcd is None else gen_weights_qcd)
        report_p1 = float(report_weights[axis1_qcd <= t1_opt].sum()
                          / report_weights.sum())
        report_p2 = float(report_weights[axis2_qcd <= t2_opt].sum()
                          / report_weights.sum())
    else:
        best = oracle_best
        best["selection_source"] = "legacy_same_sample_oracle"
        t1_opt, t2_opt = best.get("t1"), best.get("t2")
        report_p1, report_p2 = best.get("p1"), best.get("p2")

    if "t1" not in best:
        raise RuntimeError(
            "No ABCD working point found. Try lowering the minimum ABCD "
            "region counts.")

    working_point_statistically_valid = statistically_valid_regions(
        best["region_statistics"], region_minimums)

    print(
        f"Threshold source: {best['selection_source']} "
        f"(selection p1={best['p1']:.3f}, p2={best['p2']:.3f})", flush=True)
    print(f"Thresholds: t1={t1_opt:.4g}, t2={t2_opt:.4g}", flush=True)
    print(f"Independent-report nonclosure: {100.0*best['nonclosure']:.2f}%", flush=True)
    if statistically_valid_closure and not working_point_statistically_valid:
        print(
            "WARNING: the independently reported fixed working point does not "
            "meet the minimum effective statistics in every ABCD region.",
            flush=True)

    wandb.log({
        "ABCD/opt_p1":     best["p1"],
        "ABCD/opt_p2":     best["p2"],
        "ABCD/opt_t1":     float(t1_opt),
        "ABCD/opt_t2":     float(t2_opt),
        "ABCD/nonclosure": float(best["nonclosure"]),
        "ABCD/A": int(best["A"]), "ABCD/B": int(best["B"]),
        "ABCD/C": int(best["C"]), "ABCD/D": int(best["D"]),
        "ABCD/A_neff": best["region_statistics"]["A"]["effective_count"],
        "ABCD/B_neff": best["region_statistics"]["B"]["effective_count"],
        "ABCD/C_neff": best["region_statistics"]["C"]["effective_count"],
        "ABCD/D_neff": best["region_statistics"]["D"]["effective_count"],
        "ABCD/statistically_valid": int(working_point_statistically_valid),
    })

    # ── Plots ─────────────────────────────────────────────────────────────────
    fs, fs_leg, fs_legend = 28, 24, 16
    fig_size = (8, 6)

    # class_names/class_colors (and sig_class_names/sig_class_colors, if a
    # signal file was given) were already resolved near the top of ABCD().

    # 2D closure scan — full (p1, p2) grid coloured by |non-closure|
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pct_abs = np.clip(np.abs(nc_grid) * 100.0, 0.0, 100.0)
    vmax = float(np.nanpercentile(pct_abs, 95)) if np.any(np.isfinite(pct_abs)) else 100.0
    mesh = ax.pcolormesh(percent, percent, pct_abs.T, cmap="viridis_r",
                         vmin=0.0, vmax=vmax, shading="auto")
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("|Non-closure| (%)", fontsize=fs_leg)
    wp_validity_label = (
        "valid statistics" if working_point_statistically_valid
        else "insufficient statistics")
    ax.scatter([report_p1], [report_p2], marker="*", s=400, color="red",
               edgecolor="black", linewidth=1.0, zorder=5,
               label=f"Fixed threshold: test p1={report_p1:.3f}, "
                     f"p2={report_p2:.3f}\n"
                     f"|test non-closure|={100.0*abs(best['nonclosure']):.2f}%\n"
                     f"{wp_validity_label}")
    ax.set_xlabel("Percentile threshold, axis 1 (AE reco loss)", fontsize=fs_leg)
    ax.set_ylabel("Percentile threshold, axis 2 (NURD contrastive MD)", fontsize=fs_leg)
    ax.set_title("ABCD closure scan (full grid)", fontsize=fs_leg)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.9)
    fig.tight_layout()
    out_scan2d = os.path.join(plot_dir, "closure_scan_2d.png")
    fig.savefig(out_scan2d, dpi=200, bbox_inches="tight")
    plt.close(fig)
    wandb.log({"Closure/scan_2d": wandb.Image(out_scan2d)})

    # helper: apply y-scale and bins depending on axis2 type
    def _hist2d_axis2(fig_or_ax, x, y, xbins, set_labels=True):
        if axis2_log_scale:
            ybins = np.geomspace(y[y > 0].min(), y.max(), 201)
        else:
            ybins = np.linspace(y.min(), y.max(), 201)
        plt.hist2d(x, y, bins=[xbins, ybins], norm=LogNorm(vmin=1), cmin=1)
        plt.xscale("log")
        if axis2_log_scale:
            plt.yscale("log")

    # 2D histogram (all bkg)
    fig = plt.figure(figsize=(6, 5))
    xbins = np.geomspace(axis1_bkg[axis1_bkg > 0].min(), axis1_bkg.max(), 201)
    _hist2d_axis2(fig, axis1_bkg, axis2_bkg, xbins)
    plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    plt.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("AE reco loss"); plt.ylabel(axis2_label)
    plt.title("AE vs NURD Contrastive (bkg only)"); plt.colorbar(label="Counts")
    out = os.path.join(plot_dir, "hist2d_bkg.png")
    plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
    wandb.log({"Hists2D/bkg": wandb.Image(out)})

    # combined scatter by class
    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() == 0:
            continue
        ax.scatter(axis1_bkg[m], axis2_bkg[m], s=0.3, alpha=0.15,
                   color=class_colors[cls], label=name, rasterized=True)
    for lbl in sig_unique_labels:
        sm = sig_labels_masked == lbl
        if sm.sum() == 0:
            continue
        ax.scatter(sig_axis1[sm], sig_axis2[sm], s=0.5, alpha=0.4,
                   color=sig_class_colors[lbl], label=sig_class_names[lbl], rasterized=True)
    ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    if axis2_log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("AE reco loss", fontsize=fs)
    ax.set_ylabel(axis2_label, fontsize=fs)
    ax.set_title("AE vs NURD Contrastive — all classes")
    ax.legend(markerscale=10, fontsize=fs_legend)
    out_combined = os.path.join(plot_dir, "hist2d_by_class_combined.png")
    fig.savefig(out_combined, dpi=200, bbox_inches="tight"); plt.close(fig)
    wandb.log({"Hists2D/by_class_combined": wandb.Image(out_combined)})

    # individual hist2d per signal class
    for lbl in sig_unique_labels:
        sm = sig_labels_masked == lbl
        if sm.sum() < 2:
            continue
        name = sig_class_names[lbl]
        x_sig, y_sig = sig_axis1[sm], sig_axis2[sm]
        fig = plt.figure(figsize=(6, 5))
        xbins_s = np.geomspace(x_sig[x_sig > 0].min(), x_sig.max(), 101)
        _hist2d_axis2(fig, x_sig, y_sig, xbins_s)
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel(axis2_label, fontsize=fs)
        plt.title(f"AE vs NURD Contrastive — {name} (signal)"); plt.colorbar(label="Counts")
        out_sig = os.path.join(plot_dir, f"hist2d_signal_{name}.png")
        plt.savefig(out_sig, dpi=200, bbox_inches="tight"); plt.close()
        wandb.log({f"Hists2D/signal_{name}": wandb.Image(out_sig)})

    # individual hist2d per class
    for cls, name in class_names.items():
        m = labels_masked == cls
        if m.sum() < 2:
            continue
        x_cls, y_cls = axis1_bkg[m], axis2_bkg[m]
        fig = plt.figure(figsize=(6, 5))
        xbins_c = np.geomspace(x_cls[x_cls > 0].min(), x_cls.max(), 101)
        _hist2d_axis2(fig, x_cls, y_cls, xbins_c)
        plt.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("AE reco loss", fontsize=fs)
        plt.ylabel(axis2_label, fontsize=fs)
        plt.title(f"AE vs NURD Contrastive — {name}"); plt.colorbar(label="Counts")
        out_cls = os.path.join(plot_dir, f"hist2d_{name}.png")
        plt.savefig(out_cls, dpi=200, bbox_inches="tight"); plt.close()
        wandb.log({f"Hists2D/{name}": wandb.Image(out_cls)})

    # PCA-MD KDE contours (MD mode only — not meaningful for logit axis).
    # KDE contours replace the old per-point scatter for all PCA plots.
    if not use_logit and not config.get("skip_pca_md_plots"):
        fig, ax = plt.subplots(figsize=fig_size)
        groups = [
            (name, class_colors[cls], axis1_bkg[labels_masked == cls], axis2_pca[labels_masked == cls])
            for cls, name in class_names.items() if (labels_masked == cls).sum() > 0
        ]
        for lbl in sig_unique_labels:
            sm = sig_labels_masked == lbl
            if sm.sum() == 0:
                continue
            groups.append((sig_class_names[lbl], sig_class_colors[lbl], sig_axis1[sm], sig_axis2_pca[sm]))
        handles = plot_kde_contours(ax, groups, log_x=True, log_y=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("AE reco loss", fontsize=fs)
        ax.set_ylabel("NURD Contrastive score (kernel PCA-MD)", fontsize=fs)
        ax.axvline(t1_opt, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(t2_opt, color="black", linestyle="--", linewidth=1.0)
        ax.set_title("AE vs kernel PCA-MD — KDE contours")
        ax.legend(handles=handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        ax.grid(alpha=0.3)
        out_pca_kde = os.path.join(plot_dir, "hist2d_pca_md_kde.png")
        fig.savefig(out_pca_kde, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"Hists2D/pca_md_kde": wandb.Image(out_pca_kde)})

    # PCA embedding KDE contours (2D kernel PCA, fit on QCD)
    if not config.get("skip_embedding_pca"):
        emb2_fit = fit_class_kernel_transform(
            latents_masked, labels_masked == qcd_label, 2,
            class_names.get(qcd_label, "QCD"), kernel=pca_kernel, gamma=pca_gamma,
            fit_sample_cap=pca_kernel_fit_samples)
        emb_2d = _kernel_whiten(emb2_fit, latents_masked)
        sig_emb_2d = (
            _kernel_whiten(emb2_fit, sig_latents_masked)
            if sig_latents_masked is not None else None)

        fig, ax = plt.subplots(figsize=fig_size)
        groups = [
            (name, class_colors[cls], emb_2d[labels_masked == cls, 0], emb_2d[labels_masked == cls, 1])
            for cls, name in class_names.items() if (labels_masked == cls).sum() > 0
        ]
        if sig_emb_2d is not None:
            for lbl in sig_unique_labels:
                sm = sig_labels_masked == lbl
                if sm.sum() == 0:
                    continue
                groups.append((sig_class_names[lbl], sig_class_colors[lbl], sig_emb_2d[sm, 0], sig_emb_2d[sm, 1]))
        handles = plot_kde_contours(ax, groups, log_x=False, log_y=False)
        ax.set_xlabel("Kernel PCA Component 1", fontsize=fs)
        ax.set_ylabel("Kernel PCA Component 2", fontsize=fs)
        ax.set_title(f"NURD latent — kernel PCA KDE contours (fit on {class_names.get(qcd_label, 'QCD')})")
        ax.legend(handles=handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_pca_embed = os.path.join(plot_dir, "pca_kde_embeddings.png")
        fig.savefig(out_pca_embed, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/kde_embeddings": wandb.Image(out_pca_embed)})

    # Corner plot: pairwise kernel-PCA components, KDE contours (MD mode only)
    if not use_logit and n_pca >= 2:
        pairs  = [(i, j) for i in range(n_pca) for j in range(i + 1, n_pca)]
        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
        for ax, (ci, cj) in zip(axes, pairs):
            groups = [
                (name, class_colors[cls], emb_pca[labels_masked == cls, ci], emb_pca[labels_masked == cls, cj])
                for cls, name in class_names.items() if (labels_masked == cls).sum() > 0
            ]
            if sig_emb_pca is not None:
                for lbl in sig_unique_labels:
                    sm = sig_labels_masked == lbl
                    if sm.sum() == 0:
                        continue
                    groups.append((sig_class_names[lbl], sig_class_colors[lbl],
                                   sig_emb_pca[sm, ci], sig_emb_pca[sm, cj]))
            handles = plot_kde_contours(ax, groups, log_x=False, log_y=False)
            ax.set_xlabel(f"PCA Component {ci + 1}", fontsize=fs)
            ax.set_ylabel(f"PCA Component {cj + 1}", fontsize=fs)
            ax.legend(handles=handles, fontsize=fs_legend)
            ax.tick_params(axis="both", labelsize=fs_leg)
        fig.suptitle(
            "Kernel PCA-MD space — pairwise components, KDE contours "
            f"(NURD latent, fit on {class_names.get(qcd_label, 'QCD')})", fontsize=fs)
        plt.tight_layout()
        out_corner = os.path.join(plot_dir, "pca_corner.png")
        fig.savefig(out_corner, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/corner": wandb.Image(out_corner)})

    # ── No-decorrelation baseline comparison (optional) ───────────────────────
    # --baseline_ckpt is a second encoder trained identically except with
    # --lambda_info 0 (no NURD critic pressure) -- purely a diagnostic to show
    # what the contrastive model's clustering looks like before decorrelation.
    # It plays no role in the ABCD closure statistic above.
    if config.get("baseline_ckpt"):
        print("Embedding baseline (no-decorrelation) checkpoint...", flush=True)
        baseline_model, _ = load_nurd_model(config["baseline_ckpt"], device)
        baseline_latents_all, _ = embed_pf(baseline_model, config["test_pt"], device)
        baseline_latents_masked = baseline_latents_all[mask]

        baseline_sig_latents_masked = None
        if config.get("signal_pt"):
            baseline_sig_latents, _ = embed_pf(
                baseline_model, config["signal_pt"], device)
            baseline_sig_latents_masked = baseline_sig_latents[sig_mask]

        del baseline_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        baseline_kpca = fit_class_kernel_transform(
            baseline_latents_masked, labels_masked == qcd_label, 2,
            class_names.get(qcd_label, "QCD"), kernel=pca_kernel, gamma=pca_gamma,
            fit_sample_cap=pca_kernel_fit_samples)
        baseline_emb_2d = _kernel_whiten(baseline_kpca, baseline_latents_masked)
        baseline_sig_emb_2d = (
            _kernel_whiten(baseline_kpca, baseline_sig_latents_masked)
            if baseline_sig_latents_masked is not None else None)

        groups = [
            (name, class_colors[cls], baseline_emb_2d[labels_masked == cls, 0],
             baseline_emb_2d[labels_masked == cls, 1])
            for cls, name in class_names.items() if (labels_masked == cls).sum() > 0
        ]
        if baseline_sig_emb_2d is not None:
            for lbl in sig_unique_labels:
                sm = sig_labels_masked == lbl
                if sm.sum() == 0:
                    continue
                groups.append((sig_class_names[lbl], sig_class_colors[lbl],
                               baseline_sig_emb_2d[sm, 0], baseline_sig_emb_2d[sm, 1]))

        fig, ax = plt.subplots(figsize=fig_size)
        handles = plot_kde_contours(ax, groups, log_x=False, log_y=False)
        ax.set_xlabel("Kernel PCA Component 1", fontsize=fs)
        ax.set_ylabel("Kernel PCA Component 2", fontsize=fs)
        ax.set_title("NURD latent, no decorrelation (contrastive-only baseline) "
                     f"— kernel PCA KDE contours (fit on {class_names.get(qcd_label, 'QCD')})")
        ax.legend(handles=handles, fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_baseline = os.path.join(plot_dir, "pca_no_decorrelation_kde.png")
        fig.savefig(out_baseline, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({"PCA/no_decorrelation_kde": wandb.Image(out_baseline)})
        print(f"No-decorrelation baseline PCA plot saved to: {out_baseline}", flush=True)

    # Profile plots
    for (x_arr, y_arr, xlabel, ylabel, title, key, logx_flag) in [
        (axis2_bkg, axis1_bkg, axis2_label, "Mean AE reco loss",
         f"⟨AE loss⟩ vs {axis2_label}", "AE_vs_contrastive", axis2_log_scale),
        (axis1_bkg, axis2_bkg, "AE reco loss", f"Mean {axis2_label}",
         f"⟨{axis2_label}⟩ vs AE loss", "contrastive_vs_AE", True),
    ]:
        fig, ax = plt.subplots(figsize=fig_size)
        profile_plot(ax, x_arr, y_arr, nbins=60, logx=logx_flag)
        ax.set_xlabel(xlabel, fontsize=fs); ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_p = os.path.join(plot_dir, f"profile_{key}.png")
        fig.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({f"Profiles/{key}": wandb.Image(out_p)})

    for (x_arr, y_arr, xlabel, ylabel, title, key, logx_flag) in [
        (axis2_bkg, axis1_bkg, axis2_label, "Mean AE reco loss",
         f"⟨AE loss⟩ vs {axis2_label} (by class)", "AE_vs_contrastive_by_class", axis2_log_scale),
        (axis1_bkg, axis2_bkg, "AE reco loss", f"Mean {axis2_label}",
         f"⟨{axis2_label}⟩ vs AE loss (by class)", "contrastive_vs_AE_by_class", True),
    ]:
        fig, ax = plt.subplots(figsize=fig_size)
        for cls, name in class_names.items():
            m = labels_masked == cls
            if m.sum() < 20:
                continue
            profile_plot(ax, x_arr[m], y_arr[m], nbins=40, logx=logx_flag, label=name)
        ax.set_xlabel(xlabel, fontsize=fs); ax.set_ylabel(ylabel, fontsize=fs)
        ax.set_title(title); ax.legend(fontsize=fs_legend)
        plt.tick_params(axis="x", labelsize=fs_leg)
        plt.tick_params(axis="y", labelsize=fs_leg)
        out_p = os.path.join(plot_dir, f"profile_{key}.png")
        fig.savefig(out_p, dpi=200, bbox_inches="tight"); plt.close(fig)
        wandb.log({f"Profiles/{key}": wandb.Image(out_p)})

    # 1D closure scan
    effs, closure_ratio, closure_unc, curve_abs_nonclosure = [], [], [], []
    curve_rejected_points = 0
    Ntot_bkg = float(gen_weights_qcd.sum()) if gen_weights_qcd is not None else float(len(axis1_qcd))

    for p in percent:
        _t1, _t2, statistics = abcd_statistics(
            axis1_qcd, axis2_qcd, p, p, weights=gen_weights_qcd)
        A, B, C, D = abcd_yields(statistics)
        if (statistically_valid_closure
                and not statistically_valid_regions(
                    statistics, region_minimums)):
            curve_rejected_points += 1
            continue
        if statistically_valid_closure:
            ratio, sigma = closure_ratio_and_uncertainty(statistics)
        else:
            A_hat = (B * C) / max(D, 1e-8)
            ratio = A_hat / max(A, 1e-8)
            invA = 0.0 if A == 0 else 1.0 / A
            invB = 0.0 if B == 0 else 1.0 / B
            invC = 0.0 if C == 0 else 1.0 / C
            invD = 0.0 if D == 0 else 1.0 / D
            rel_var = invA + invB + invC + invD
            sigma = abs(ratio) * np.sqrt(rel_var) if rel_var > 0 else 0.0
        effs.append(A / max(Ntot_bkg, 1.0))
        closure_ratio.append(ratio)
        closure_unc.append(sigma)
        curve_nc, _ = nonclosure_A(A, B, C, D)
        curve_abs_nonclosure.append(abs(curve_nc))

    effs          = np.array(effs)
    closure_ratio = np.array(closure_ratio)
    closure_unc   = np.array(closure_unc)
    order         = np.argsort(effs)
    effs          = effs[order]
    closure_ratio = closure_ratio[order]
    closure_unc   = closure_unc[order]

    eff_opt   = best["A"] / max(Ntot_bkg, 1.0)
    ratio_opt = best["A_hat"] / max(best["A"], 1e-8)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(effs, closure_ratio, c="g", label=f"AE + NURD Contrastive ({axis2_label})")
    ax.fill_between(effs, closure_ratio - closure_unc, closure_ratio + closure_unc,
                    facecolor="g", alpha=0.5, interpolate=True)
    ax.plot(effs, np.ones_like(effs),       linestyle="-",  color="black")
    ax.plot(effs, np.full_like(effs, 0.95), linestyle="--", color="black")
    ax.plot(effs, np.full_like(effs, 1.05), linestyle="--", color="black")
    fixed_marker = "o" if working_point_statistically_valid else "X"
    fixed_label = (
        "Fixed threshold" if working_point_statistically_valid
        else "Fixed threshold (insufficient statistics)")
    ax.plot([eff_opt], [ratio_opt], marker=fixed_marker, c="red",
            label=fixed_label)
    ax.set_xlabel("Selection Efficiency (bkg A/Ntot)", fontsize=fs)
    ax.set_ylabel("Predicted Bkg. / True Bkg.",        fontsize=fs)
    ax.set_ylim([0.0, 1.5]); ax.set_xscale("log")
    plt.tick_params(axis="x", labelsize=fs_leg)
    plt.tick_params(axis="y", labelsize=fs_leg)
    plt.legend(loc="lower right", fontsize=fs_legend)
    closure_path = os.path.join(plot_dir, "cut_and_count_bkg_check.png")
    plt.savefig(closure_path, dpi=200, bbox_inches="tight"); plt.close()
    wandb.log({"Closure/plot": wandb.Image(closure_path)})

    finite_grid = np.abs(nc_grid[np.isfinite(nc_grid)])
    finite_curve = np.asarray([
        value for value in curve_abs_nonclosure if np.isfinite(value)])
    diagnostics = {
        "evaluation_protocol": best["selection_source"],
        "qcd_events": int(len(axis1_qcd)),
        "weighted_qcd": bool(gen_weights_qcd is not None),
        "statistical_filter": {
            "enabled": statistically_valid_closure,
            "count_basis": (
                "effective_count" if gen_weights_qcd is not None
                else "raw_count"),
            "minimums": region_minimums,
            "uncertainty_method": (
                "weighted_sumw2" if statistically_valid_closure
                and gen_weights_qcd is not None
                else "unweighted_poisson" if statistically_valid_closure
                else "original_evaluator"),
            "grid_rejected_points": int(grid_rejected_points),
            "diagonal_curve_rejected_points": int(curve_rejected_points),
        },
        "working_point": {
            "nonclosure": float(best["nonclosure"]),
            "absolute_nonclosure": float(abs(best["nonclosure"])),
            "selection_nonclosure": float(
                best.get("selection_nonclosure", best["nonclosure"])),
            "A": float(best["A"]), "B": float(best["B"]),
            "C": float(best["C"]), "D": float(best["D"]),
            "statistically_valid": bool(working_point_statistically_valid),
            "regions": best["region_statistics"],
        },
        "heldout_grid": {
            "points": int(finite_grid.size),
            "median_absolute_nonclosure": (
                float(np.median(finite_grid)) if finite_grid.size else None),
            "p90_absolute_nonclosure": (
                float(np.percentile(finite_grid, 90)) if finite_grid.size else None),
        },
        "heldout_diagonal_curve": {
            "points": int(finite_curve.size),
            "median_absolute_nonclosure": (
                float(np.median(finite_curve)) if finite_curve.size else None),
            "p90_absolute_nonclosure": (
                float(np.percentile(finite_curve, 90)) if finite_curve.size else None),
        },
    }
    diagnostics_path = os.path.join(outdir, "diagnostics.json")
    with open(diagnostics_path, "w", encoding="utf-8") as output:
        json.dump(diagnostics, output, indent=2)
    print(f"Diagnostics saved to: {diagnostics_path}", flush=True)

    # Save thresholds JSON so make_datacard_ttbar.py can skip the scan
    thresholds_path = os.path.join(outdir, "abcd_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump({
            "t1":    float(t1_opt),
            "t2":    float(t2_opt),
            "p1":    float(best["p1"]),
            "p2":    float(best["p2"]),
            "n_pca": config.get("n_pca", None),
            "nonclosure": float(best["nonclosure"]),
            "selection_nonclosure": float(
                best.get("selection_nonclosure", best["nonclosure"])),
            "selection_source": best["selection_source"],
            "report_p1": report_p1,
            "report_p2": report_p2,
        }, f, indent=2)
    print(f"Thresholds saved to: {thresholds_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         required=True,
                        help="Path to NURD main checkpoint (checkpoint_main.pth.tar)")
    parser.add_argument("--ae_ckpt",      required=True,
                        help="Path to AE checkpoint (checkpoint_ae.pth)")
    parser.add_argument("--test_pt",      required=True,
                        help="Path to test .pt file (SM cocktail)")
    parser.add_argument("--reference_pt", default=None,
                        help="Independent sample used only to fit the MD reference. "
                             "If omitted, preserves the legacy same-sample behavior.")
    parser.add_argument("--reference_weight_path", default=None,
                        help="Optional per-event physics weights for --reference_pt.")
    parser.add_argument("--signal_pt",    default=None,
                        help="Optional signal .pt file. May bundle several signal "
                             "processes as distinct label values in one file.")
    parser.add_argument("--signal_class_names", default=None,
                        help="Comma-separated signal process names ordered by the "
                             "label index used inside --signal_pt (e.g. ttH,HHbbgg,tttt). "
                             "Defaults to SignalN for each label found.")
    parser.add_argument("--baseline_ckpt", default=None,
                        help="Optional second NURD checkpoint trained with "
                             "--lambda_info 0 (contrastive-only, no decorrelation "
                             "pressure). Its latents are embedded separately and "
                             "plotted as a kernel-PCA KDE comparison "
                             "(pca_no_decorrelation_kde.png) against --ckpt's "
                             "decorrelated latents -- it plays no role in the ABCD "
                             "closure statistic itself.")
    parser.add_argument("--pca_kernel",   default="rbf",
                        help="Kernel for sklearn.decomposition.KernelPCA "
                             "(used for MD whitening and all PCA embedding plots)")
    parser.add_argument("--pca_gamma",    type=float, default=None,
                        help="KernelPCA gamma (default: sklearn's 1/n_features heuristic)")
    parser.add_argument("--pca_kernel_fit_samples", type=int, default=5000,
                        help="Per-class subsample size used to fit each KernelPCA "
                             "(fit cost is quadratic in this; transform is applied "
                             "to the full sample and is linear)")
    parser.add_argument("--class_names",  default=None,
                        help="Comma-separated background class names ordered by "
                             "label index (matches the double_disco_hlt data "
                             "config's samples: order). Defaults to the current "
                             "11-class DY/QCD/WJets.../TT... scheme.")
    parser.add_argument("--gen_weight_path", default=None,
                        help="Path to per-event gen weights .pt (e.g. weight_test.pt). "
                             "Must have same number of entries as --test_pt. "
                             "Applied to QCD events only for weighted ABCD counts.")
    parser.add_argument("--outdir",       default="outputs_abcd")
    parser.add_argument("--min_A",        type=int, default=50)
    parser.add_argument("--min_B",        type=int, default=50)
    parser.add_argument("--min_C",        type=int, default=50)
    parser.add_argument("--min_D",        type=int, default=500)
    parser.add_argument(
        "--statistically_valid_closure", action="store_true",
        help="Mask scan/curve points that fail minimum statistics in any "
             "ABCD region. Uses raw counts for unit-weight data, effective "
             "counts for weighted data, and sumw2 uncertainties.")
    parser.add_argument("--n_pca",        type=int, default=None,
                        help="Number of PCA components for MD (default: keep all latent dims)")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_project",  default="AE vs. Contrastive ABCD",
                        help="W&B project to log to")
    parser.add_argument("--resume_run_id",  default=None,
                        help="Resume an existing W&B run (e.g. the training run from a sweep)")
    parser.add_argument("--skip_pca_md_plots",   action="store_true")
    parser.add_argument("--skip_embedding_pca",  action="store_true")
    parser.add_argument("--min_md",              action="store_true",
                        help="Use min-MD across labels 0,1,3 instead of QCD-only MD "
                             "(index meaning depends on --class_names' order)")
    parser.add_argument("--axis2_logit",         action="store_true",
                        help="Use 1-P(QCD) classifier score as axis 2 instead of Mahalanobis distance")
    parser.add_argument("--qcd_label",           type=int, default=1,
                        help="Class label index for QCD (used by logit mode)")
    args = parser.parse_args()
    ABCD(vars(args))
