import os
from time import strftime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from utils import load_cal
from calibrate import fit_group as fit_group_base
from integration import fit_group as fit_group_weighted
from integration_full import fit_group as fit_group_full

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_boot(df, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for env, sub in df.groupby("env"):
        rs = int(rng.integers(0, 2**32 - 1))
        boot = sub.sample(n=len(sub), replace=True, random_state=rs)
        parts.append(boot)
    return pd.concat(parts, ignore_index=True)


def _best_by_metric(g, param_col, metric):
    tmp = (
        g.sort_values(["env", metric])
        .groupby("env")
        .first()
        .reset_index()
    )
    return tmp[["env", param_col, metric]]


def _best_full_by_metric(g, metric):
    tmp = (
        g.sort_values(["env", metric])
        .groupby("env")
        .first()
        .reset_index()
    )
    return tmp[["env", "w_ht", "w_hn", "w_ut", "w_un", metric]]


def _worker(args):
    (
        b_ix,
        df,
        bias_range,
        weight_range,
        full_weight_range,
        full_weight_grid,
        integration_mode,
        metric,
        lr,
        steps,
        reps,
        n_agents,
        noise,
    ) = args

    boot = sample_boot(df, seed=b_ix)

    # Base-rate model calibration
    g_base = fit_group_base(
        boot,
        lr=lr,
        steps=steps,
        reps=reps,
        bias_range=bias_range,
        n_agents=n_agents,
        nproc=None,
    )
    best_base = _best_by_metric(g_base, "bias", metric)

    best_w = None
    best_f = None

    if integration_mode in ("single", "both"):
        g_w = fit_group_weighted(
            boot,
            weight_range=weight_range,
            lr=lr,
            steps=steps,
            reps=reps,
            n_agents=n_agents,
            noise=noise,
            seed=b_ix,
        )
        best_w = _best_by_metric(g_w, "cellA_weight", metric)

    if integration_mode in ("full", "both"):
        g_f = fit_group_full(
            boot,
            weight_range=full_weight_range,
            weight_grid=full_weight_grid,
            lr=lr,
            steps=steps,
            reps=reps,
            n_agents=n_agents,
            noise=noise,
            seed=b_ix,
        )
        best_f = _best_full_by_metric(g_f, metric)

    rows = []
    for env in ["healthy", "unhealthy", "balanced"]:
        bb = best_base[best_base.env == env].iloc[0]
        row = {
            "bootstrap": b_ix,
            "env": env,
            "bias": float(bb["bias"]),
            f"{metric}_base": float(bb[metric]),
        }

        if best_w is not None:
            bw = best_w[best_w.env == env].iloc[0]
            row.update({
                "cellA_weight": float(bw["cellA_weight"]),
                f"{metric}_weighted": float(bw[metric]),
                f"delta_{metric}": float(bw[metric] - bb[metric]),
            })

        if best_f is not None:
            bf = best_f[best_f.env == env].iloc[0]
            row.update({
                "w_ht": float(bf["w_ht"]),
                "w_hn": float(bf["w_hn"]),
                "w_ut": float(bf["w_ut"]),
                "w_un": float(bf["w_un"]),
                f"{metric}_full": float(bf[metric]),
                f"delta_{metric}_full": float(bf[metric] - bb[metric]),
            })

        rows.append(row)

    return rows


def summarize(boot_df, metric="mae"):
    rows = []
    for env in ["healthy", "unhealthy", "balanced"]:
        sub = boot_df[boot_df.env == env]
        cols = [
            f"{metric}_base",
            f"{metric}_weighted",
            f"delta_{metric}",
            f"{metric}_full",
            f"delta_{metric}_full",
        ]
        for col in cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if len(vals):
                rows.append({
                    "env": env,
                    "metric": col,
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "ci_2.5": vals.quantile(0.025),
                    "ci_97.5": vals.quantile(0.975),
                    "n_bootstrap": len(vals),
                })
    return pd.DataFrame(rows)


def run(
    n_bootstrap=100,
    bias_step=0.01,
    weight_min=0.0,
    weight_max=3.0,
    weight_step=None,
    full_weight_range=None,
    full_weight_grid=None,
    integration_mode="full",
    metric="mae",
    lr=0.9,
    steps=24,
    reps=1,
    n_agents=52,
    noise=0.0,
    nproc=None,
):
    df = load_cal()

    n_vals = int(1.0 / bias_step) + 1
    bias_range = np.round(np.linspace(0, 1, n_vals), 2)

    if weight_step is None:
        weight_step = (weight_max - weight_min) / (n_vals - 1)
    n_w = int(round((weight_max - weight_min) / weight_step)) + 1
    weight_range = np.round(np.linspace(weight_min, weight_max, n_w), 2)

    if nproc is None:
        nproc = cpu_count()

    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"repo/results/boot_compare_{ts}"
    os.makedirs(outdir, exist_ok=True)

    if integration_mode not in ("single", "full", "both"):
        raise ValueError("integration_mode must be 'single', 'full', or 'both'")

    args = [
        (
            i,
            df,
            bias_range,
            weight_range,
            full_weight_range,
            full_weight_grid,
            integration_mode,
            metric,
            lr,
            steps,
            reps,
            n_agents,
            noise,
        )
        for i in range(n_bootstrap)
    ]

    if nproc <= 1:
        all_rows = []
        for a in args:
            all_rows.extend(_worker(a))
    else:
        with Pool(processes=nproc) as pool:
            all_rows = []
            for rows in pool.map(_worker, args):
                all_rows.extend(rows)

    boot_df = pd.DataFrame(all_rows)
    boot_df.to_csv(f"{outdir}/boot_compare.csv", index=False)

    summary = summarize(boot_df, metric=metric)
    summary.to_csv(f"{outdir}/boot_compare_summary.csv", index=False)

    return boot_df, summary


def append_bootstrap(
    existing_csv,
    n_bootstrap=100,
    bias_step=0.01,
    weight_min=0.0,
    weight_max=3.0,
    weight_step=None,
    full_weight_range=None,
    full_weight_grid=None,
    integration_mode="full",
    metric="mae",
    lr=0.9,
    steps=24,
    reps=1,
    n_agents=52,
    noise=0.0,
    nproc=None,
):
    boot_df_existing = pd.read_csv(existing_csv)
    if "bootstrap" not in boot_df_existing.columns:
        raise ValueError("existing_csv must include a 'bootstrap' column")

    start_ix = int(boot_df_existing["bootstrap"].max()) + 1
    df = load_cal()

    n_vals = int(1.0 / bias_step) + 1
    bias_range = np.round(np.linspace(0, 1, n_vals), 2)

    if weight_step is None:
        weight_step = (weight_max - weight_min) / (n_vals - 1)
    n_w = int(round((weight_max - weight_min) / weight_step)) + 1
    weight_range = np.round(np.linspace(weight_min, weight_max, n_w), 2)

    if nproc is None:
        nproc = cpu_count()

    if integration_mode not in ("single", "full", "both"):
        raise ValueError("integration_mode must be 'single', 'full', or 'both'")
    if metric not in ("mae", "ks"):
        raise ValueError("metric must be 'mae' or 'ks'")

    args = [
        (
            i,
            df,
            bias_range,
            weight_range,
            full_weight_range,
            full_weight_grid,
            integration_mode,
            metric,
            lr,
            steps,
            reps,
            n_agents,
            noise,
        )
        for i in range(start_ix, start_ix + n_bootstrap)
    ]

    if nproc <= 1:
        all_rows = []
        for a in args:
            all_rows.extend(_worker(a))
    else:
        with Pool(processes=nproc) as pool:
            all_rows = []
            for rows in pool.map(_worker, args):
                all_rows.extend(rows)

    boot_df_new = pd.DataFrame(all_rows)
    boot_df = pd.concat([boot_df_existing, boot_df_new], ignore_index=True)

    outdir = os.path.dirname(existing_csv)
    boot_df.to_csv(existing_csv, index=False)

    summary = summarize(boot_df, metric=metric)
    summary.to_csv(os.path.join(outdir, "boot_compare_summary.csv"), index=False)

    return boot_df, summary


if __name__ == "__main__":
    run()