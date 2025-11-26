import os
from time import strftime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from calibrate import fit_group, fit_ind


def sample_boot(df, seed):
    return df.sample(frac=1.0, replace=True, random_state=seed).reset_index(drop=True)


def best_bias(g, metric="ks"):
    tmp = (
        g.sort_values(["env", metric])
        .groupby("env")
        .first()
        .reset_index()
    )
    return dict(zip(tmp.env, tmp.bias))


def _worker(args):
    b_ix, df, bias_range, lr, steps, n_agents = args

    boot = sample_boot(df, seed=b_ix)

    g = fit_group(
        boot,
        lr=lr,
        steps=steps,
        reps=1,
        bias_range=bias_range,
        n_agents=n_agents,
        nproc=None,
    )
    bb = best_bias(g, metric="ks")

    out = {
        "bootstrap": b_ix,
        "group_healthy": bb.get("healthy", np.nan),
        "group_unhealthy": bb.get("unhealthy", np.nan),
        "group_balanced": bb.get("balanced", np.nan),
    }

    i = fit_ind(
        boot,
        lr=lr,
        steps=steps,
        reps=1,
        bias_range=bias_range,
        nproc=None,
    )

    all_bias = i["bias"].values
    if len(all_bias):
        out["individual_overall_mean"] = float(np.mean(all_bias))
        out["individual_overall_std"] = float(np.std(all_bias))
    else:
        out["individual_overall_mean"] = np.nan
        out["individual_overall_std"] = np.nan

    for env in ["healthy", "unhealthy", "balanced"]:
        vals = i.loc[i["env"] == env, "bias"].values
        key_mean = f"individual_{env}_mean"
        key_std = f"individual_{env}_std"
        if len(vals):
            out[key_mean] = float(np.mean(vals))
            out[key_std] = float(np.std(vals))
        else:
            out[key_mean] = np.nan
            out[key_std] = np.nan

    return out


def summarize(boot_df):
    rows = []

    for env in ["healthy", "unhealthy", "balanced"]:
        col = f"group_{env}"
        if col in boot_df.columns:
            vals = boot_df[col].dropna()
            if len(vals):
                rows.append({
                    "parameter": f"Group {env.capitalize()}",
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "ci_2.5": vals.quantile(0.025),
                    "ci_97.5": vals.quantile(0.975),
                    "n_bootstrap": len(vals),
                })

    for env in ["healthy", "unhealthy", "balanced"]:
        col = f"individual_{env}_mean"
        if col in boot_df.columns:
            vals = boot_df[col].dropna()
            if len(vals):
                rows.append({
                    "parameter": f"Individual {env.capitalize()} Mean",
                    "median": vals.median(),
                    "mean": vals.mean(),
                    "std": vals.std(),
                    "ci_2.5": vals.quantile(0.025),
                    "ci_97.5": vals.quantile(0.975),
                    "n_bootstrap": len(vals),
                })



    return pd.DataFrame(rows)


def run(
    n_bootstrap=1000,
    bias_step=0.1,
    lr=0.9,
    steps=24,
    n_agents=52,
    nproc=None,
    data_path="data/calibrate.csv",
):
    
   
    df = pd.read_csv(data_path)

    n_vals = int(1.0 / bias_step) + 1
    bias_range = np.round(np.linspace(0, 1, n_vals), 2)

    if nproc is None:
        nproc = cpu_count()

    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"results/boot_{ts}"
    os.makedirs(outdir, exist_ok=True)

    args = [(i, df, bias_range, lr, steps, n_agents) for i in range(n_bootstrap)]

    if nproc <= 1:
        rows = [_worker(a) for a in args]
    else:
        with Pool(processes=nproc) as pool:
            rows = pool.map(_worker, args)

    boot_df = pd.DataFrame(rows)
    summary = summarize(boot_df)
    summary.to_csv(f"{outdir}/bootstrap.csv", index=False)

    return summary

if __name__ == "__main__":
    run()