import os
from time import strftime
import pandas as pd
import numpy as np

from utils import load_val, mae, ks
from runner import run_reps, run_multibias, final_ht


def prep():
    df = load_val().copy()
    df["env_val"] = df["env"].map({
        "healthy": "healthy_b",
        "unhealthy": "unhealthy_b",
        "balanced": "balanced"
    })
    return df


def get_best_bias(group_df, metric="ks"):
    best = (
        group_df.sort_values(["env", metric])
        .groupby("env")
        .first()
        .reset_index()
    )
    return dict(zip(best.env, best.bias))


def fit_group(df, best_bias, lr=0.9, steps=24, reps=50):
    out = []

    for env in ["healthy_b", "unhealthy_b", "balanced"]:
        tgt = df[df.env_val == env]
        n = len(tgt)
        emp_vals = tgt["ht"].astype(float).values  # empirical target

        # simulate
        sims = run_reps(env, best_bias[env.replace("_b","")], lr, n, steps, reps)
        sims = np.vstack(sims)

        # flatten simulations
        sim_vals = sims.flatten().astype(float)

        # compute metrics
        ks_val = ks(sim_vals, emp_vals)
        mae_val = mae(sim_vals, emp_vals.mean())

        out.append({
            "env": env.replace("_b",""),
            "sim_ht": sims,
            "bias": best_bias[env.replace("_b","")],
            "n_agents": int(n),
            "ks": ks_val,
            "mae": mae_val
        })

    return pd.DataFrame(out)



def fit_ind(df, ind_df, lr=0.9, steps=24, reps=1):
    rows = []

    for env in ["healthy_b", "unhealthy_b", "balanced"]:
        sub = df[df.env_val == env]
        if len(sub) == 0:
            continue

        base_env = env.replace("_b", "")

        calib_biases = ind_df[ind_df.env == base_env]["bias"].values
        n_val = len(sub)

        # sample biases for validation participants from calibrated distribution
        sampled_biases = np.random.choice(calib_biases, size=n_val, replace=True)

        all_runs = []
        for _ in range(reps):
            sim = run_multibias(env, sampled_biases, lr, steps)
            hts = final_ht(sim)
            all_runs.append(hts)

        avg_hts = np.mean(np.vstack(all_runs), axis=0)

        
        for (idx, row), ht, b in zip(sub.iterrows(), avg_hts, sampled_biases):
            rows.append({
                "id": row.id,                 
                "env": row.env,               
                "sim_ht": float(ht),
                "bias": float(b),             
            })

    return pd.DataFrame(rows)


def run(
    lr=0.9,
    steps=24,
    reps=1,
    metric="ks",
    cal_dir=None,
):
    if cal_dir is None:
        raise ValueError("Provide calibration directory, e.g. results/cal_240421_120100")

    df = prep()
    group_df = pd.read_csv(f"{cal_dir}/group.csv")
    ind_df = pd.read_csv(f"{cal_dir}/individual.csv")

    best_bias = get_best_bias(group_df, metric)

    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"repo/results/val_{ts}"
    os.makedirs(outdir, exist_ok=True)

    g = fit_group(df, best_bias, lr, steps, reps)
    i = fit_ind(df, ind_df, lr, steps)

    g.to_csv(f"{outdir}/group_val.csv", index=False)
    i.to_csv(f"{outdir}/individual_val.csv", index=False)

    return g, i


if __name__ == "__main__":
    run(cal_dir="repo/results/cal_251125_163653")