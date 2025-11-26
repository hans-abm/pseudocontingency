import os
from time import strftime
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from model import LearningModel
from utils import load_cal, ks


# ------------------------------------------------------------
# Helper: run model and return final ht beliefs
# ------------------------------------------------------------

def _run_final_ht(env, bias, lr, steps=24, n_agents=52):
    m = LearningModel(
        N=n_agents,
        width=steps,
        height=n_agents,
        env_type=env,
        bias_strength=bias,
        learning_rate=lr,
        noise = 0,
    )
    for _ in range(steps):
        m.step()
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    return df[df.Step == df.Step.max()]["ht"].values.astype(float)


# ------------------------------------------------------------
# GROUP-LEVEL: best bias per learning rate
# ------------------------------------------------------------

def _group_best_bias(lr, bias_range, emp):
    best = {env: {"bias": None, "ks": np.inf} for env in emp.keys()}

    for b in bias_range:
        for env, emp_vals in emp.items():
            sim_vals = _run_final_ht(env, b, lr)
            ks_val = ks(sim_vals, emp_vals)

            if ks_val < best[env]["ks"]:
                best[env]["ks"] = ks_val
                best[env]["bias"] = b

    return lr, best


# ------------------------------------------------------------
# INDIVIDUAL-LEVEL: best bias per participant per learning rate
# ------------------------------------------------------------

def _ind_worker(args):
    row, lr, bias_range, num_runs = args
    tgt = row.ht
    env = row.env

    n = len(bias_range)
    all_runs = []

    for _ in range(num_runs):
        m = LearningModel(
            N=n,
            width=24,
            height=n,
            env_type=env,
            bias_strength=bias_range[0],
            learning_rate=lr,
            noise=0
        )

        # assign each agent its bias
        for i, a in enumerate(m.schedule.agents):
            a.bias_strength = bias_range[i]

        for _ in range(24):
            m.step()

        df = m.datacollector.get_agent_vars_dataframe().reset_index()
        final_ht = df[df.Step == df.Step.max()]["ht"].values.astype(float)

        all_runs.append(final_ht)

    # shape: (num_runs, n_biases)
    mat = np.vstack(all_runs)
    avg_ht = mat.mean(axis=0)

    # compute error to target
    err = np.abs(avg_ht - tgt)
    i = int(err.argmin())

    return {
        "id": row.id,
        "env": env,
        "lr": lr,
        "bias": float(bias_range[i])
    }


def _individual_best_bias(lr, bias_range, df, nproc, num_runs):
    args = [(row, lr, bias_range, num_runs) for _, row in df.iterrows()]


    if nproc <= 1:
        out = [_ind_worker(a) for a in args]
    else:
        with Pool(nproc) as p:
            out = p.map(_ind_worker, args)

    return pd.DataFrame(out)


# ------------------------------------------------------------
# Compute stability metrics
# ------------------------------------------------------------

def _compute_stability(group_all, ind_all):
    out = []
    envs = ["healthy", "unhealthy", "balanced"]

    # Learning rates present in the data
    lrs = sorted(ind_all.lr.unique())

    for env in envs:
        # ----- Group-level stability -----
        g_vals = [d[env]["bias"] for _, d in group_all]
        g_mean, g_sd = np.mean(g_vals), np.std(g_vals)
        g_cv = g_sd / g_mean if g_mean != 0 else np.nan

        # ----- Individual-level stability -----
        lr_means = []
        for lr in lrs:
            subset = ind_all[(ind_all.env == env) & (ind_all.lr == lr)]
            lr_mean = subset["bias"].mean()
            lr_means.append(lr_mean)


        i_mean = np.mean(lr_means)
        i_sd = np.std(lr_means)
        i_cv = i_sd / i_mean if i_mean != 0 else np.nan

        out.append({
            "condition": env,
            "group_mean": g_mean,
            "group_sd": g_sd,
            "group_cv": g_cv,
            "ind_mean": i_mean,
            "ind_sd": i_sd,
            "ind_cv": i_cv,
        })

    return pd.DataFrame(out)

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def run(
    learning_rates=None,
    bias_range=None,
    nproc=None,
    num_runs = 10,
):
    if learning_rates is None:
        learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]

    if bias_range is None:
        bias_range = np.round(np.linspace(0, 1, 21), 2)

    if nproc is None:
        nproc = cpu_count()

    # empirical data from calibrate
    df = load_cal()

    emp = {
        "healthy": df[df.env == "healthy"]["ht"].values,
        "unhealthy": df[df.env == "unhealthy"]["ht"].values,
        "balanced": df[df.env == "balanced"]["ht"].values,
    }

    # --------------------------------------------------------
    # GROUP LEVEL
    # --------------------------------------------------------
    group_all = []
    for lr in learning_rates:
        lr_val, best = _group_best_bias(lr, bias_range, emp)
        group_all.append((lr_val, best))

    # --------------------------------------------------------
    # INDIVIDUAL LEVEL
    # --------------------------------------------------------
    ind_all_list = []
    for lr in learning_rates:
        res_lr = _individual_best_bias(lr, bias_range, df, nproc, num_runs)
        ind_all_list.append(res_lr)
    ind_all = pd.concat(ind_all_list, ignore_index=True)

    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------
    stab = _compute_stability(group_all, ind_all)

    # save
    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"results/sens_{ts}"
    os.makedirs(outdir, exist_ok=True)

    stab.to_csv(f"{outdir}/sensitivity.csv", index=False)

    return stab


if __name__ == "__main__":
    run()