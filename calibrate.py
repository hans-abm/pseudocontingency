import os
from time import strftime
import multiprocessing as mp

import numpy as np
import pandas as pd

from utils import load_cal, ks, mae
from runner import run_reps, run_multibias, final_ht


def prep():
    return load_cal().copy()


def _group_worker(args):
    bias, lr, steps, reps, n_agents, emp = args
    rows = []

    for env, emp_vals in emp.items():

        sims = run_reps(env, bias, lr, n_agents, steps, reps)
        sims_mat = np.vstack(sims)
        sims_avg = sims_mat.mean(axis=0)

        rows.append({
            "env": env,
            "bias": bias,
            "mae": mae(sims_avg, emp_vals),
            "ks": ks(sims_avg, emp_vals),
            "sim_ht": sims_mat,
            "n_agents": int(n_agents),
        })

    return rows


def fit_group(df, lr=0.9, steps=24, reps=1, bias_range=None, n_agents=52, nproc=None):
    if bias_range is None:
        bias_range = np.round(np.linspace(0, 1, 11), 2)

    emp = {
        "healthy": df[df.env == "healthy"]["ht"].values,
        "unhealthy": df[df.env == "unhealthy"]["ht"].values,
        "balanced": df[df.env == "balanced"]["ht"].values,
    }

    args = [(b, lr, steps, reps, n_agents, emp) for b in bias_range]

    rows = []
    if nproc is None or nproc <= 1:
        for a in args:
            rows.extend(_group_worker(a))
    else:
        with mp.Pool(nproc) as pool:
            for out in pool.imap_unordered(_group_worker, args):
                rows.extend(out)

    g = pd.DataFrame(rows)
    g = g.sort_values(["env", "bias"]).reset_index(drop=True)
    return g


def _ind_worker(args):
    row, lr, steps, bias_range, reps = args
    tgt = row.ht
    env = row.env

    sims_list = []
    for _ in range(reps):
        m = run_multibias(env, bias_range, lr, steps)
        sims_list.append(final_ht(m))

    sims_mat = np.vstack(sims_list)
    sims_avg = sims_mat.mean(axis=0)

    err_vec = np.abs(sims_avg - tgt)
    i = int(err_vec.argmin())

    return {
        "id": row.id,
        "env": env,
        "sim_ht": float(sims_avg[i]),
        "bias": float(bias_range[i]),
        "mae": float(err_vec[i]),
    }


def fit_ind(df, lr=0.9, steps=24, reps=1, bias_range=None, nproc=None):
    if bias_range is None:
        bias_range = np.round(np.linspace(0, 1, 11), 2)

    args = [(row, lr, steps, bias_range, reps) for _, row in df.iterrows()]

    if nproc is None or nproc <= 1:
        out = [_ind_worker(a) for a in args]
    else:
        with mp.Pool(nproc) as pool:
            out = pool.map(_ind_worker, args)

    i = pd.DataFrame(out)
    i = i.sort_values("id").reset_index(drop=True)
    return i


def run(
    lr=0.9,
    bias_range=None,
    steps=24,
    reps=1,
    n_agents=52,
    nproc_group=1,
    nproc_ind=1,
):
    df = prep()
    ts = strftime("%y%m%d_%H%M%S")
    outdir = f"repo/results/cal_{ts}"
    os.makedirs(outdir, exist_ok=True)

    g = fit_group(df, lr, steps, reps, bias_range, n_agents, nproc_group)
    i = fit_ind(df, lr, steps, reps, bias_range, nproc_ind)

    g.to_csv(f"{outdir}/group.csv", index=False)
    i.to_csv(f"{outdir}/individual.csv", index=False)

    return g, i


if __name__ == "__main__":
    run(bias_range=np.round(np.linspace(0, 1, 21), 2))