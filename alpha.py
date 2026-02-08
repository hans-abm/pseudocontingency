import os
from time import strftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import LearningModel


def _set_style():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["savefig.dpi"] = 600
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _final_ht(env, bias, lr, steps=24, n_agents=52, noise=0.0, seed=None):
    m = LearningModel(
        N=n_agents,
        width=steps,
        height=n_agents,
        env_type=env,
        bias_strength=bias,
        learning_rate=lr,
        noise=noise,
        seed=seed,
    )
    for _ in range(steps):
        m.step()
    df = m.datacollector.get_agent_vars_dataframe().reset_index()
    return df[df.Step == df.Step.max()]["ht"].values.astype(float)


def _belief_surface(envs, learning_rates, bias_range, steps, n_agents, reps, noise, seed):
    rows = []
    rng = np.random.default_rng(seed)

    for env in envs:
        for lr in learning_rates:
            for bias in bias_range:
                all_runs = []
                for _ in range(reps):
                    run_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
                    final_ht = _final_ht(
                        env=env,
                        bias=bias,
                        lr=lr,
                        steps=steps,
                        n_agents=n_agents,
                        noise=noise,
                        seed=run_seed,
                    )
                    all_runs.append(final_ht)

                mat = np.vstack(all_runs)
                rows.append({
                    "env": env,
                    "learning_rate": float(lr),
                    "bias": float(bias),
                    "mean_ht": float(mat.mean()),
                    "std_ht": float(mat.std()),
                })

    return pd.DataFrame(rows)


def run(
    envs=None,
    learning_rates=None,
    bias_range=None,
    steps=24,
    n_agents=52,
    reps=10,
    noise=0.0,
    seed=None,
    outdir=None,
    plot=True,
):
    if envs is None:
        envs = ["healthy", "unhealthy", "balanced"]
    if learning_rates is None:
        learning_rates = np.round(np.linspace(0, 1, 11), 2)
    if bias_range is None:
        bias_range = np.round(np.linspace(0, 1, 11), 2)

    df = _belief_surface(
        envs=envs,
        learning_rates=learning_rates,
        bias_range=bias_range,
        steps=steps,
        n_agents=n_agents,
        reps=reps,
        noise=noise,
        seed=seed,
    )

    if outdir is None:
        ts = strftime("%y%m%d_%H%M%S")
        outdir = f"repo/results/alpha_{ts}"
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/belief_surface.csv", index=False)

    if plot:
        plot_belief_surface(df, outdir)

    return df


def plot_belief_surface(df, outdir):
    _set_style()
    envs = ["healthy", "unhealthy", "balanced"]
    labels = ["Healthy", "Unhealthy", "Balanced"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    vmin = df["mean_ht"].min()
    vmax = df["mean_ht"].max()
    levels = np.linspace(vmin, vmax, 150)

    mappable = None

    for i, (ax, env, label) in enumerate(zip(axes, envs, labels)):
        sub = df[df.env == env].sort_values(["learning_rate", "bias"])

        x_unique = np.sort(sub["bias"].unique())
        y_unique = np.sort(sub["learning_rate"].unique())
        Z = sub["mean_ht"].values.reshape(len(y_unique), len(x_unique))

        cf = ax.contourf(
            x_unique,
            y_unique,
            Z,
            levels=levels,  
            cmap="viridis"
        )

        if mappable is None: 
            mappable = cf

        ax.set_title(label)
        ax.set_xlabel(r"$\beta$")
        ax.set_xlim(x_unique.min(), x_unique.max())
        ax.set_ylim(y_unique.min(), y_unique.max())
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 0.5, 1])

        if i == 0:
            ax.set_ylabel(r"$\alpha$")
        else:
            ax.set_ylabel("")


    cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.9)
    cbar.set_label("Mean belief")
    cbar.set_ticks([vmin, 0.5, vmax])
    cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    os.makedirs(outdir, exist_ok=True)
    png_path = f"{outdir}/belief_surface.png"
    pdf_path = f"{outdir}/belief_surface.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    run()