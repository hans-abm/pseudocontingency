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


def _mean_ht_series(env, bias, lr, steps=24, n_agents=52, noise=0.0, seed=None):
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
    return df.groupby("Step")["ht"].mean().values.astype(float)


def run(
    envs=None,
    learning_rates=None,
    bias=1,
    steps=24,
    n_agents=1000,
    reps=50,
    noise=0.0,
    seed=None,
    outdir=None,
    plot=True,
):
    if envs is None:
        envs = ["healthy", "unhealthy", "balanced"]
    if learning_rates is None:
        learning_rates = [0.2, 0.4, 0.6, 0.8, 1.0]

    rng = np.random.default_rng(seed)
    records = []

    for env in envs:
        for lr in learning_rates:
            series_runs = []
            for _ in range(reps):
                run_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
                series = _mean_ht_series(
                    env=env,
                    bias=bias,
                    lr=lr,
                    steps=steps,
                    n_agents=n_agents,
                    noise=noise,
                    seed=run_seed,
                )
                series_runs.append(series)

            mat = np.vstack(series_runs)
            mean_series = mat.mean(axis=0)
            for step, val in enumerate(mean_series):
                records.append({
                    "env": env,
                    "learning_rate": float(lr),
                    "bias": float(bias),
                    "step": int(step),
                    "mean_ht": float(val),
                })

    df = pd.DataFrame(records)

    if outdir is None:
        ts = strftime("%y%m%d_%H%M%S")
        outdir = f"repo/results/convergence_{ts}"
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/convergence_mean.csv", index=False)

    if plot:
        plot_convergence(df, outdir)

    return df


def plot_convergence(df, outdir):
    _set_style()
    envs = ["healthy", "unhealthy", "balanced"]
    labels = ["Healthy", "Unhealthy", "Balanced"]
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    # Leave room on the right for a figure-level legend.
    fig.subplots_adjust(wspace=0.3, right=0.83)

    for ax, env, label in zip(axes, envs, labels):
        sub = df[df.env == env]
        lrs = sorted(sub.learning_rate.unique())
        colors = {lr: cmap(0.15 + 0.7 * (i / max(len(lrs) - 1, 1))) for i, lr in enumerate(lrs)}

        for lr, group in sub.groupby("learning_rate"):
            ax.plot(
                group["step"],
                group["mean_ht"],
                lw=1.8,
                color=colors[lr],
                label=fr"$\alpha$={lr}",
            )

        ax.set_title(label)
        ax.set_xlabel("Steps")
        ax.set_xlim(0, int(sub["step"].max()))
        ax.set_xticks([0, 8, 16, 24])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean belief")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        fontsize=9,
        title="Learning rate",
    )

    os.makedirs(outdir, exist_ok=True)
    png_path = f"{outdir}/convergence.png"
    pdf_path = f"{outdir}/convergence.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run()
