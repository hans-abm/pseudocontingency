import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_cal, load_val, _parse_sim_ht


def set_style():
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42




def plot_calibration_combined(cal_dir):
    set_style()

    emp = load_cal()
    g = pd.read_csv(f"{cal_dir}/group.csv")
    ind = pd.read_csv(f"{cal_dir}/individual.csv")

    envs = ["healthy", "unhealthy", "balanced"]
    labels = ["Healthy", "Unhealthy", "Balanced"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.0))
    fig.subplots_adjust(hspace=0.5)

    for col, (env, label) in enumerate(zip(envs, labels)):

        # --------------------
        # TOP ROW: GROUP
        # --------------------
        ax_g = axes[0, col]
        emp_env = emp[emp.env == env]["ht"].values
        best_row = g[g.env == env].sort_values("ks").iloc[0]
        sim_env = _parse_sim_ht(best_row["sim_ht"])

        sd_emp = emp_env.std()
        sd_sim = sim_env.std()

        ax_g.text(0.05, 0.92, f"SD={sd_emp:.2f}", transform=ax_g.transAxes,
                  color="#1f77b4", fontsize=10)
        ax_g.text(0.05, 0.83, f"SD={sd_sim:.2f}", transform=ax_g.transAxes,
                  color="#d62728", fontsize=10)


        ax_g.hist(emp_env, bins=5, alpha=0.6, color="#1f77b4", label="Empirical")
        ax_g.hist(sim_env, bins=5, alpha=0.6, color="#d62728", label="Simulated")

        ax_g.axvline(emp_env.mean(), color="#1f77b4", linestyle="--")
        ax_g.axvline(sim_env.mean(), color="#d62728", linestyle="--")

        beta_val = best_row["bias"]

        ax_g.set_title(
            rf"{label} ($\beta = {beta_val:.2f}$)",
            fontsize=12
        )

        ax_g.set_xlabel("Health-taste belief")
        ax_g.set_ylabel("Count")
        ax_g.set_xticks([0, 0.5, 1])


        # --------------------
        # BOTTOM ROW: INDIVIDUAL
        # --------------------
        ax_i = axes[1, col]
        emp_env = emp[emp.env == env]["ht"].values
        sim_env_ind = ind[ind.env == env]["sim_ht"].values.astype(float)

        sd_emp = emp_env.std()
        sd_sim = sim_env_ind.std()

        ax_i.text(0.05, 0.92, f"SD={sd_emp:.2f}", transform=ax_i.transAxes,
                  color="#1f77b4", fontsize=10)
        ax_i.text(0.05, 0.83, f"SD={sd_sim:.2f}", transform=ax_i.transAxes,
                  color="#d62728", fontsize=10)


        ax_i.hist(emp_env, bins=5, alpha=0.6, color="#1f77b4")
        ax_i.hist(sim_env_ind, bins=5, alpha=0.6, color="#d62728")

        ax_i.axvline(emp_env.mean(), color="#1f77b4", linestyle="--")
        ax_i.axvline(sim_env_ind.mean(), color="#d62728", linestyle="--")

        med_beta = ind[ind.env == env]["bias"].median()

        ax_i.set_title(
            rf"{label} ($\beta_i = {med_beta:.2f}$)",
            fontsize=12
        )

        ax_i.set_xlabel("Health-taste belief")
        ax_i.set_ylabel("Count")
        ax_i.set_xticks([0, 0.5, 1])


    # row labels a / b
    axes[0, 0].text(
        -0.25, 1.05, "a", transform=axes[0, 0].transAxes,
        fontsize=14, fontweight="bold"
    )
    axes[1, 0].text(
        -0.25, 1.05, "b", transform=axes[1, 0].transAxes,
        fontsize=14, fontweight="bold"
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    
    # remove y-labels from all but first column
    for r in range(2):
        for c in range(1, 3):
            axes[r, c].set_ylabel("")
    
    # remove x-labels from top row
    for c in range(3):
        axes[0, c].set_xlabel("")
    
    # keep x-labels only in bottom row
    for c in range(3):
        axes[1, c].set_xlabel("Health-taste belief")

    
    os.makedirs(cal_dir, exist_ok=True)
    png_path = f"{cal_dir}/calibration_combined.png"
    pdf_path = f"{cal_dir}/calibration_combined.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()


# ------------------------------------------------------------
# Combined VALIDATION: group + individual
# ------------------------------------------------------------

def plot_validation_combined(val_dir):
    set_style()

    emp = load_val()
    g = pd.read_csv(f"{val_dir}/group_val.csv")
    ind = pd.read_csv(f"{val_dir}/individual_val.csv")

    # unified naming – no _b
    envs = ["healthy", "unhealthy", "balanced"]
    labels = ["Healthy", "Unhealthy", "Balanced"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.0))
    fig.subplots_adjust(hspace=0.5)

    for col, (env, label) in enumerate(zip(envs, labels)):

        # --------------------
        # TOP ROW: GROUP
        # --------------------
        ax_g = axes[0, col]

        emp_env = emp[emp.env == env]["ht"].values
        row = g[g.env == env].iloc[0]   # now matches exactly
        sim_env = _parse_sim_ht(row["sim_ht"])

        sd_emp = emp_env.std()
        sd_sim = sim_env.std()

        ax_g.text(0.05, 0.92, f"SD={sd_emp:.2f}", transform=ax_g.transAxes,
                  color="#1f77b4", fontsize=10)
        ax_g.text(0.05, 0.83, f"SD={sd_sim:.2f}", transform=ax_g.transAxes,
                  color="#d62728", fontsize=10)

        ax_g.hist(emp_env, bins=5, alpha=0.6, color="#1f77b4", label="Empirical")
        ax_g.hist(sim_env, bins=5, alpha=0.6, color="#d62728", label="Simulated")

        ax_g.axvline(emp_env.mean(), color="#1f77b4", linestyle="--")
        ax_g.axvline(sim_env.mean(), color="#d62728", linestyle="--")

        beta_val = row["bias"] if "bias" in row else np.nan

        ax_g.set_title(
            rf"{label} ($\beta = {beta_val:.2f}$)",
            fontsize=12
        )

        ax_g.set_xlabel("Health-taste belief")
        ax_g.set_ylabel("Count")
        ax_g.set_xticks([0, 0.5, 1])

        # --------------------
        # BOTTOM ROW: INDIVIDUAL
        # --------------------
        ax_i = axes[1, col]

        emp_env = emp[emp.env == env]["ht"].values
        sim_env_ind = ind[ind.env == env]["sim_ht"].astype(float).values

        sd_emp = emp_env.std()
        sd_sim = sim_env_ind.std()

        ax_i.text(0.05, 0.92, f"SD={sd_emp:.2f}", transform=ax_i.transAxes,
                  color="#1f77b4", fontsize=10)
        ax_i.text(0.05, 0.83, f"SD={sd_sim:.2f}", transform=ax_i.transAxes,
                  color="#d62728", fontsize=10)

        ax_i.hist(emp_env, bins=5, alpha=0.6, color="#1f77b4")
        ax_i.hist(sim_env_ind, bins=5, alpha=0.6, color="#d62728")

        ax_i.axvline(emp_env.mean(), color="#1f77b4", linestyle="--")
        ax_i.axvline(sim_env_ind.mean(), color="#d62728", linestyle="--")

        med_beta = ind[ind.env == env]["bias"].median()

        ax_i.set_title(
            rf"{label} ($\beta_i = {med_beta:.2f}$)",
            fontsize=12
        )

        ax_i.set_xlabel("Health-taste belief")
        ax_i.set_ylabel("Count")
        ax_i.set_xticks([0, 0.5, 1])

    # Row labels
    axes[0, 0].text(-0.25, 1.05, "a", transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight="bold")
    axes[1, 0].text(-0.25, 1.05, "b", transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight="bold")

    # Remove y-labels except first column
    for r in range(2):
        for c in range(1, 3):
            axes[r, c].set_ylabel("")

    # Remove top x-labels
    for c in range(3):
        axes[0, c].set_xlabel("")

    # Keep bottom x-labels
    for c in range(3):
        axes[1, c].set_xlabel("Health-taste belief")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

    # Save files
    os.makedirs(val_dir, exist_ok=True)
    png_path = f"{val_dir}/validation_combined.png"
    pdf_path = f"{val_dir}/validation_combined.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Ridge plot
# ---------------------------------------------------------------------------

def plot_bias_ridge(cal_dir):
    set_style()

    df = pd.read_csv(f"{cal_dir}/individual.csv").copy()

    order = ["healthy", "unhealthy", "balanced"]
    df = df[df.env.isin(order)]
    df["env"] = pd.Categorical(df.env, categories=order, ordered=True)
    df = df.sort_values("env")

    palette = {
        "healthy": "#1f77b4",
        "unhealthy": "#d62728",
        "balanced": "#2ca02c"
    }

    g = sns.FacetGrid(
        df, row="env", hue="env",
        aspect=2.0, height=1.5,
        palette=palette
    )

    g.map(
        sns.kdeplot, "bias",
        bw_adjust=0.5, clip_on=False,
        fill=True, alpha=1, linewidth=1.5
    )

    g.map(
        sns.kdeplot, "bias",
        clip_on=False, color="w", lw=2, bw_adjust=0.5
    )

    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.15, 0.2, label.title(), fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=12)

    g.map(label, "bias")

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    g.figure.subplots_adjust(hspace=-0.4)
    g.figure.set_size_inches(7, 3.5)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.set(xlim=(-0.25, 1.25))
    g.axes[-1, 0].set_xlabel("Bias strength", fontsize=12)

    os.makedirs(cal_dir, exist_ok=True)
    png_path = f"{cal_dir}/bias_ridge.png"
    pdf_path = f"{cal_dir}/bias_ridge.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def run():
    cal_dir = "results/cal_251125_163653"
    val_dir = "results/val_251126_092624"

    plot_bias_ridge(cal_dir)
    plot_calibration_combined(cal_dir)
    plot_validation_combined(val_dir)

if __name__ == "__main__":
    run()