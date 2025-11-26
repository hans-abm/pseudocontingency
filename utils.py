import os
import pandas as pd
import numpy as np

DATA_DIR = "repo\data"

def load_cal():
    path = os.path.join(DATA_DIR, "calibrate.csv")
    return pd.read_csv(path)

def load_val():
    path = os.path.join(DATA_DIR, "validate.csv")
    return pd.read_csv(path)

def mae(a, b):
    return float(np.mean(np.abs(a - b)))

def ks(a, b):
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)

def boot(df):
    idx = np.random.choice(len(df), size=len(df), replace=True)
    return df.iloc[idx].reset_index(drop=True)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _parse_sim_ht(x):
    if isinstance(x, (list, np.ndarray)):
        return np.array(x).astype(float)

    x = x.replace("[", " ").replace("]", " ")
    vals = x.split()
    return np.array([float(v) for v in vals])

def model_performance(cal_dir):

    emp = load_cal()
    g = pd.read_csv(f"{cal_dir}/group.csv")
    ind = pd.read_csv(f"{cal_dir}/individual.csv")

    envs = ["healthy", "unhealthy", "balanced"]
    rows = []

    for env in envs:
        # empirical
        emp_env = emp[emp.env == env]["ht"].values.astype(float)
        sd_emp = emp_env.std()

        # -----------------------------------------------------
        # GROUP LEVEL
        # -----------------------------------------------------
        best_g = g[g.env == env].sort_values("ks").iloc[0]
        sim_group = _parse_sim_ht(best_g["sim_ht"]).astype(float)

        mae_group = mae(sim_group, emp_env.mean())
        ks_group = ks(sim_group, emp_env)

        sd_group = sim_group.std()
        sd_ratio_group = sd_emp / sd_group if sd_group > 0 else np.nan

        # -----------------------------------------------------
        # INDIVIDUAL LEVEL
        # -----------------------------------------------------
        sim_ind = ind[ind.env == env]["sim_ht"].astype(float).values
        sd_ind = sim_ind.std()

        mae_ind = mae(sim_ind, emp_env.mean())
        ks_ind = ks(sim_ind, emp_env)

        sd_ratio_ind = sd_emp / sd_ind if sd_ind > 0 else np.nan

        # -----------------------------------------------------
        # RATIO GROUP OVER INDIVIDUAL
        # -----------------------------------------------------
        ratio_over_ratio = (
            sd_ratio_group / sd_ratio_ind
            if sd_ratio_ind > 0 else np.nan
        )

        rows.append({
            "env": env,

            # MAE
            "mae_group": mae_group,
            "mae_individual": mae_ind,

            # KS
            "ks_group": ks_group,
            "ks_individual": ks_ind,

            # SDs
            "sd_empirical": sd_emp,
            "sd_group": sd_group,
            "sd_individual": sd_ind,

            # SD ratios
            "sd_ratio_group": sd_ratio_group,
            "sd_ratio_individual": sd_ratio_ind,

            "sd_ratio_group_over_individual": ratio_over_ratio
        })

    out = pd.DataFrame(rows)

    out_path = os.path.join(cal_dir, "model_performance.csv")
    out.to_csv(out_path, index=False)

    return out

def model_performance_val(val_dir):
    """
    Compute MAE, KS, SD, SD ratios for group- and individual-level validation.
    Saves results as model_performance_val.csv in <val_dir>.
    """

    emp = load_val()
    g = pd.read_csv(f"{val_dir}/group_val.csv")
    ind = pd.read_csv(f"{val_dir}/individual_val.csv")

    envs = ["healthy", "unhealthy", "balanced"]
    rows = []

    for env in envs:
        # empirical
        emp_env = emp[emp.env == env]["ht"].values.astype(float)
        sd_emp = emp_env.std()

        # -----------------------------------------------------
        # GROUP LEVEL (one row per env)
        # -----------------------------------------------------
        row_g = g[g.env == env].iloc[0]
        sim_group = _parse_sim_ht(row_g["sim_ht"]).astype(float)

        mae_group = mae(sim_group, emp_env.mean())
        ks_group = ks(sim_group, emp_env)

        sd_group = sim_group.std()
        sd_ratio_group = sd_emp / sd_group if sd_group > 0 else np.nan

        # -----------------------------------------------------
        # INDIVIDUAL LEVEL
        # -----------------------------------------------------
        sim_ind = ind[ind.env == env]["sim_ht"].astype(float).values
        sd_ind = sim_ind.std()

        mae_ind = mae(sim_ind, emp_env.mean())
        ks_ind = ks(sim_ind, emp_env)

        sd_ratio_ind = sd_emp / sd_ind if sd_ind > 0 else np.nan

        # -----------------------------------------------------
        # Ratio of ratios
        # -----------------------------------------------------
        ratio_over_ratio = (
            sd_ratio_group / sd_ratio_ind if sd_ratio_ind > 0 else np.nan
        )

        rows.append({
            "env": env,

            # MAE
            "mae_group": mae_group,
            "mae_individual": mae_ind,

            # KS
            "ks_group": ks_group,
            "ks_individual": ks_ind,

            # SD
            "sd_empirical": sd_emp,
            "sd_group": sd_group,
            "sd_individual": sd_ind,

            # SD ratios
            "sd_ratio_group": sd_ratio_group,
            "sd_ratio_individual": sd_ratio_ind,

            # ratio of ratios
            "sd_ratio_group_over_individual": ratio_over_ratio,
        })

    out = pd.DataFrame(rows)
    out_path = os.path.join(val_dir, "model_performance_val.csv")
    out.to_csv(out_path, index=False)

    return out
