import os
from warnings import filterwarnings

filterwarnings(
    "ignore",
    category=FutureWarning
)

import calibrate
import validate
import sensitivity
import bootstrap
import plots
from utils import model_performance, model_performance_val



def _latest_result_dir(prefix):

    dirs = [d for d in os.listdir("repo/results") if d.startswith(prefix)]
    if not dirs:
        raise RuntimeError(f"No directories found with prefix {prefix}")
    dirs.sort()
    return os.path.join("repo/results", dirs[-1])


def run_all(
    lr=0.9,
    steps=24,
    reps_val=1,
    reps_cal=1,
    reps_sens=10,
    bias_range=None,
    n_bootstrap=1000,
    n_agents=52,
    nproc_group=None,
    nproc_ind=None,
    data_path="repo/data/calibrate.csv"
):

    print("\n--- Starting full reproducibility pipeline ---\n")

    # ----------------------------------------------------------------------
    # 1. CALIBRATION
    # ----------------------------------------------------------------------
    print("Step 1: Calibration")

    calibrate.run(
        lr=lr,
        bias_range=bias_range,
        steps=steps,
        reps=reps_cal,
        n_agents=n_agents,
        nproc_group=nproc_group,
        nproc_ind=nproc_ind,
    )

    cal_dir = _latest_result_dir("cal_")
    model_performance(cal_dir)


    print(f"Calibration saved to: {cal_dir}")


    # ----------------------------------------------------------------------
    # 2. VALIDATION
    # ----------------------------------------------------------------------
    print("\nStep 2: Validation")

    validate.run(
        lr=lr,
        steps=steps,
        reps=reps_val,
        metric="ks",
        cal_dir=cal_dir
    )

    val_dir = _latest_result_dir("val_")
    model_performance_val(val_dir)

    print(f"Validation saved to: {val_dir}")


    # ----------------------------------------------------------------------
    # 3. PLOTS
    # ----------------------------------------------------------------------
    print("\nStep 3: Generating plots")

    plots.plot_bias_ridge(cal_dir)
    plots.plot_calibration_combined(cal_dir)
    plots.plot_validation_combined(val_dir)

    print("Plots saved to:")
    print(f"  {cal_dir}")
    print(f"  {val_dir}")


    # ----------------------------------------------------------------------
    # 4. SENSITIVITY ANALYSIS
    # ----------------------------------------------------------------------
    print("\nStep 4: Sensitivity analysis")

    sensitivity.run(
        learning_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
        bias_range=bias_range,
        num_runs=reps_sens,
    )

    sens_dir = _latest_result_dir("sens_")
    print(f"Sensitivity analysis saved to: {sens_dir}")


    # ----------------------------------------------------------------------
    # 5. BOOTSTRAP
    # ----------------------------------------------------------------------
    print("\nStep 5: Bootstrap analysis")

    bootstrap.run(
        n_bootstrap=n_bootstrap,
        bias_step=0.1,
        lr=lr,
        steps=steps,
        n_agents=n_agents,
        nproc=None,
        data_path=data_path,
    )

    boot_dir = _latest_result_dir("boot_")
    print(f"Bootstrap summary saved to: {boot_dir}")


    print("\n--- Full pipeline complete ---")

    return {
        "calibration_dir": cal_dir,
        "validation_dir": val_dir,
        "sensitivity_dir": sens_dir,
        "bootstrap_dir": boot_dir,
    }


if __name__ == "__main__":
    run_all()