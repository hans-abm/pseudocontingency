import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from time import strftime
from scipy.stats import ks_2samp

# PNAS-compliant figure settings
def set_pnas_style():
    """Set matplotlib parameters to comply with PNAS guidelines"""
    # Use standard fonts as recommended by PNAS
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12  # Base font size
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    
    # Turn off default grid
    plt.rcParams['axes.grid'] = False
    
    # Set line widths
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 2.0
    
    # Set figure DPI to PNAS standards
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    
    # Ensure vector text is preserved in PDFs
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

# PNAS column width options (in inches)
PNAS_ONE_COL = 3.42  # 20.5 picas
PNAS_ONE_HALF_COL = 4.5  # 27 picas
PNAS_TWO_COL = 7.0  # 42.125 picas
MAX_HEIGHT = 9.0  # 54 picas


def plot_distribution_comparison(best_params, num_simulations=20, results_dir=None, run_simulation=None, emp_healthy=None, emp_unhealthy=None, emp_balanced=None):
    """
    Plot the comparison of simulated distributions with empirical data for all environments.
    
    Parameters:
    best_params (list): List of best bias strength parameters [healthy, unhealthy, balanced]
    num_simulations (int): Number of times to run the simulation
    results_dir (str): Directory to save results
    run_simulation (function): Function to run simulations
    emp_healthy, emp_unhealthy, emp_balanced: Empirical data for different conditions
    """
    # Set PNAS style
    set_pnas_style()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    if results_dir is None:
        results_dir = f'paper_1/results/results_{strftime("%y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)

    # Run simulations
    healthy_results, _, _ = run_simulation([best_params[0]], num_simulations)
    _, unhealthy_results, _ = run_simulation([best_params[1]], num_simulations)
    _, _, balanced_results = run_simulation([best_params[2]], num_simulations)

    # Average results
    sim_healthy_ht = pd.concat(healthy_results).groupby(level=0).mean()['ht']
    sim_unhealthy_ht = pd.concat(unhealthy_results).groupby(level=0).mean()['ht']
    sim_balanced_ht = pd.concat(balanced_results).groupby(level=0).mean()['ht']

    # Standard deviations
    sd_healthy_emp = emp_healthy.std()
    sd_unhealthy_emp = emp_unhealthy.std()
    sd_balanced_emp = emp_balanced.std()
    
    sd_healthy_sim = sim_healthy_ht.std()
    sd_unhealthy_sim = sim_unhealthy_ht.std()
    sd_balanced_sim = sim_balanced_ht.std()

    # Prepare data for histogram plotting
    data_sets = [
        (emp_healthy, sim_healthy_ht, 'Healthy', best_params[0], sd_healthy_emp, sd_healthy_sim),
        (emp_unhealthy, sim_unhealthy_ht, 'Unhealthy', best_params[1], sd_unhealthy_emp, sd_unhealthy_sim),
        (emp_balanced, sim_balanced_ht, 'Balanced', best_params[2], sd_balanced_emp, sd_balanced_sim)
    ]

    bins = 5
    alpha = 0.6
    colors = {'empirical': '#1f77b4', 'simulated': '#d62728'}

    max_y = 0
    for emp, sim, *_ in data_sets:
        counts_emp, _ = np.histogram(emp, bins=bins)
        counts_sim, _ = np.histogram(sim, bins=bins)
        max_y = max(max_y, counts_emp.max(), counts_sim.max())


    fig, axes = plt.subplots(1, 3, figsize=(PNAS_TWO_COL, 3.5))

    for ax, (emp_data, sim_data, title, bias_val, sd_emp, sd_sim) in zip(axes, data_sets):
        ax.hist(emp_data, bins=bins, alpha=alpha, color=colors['empirical'], label='Empirical')
        ax.hist(sim_data, bins=bins, alpha=alpha, color=colors['simulated'], label='Simulated')
        ax.axvline(emp_data.mean(), color=colors['empirical'], linestyle='--', linewidth=2.5)
        ax.axvline(sim_data.mean(), color=colors['simulated'], linestyle='--', linewidth=2.5)
        

        ax.text(0.05, 0.92, f'SD = {sd_emp:.2f}', transform=ax.transAxes, 
                color=colors['empirical'], fontsize=9, fontweight='bold')
        ax.text(0.05, 0.85, f'SD = {sd_sim:.2f}', transform=ax.transAxes, 
                color=colors['simulated'], fontsize=9, fontweight='bold')
        

        title_text = f'{title}\nBias = {bias_val:.2f}'
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel('Health-Taste Belief')
        ax.set_ylim(0, max_y * 1.1)
        #ax.set_xlim(0, 1)

    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{results_dir}/group_distribution_comparison.pdf', bbox_inches='tight', format='pdf')
    plt.close()
    
    # Create and save a table of standard deviations
    sd_table = pd.DataFrame({
        'Condition': ['Healthy', 'Unhealthy', 'Balanced'],
        'Empirical SD': [sd_healthy_emp, sd_unhealthy_emp, sd_balanced_emp],
        'Simulated SD': [sd_healthy_sim, sd_unhealthy_sim, sd_balanced_sim],
        'Difference': [sd_healthy_emp - sd_healthy_sim, 
                      sd_unhealthy_emp - sd_unhealthy_sim, 
                      sd_balanced_emp - sd_balanced_sim]
    })
    sd_table.to_csv(f'{results_dir}/standard_deviation_comparison.csv', index=False)
    
    return fig


def plot_individual_distribution_comparison(individual_df, results_dir=None, emp_healthy=None, emp_unhealthy=None, emp_balanced=None):
    """
    Plot the comparison of individually fitted model distributions with empirical data for all environments.
    
    Parameters:
    individual_df (DataFrame): DataFrame with individual-level fit results
    results_dir (str): Directory to save results
    emp_healthy, emp_unhealthy, emp_balanced: Empirical data for different conditions
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """

    set_pnas_style()
    
    if results_dir is None:
        results_dir = f'paper_1/results/results_{strftime("%y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)
    
    # Calculate standard deviations for empirical data
    sd_healthy_emp = emp_healthy.std()
    sd_unhealthy_emp = emp_unhealthy.std()
    sd_balanced_emp = emp_balanced.std()
    
    # Get simulated data from individual fits
    sim_healthy = individual_df[individual_df['condition'] == 'healthy']['simulated_ht']
    sim_unhealthy = individual_df[individual_df['condition'] == 'unhealthy']['simulated_ht']
    sim_balanced = individual_df[individual_df['condition'] == 'balanced']['simulated_ht']
    
    # Calculate standard deviations for simulated data
    sd_healthy_sim = sim_healthy.std()
    sd_unhealthy_sim = sim_unhealthy.std()
    sd_balanced_sim = sim_balanced.std()
    
    # Prepare data for histogram plotting
    data_sets = [
        (emp_healthy, sim_healthy, 'Healthy', sd_healthy_emp, sd_healthy_sim),
        (emp_unhealthy, sim_unhealthy, 'Unhealthy', sd_unhealthy_emp, sd_unhealthy_sim),
        (emp_balanced, sim_balanced, 'Balanced', sd_balanced_emp, sd_balanced_sim)
    ]
    
    bins = 5
    alpha = 0.6
    colors = {'empirical': '#1f77b4', 'simulated': '#d62728'}
    
    max_y = 0
    for emp, sim, *_ in data_sets:
        counts_emp, _ = np.histogram(emp, bins=bins)
        counts_sim, _ = np.histogram(sim, bins=bins)
        max_y = max(max_y, counts_emp.max(), counts_sim.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(PNAS_TWO_COL, 3.5))
    
    for ax, (emp_data, sim_data, title, sd_emp, sd_sim), condition in zip(axes, data_sets, ['healthy', 'unhealthy', 'balanced']):
        ax.hist(emp_data, bins=bins, alpha=alpha, color=colors['empirical'], label='Empirical')
        ax.hist(sim_data, bins=bins, alpha=alpha, color=colors['simulated'], label='Simulated')
        ax.axvline(emp_data.mean(), color=colors['empirical'], linestyle='--', linewidth=2.5)
        ax.axvline(sim_data.mean(), color=colors['simulated'], linestyle='--', linewidth=2.5)
        
        # Calculate mean bias for this condition
        mean_bias = individual_df[individual_df['condition'] == condition]['fitted_bias_strength'].mean()
        
        ax.text(0.05, 0.92, f'SD = {sd_emp:.2f}', transform=ax.transAxes, 
                color=colors['empirical'], fontsize=9, fontweight='bold')
        ax.text(0.05, 0.85, f'SD = {sd_sim:.2f}', transform=ax.transAxes, 
                color=colors['simulated'], fontsize=9, fontweight='bold')
        
        title_text = f'{title}\nBias = {mean_bias:.2f}'
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel('Health-Taste Belief')
        ax.set_ylim(0, max_y * 1.1)
        #ax.set_xlim(0, 1)
    
    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{results_dir}/individual_distribution_comparison.pdf', bbox_inches='tight', format='pdf')
    
    # Create and save a table of standard deviations
    sd_table = pd.DataFrame({
        'Condition': ['Healthy', 'Unhealthy', 'Balanced'],
        'Empirical SD': [sd_healthy_emp, sd_unhealthy_emp, sd_balanced_emp],
        'Individual Sim SD': [sd_healthy_sim, sd_unhealthy_sim, sd_balanced_sim],
        'Difference': [sd_healthy_emp - sd_healthy_sim, 
                      sd_unhealthy_emp - sd_unhealthy_sim, 
                      sd_balanced_emp - sd_balanced_sim]
    })
    sd_table.to_csv(f'{results_dir}/individual_standard_deviation_comparison.csv', index=False)
    
    plt.close()
    
    return fig


def plot_performance_vs_bias_strength(results, results_dir=None):
    """
    Create plots of mean absolute error and KS statistic vs. bias strength for all environments.

    Parameters:
    results (dict): Dictionary with performance metrics from optimize_and_evaluate_parameters()
    results_dir (str): Directory to save results, if None a timestamped directory is created
    """
    # Set PNAS style
    set_pnas_style()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    if results_dir is None:
        results_dir = f'paper_1/results/results_{strftime("%y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)

    # Define color palette
    palette = {
        'Healthy': '#1f77b4',   # soft blue
        'Unhealthy': '#d62728', # muted red
        'Balanced': '#2ca02c'   # green
    }

    # Convert results to dataframe format suitable for plotting
    df_error = pd.DataFrame({
        'Bias Strength': results['bias_strength'] * 3,
        'Mean Absolute Error': results['error_healthy'] + results['error_unhealthy'] + results['error_balanced'],
        'Environment': ['Healthy'] * len(results['bias_strength']) + 
                      ['Unhealthy'] * len(results['bias_strength']) + 
                      ['Balanced'] * len(results['bias_strength'])
    })

    df_ks = pd.DataFrame({
        'Bias Strength': results['bias_strength'] * 3,
        'KS Statistic': results['ks_healthy'] + results['ks_unhealthy'] + results['ks_balanced'],
        'Environment': ['Healthy'] * len(results['bias_strength']) + 
                      ['Unhealthy'] * len(results['bias_strength']) + 
                      ['Balanced'] * len(results['bias_strength'])
    })

    df_error.to_csv(f'{results_dir}/error_vs_bias_strength.csv', index=False)
    df_ks.to_csv(f'{results_dir}/ks_vs_bias_strength.csv', index=False)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(PNAS_TWO_COL, 3.5))

    # Plot Mean Absolute Error vs. Bias Strength
    sns.lineplot(
        data=df_error,
        x='Bias Strength',
        y='Mean Absolute Error',
        hue='Environment',
        palette=palette,
        linewidth=2,
        ax=axes[0]
    )

    axes[0].set_xlabel('Bias Strength')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].legend().set_visible(False)

    # Plot KS Statistic vs. Bias Strength
    sns.lineplot(
        data=df_ks,
        x='Bias Strength',
        y='KS Statistic',
        hue='Environment',
        palette=palette,
        linewidth=2,
        ax=axes[1]
    )

    axes[1].set_xlabel('Bias Strength')
    axes[1].set_ylabel('KS Statistic')
    axes[1].legend().set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)


    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{results_dir}/group_performance_vs_bias_strength.pdf', bbox_inches='tight', format='pdf')
    plt.close()

    return fig


def analyze_fitted_distribution_ridge(fitted_df, results_dir=None):
    """
    Ridgeplot of fitted bias_strength values by condition.
    
    Parameters:
    fitted_df (DataFrame): DataFrame with fitted bias_strength values
    results_dir (str): Directory to save results
    
    Returns:
    dict: Statistics of the fitted distribution
    """    

    set_pnas_style()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    condition_order = ['healthy', 'unhealthy', 'balanced']
    fitted_df['condition'] = pd.Categorical(fitted_df['condition'], categories=condition_order, ordered=True)
    fitted_df = fitted_df.sort_values('condition')

    palette = {'healthy': '#1f77b4', 'unhealthy': '#d62728', 'balanced': '#2ca02c'}

    g = sns.FacetGrid(fitted_df, row="condition", hue="condition", aspect=2.0, height=1.2, palette=palette)

    g.map(sns.kdeplot, "fitted_bias_strength",
          bw_adjust=0.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    
    g.map(sns.kdeplot, "fitted_bias_strength",
          clip_on=False, color="w", lw=2, bw_adjust=0.5)

    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.15, 0.2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes, fontsize=12)

    g.map(label, "fitted_bias_strength")

    g.figure.subplots_adjust(hspace=-0.4)

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.set(xlim=(-.25, 1.25))
    g.axes[-1, 0].set_xlabel("Individual Bias Strength", fontsize=12)
    
    g.figure.set_figwidth(PNAS_TWO_COL)
    g.figure.set_figheight(4.0)

    if results_dir:
        plt.savefig(f"{results_dir}/individual_fitted_bias_distribution_ridgeplot.pdf", bbox_inches='tight', format='pdf')

    plt.close()

    return {
        'overall': {
            'mean': fitted_df['fitted_bias_strength'].mean(),
            'std': fitted_df['fitted_bias_strength'].std()
        },
        'by_condition': fitted_df.groupby('condition')['fitted_bias_strength'].agg(['mean', 'std'])
    }


def compare_group_vs_individual_performance(group_results, individual_df, results_dir, emp_healthy=None, emp_unhealthy=None, emp_balanced=None):
    """
    Compare the performance of group-level vs individual-level parameter fits
    
    Parameters:
    group_results (dict): Results from group-level optimization
    individual_df (DataFrame): Results from individual-level fits
    results_dir (str): Directory to save results
    emp_healthy, emp_unhealthy, emp_balanced: Empirical data for different conditions
    
    Returns:
    dict: Comparison metrics
    """
    set_pnas_style()
    
    # Calculate average errors per condition for individual fits
    individual_errors = individual_df.groupby('condition')['fit_error'].mean().to_dict()
    
    # Get best group-level parameters
    best_params = group_results['best_params']
    
    group_errors = {}
    group_ks = {}
    individual_ks = {}
    
    # Initialize dictionaries for standard deviations
    group_sim_sd = {}
    individual_sim_sd = {}
    emp_sd = {}
    
    # Load standard deviation data from files if available
    group_sd_file = f'{results_dir}/standard_deviation_comparison.csv'
    individual_sd_file = f'{results_dir}/individual_standard_deviation_comparison.csv'
    
    # Initialize SD dataframes
    group_sd_df = None
    individual_sd_df = None
    
    # Try to load group SD data
    if os.path.exists(group_sd_file):
        group_sd_df = pd.read_csv(group_sd_file)
    
    # Try to load individual SD data
    if os.path.exists(individual_sd_file):
        individual_sd_df = pd.read_csv(individual_sd_file)
    
    # Process each condition
    for condition in ['healthy', 'unhealthy', 'balanced']:
        # Map condition name to capitalized format for CSV matching
        condition_cap = condition.capitalize()
        
        # Get empirical data based on condition
        if condition == 'healthy':
            emp_data_condition = emp_healthy
            emp_sd[condition] = emp_healthy.std()
        elif condition == 'unhealthy':
            emp_data_condition = emp_unhealthy
            emp_sd[condition] = emp_unhealthy.std()
        elif condition == 'balanced':
            emp_data_condition = emp_balanced
            emp_sd[condition] = emp_balanced.std()
        
        # Get individual-level simulated data
        sim_data_condition = individual_df[individual_df['condition'] == condition]['simulated_ht']
        individual_sim_sd[condition] = sim_data_condition.std()
        
        # Get group-level SD from file if available
        if group_sd_df is not None:
            group_sim_sd[condition] = group_sd_df.loc[group_sd_df['Condition'] == condition_cap, 'Simulated SD'].values[0]
        else:
            group_sim_sd[condition] = 0.1 
            
        # Calculate KS statistic for individual fits
        ks_stat, _ = ks_2samp(emp_data_condition, sim_data_condition)
        individual_ks[condition] = ks_stat
        
        # Get group errors and KS stats
        bias_idx = group_results['bias_strength'].index(best_params[condition])
        error_key = f'error_{condition}'
        ks_key = f'ks_{condition}'
        group_errors[condition] = group_results[error_key][bias_idx]
        group_ks[condition] = group_results[ks_key][bias_idx]
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'Condition': ['healthy', 'unhealthy', 'balanced'],
        'Group MAE': [group_errors[c] for c in ['healthy', 'unhealthy', 'balanced']],
        'Individual MAE': [individual_errors[c] for c in ['healthy', 'unhealthy', 'balanced']],
        'Group KS': [group_ks[c] for c in ['healthy', 'unhealthy', 'balanced']],
        'Individual KS': [individual_ks[c] for c in ['healthy', 'unhealthy', 'balanced']]
    })
    
    # Add SD ratio data to comparison dataframe
    comparison['Emp SD'] = [emp_sd[c] for c in ['healthy', 'unhealthy', 'balanced']]
    comparison['Group Sim SD'] = [group_sim_sd[c] for c in ['healthy', 'unhealthy', 'balanced']]
    comparison['Individual Sim SD'] = [individual_sim_sd[c] for c in ['healthy', 'unhealthy', 'balanced']]
    comparison['Group SD Ratio'] = comparison['Emp SD'] / comparison['Group Sim SD']
    comparison['Individual SD Ratio'] = comparison['Emp SD'] / comparison['Individual Sim SD']
    
    # Add average row
    comparison.loc['Average'] = ['Average', 
                                comparison['Group MAE'].mean(), 
                                comparison['Individual MAE'].mean(),
                                comparison['Group KS'].mean(),
                                comparison['Individual KS'].mean(),
                                comparison['Emp SD'].mean(),
                                comparison['Group Sim SD'].mean(),
                                comparison['Individual Sim SD'].mean(),
                                comparison['Group SD Ratio'].mean(),
                                comparison['Individual SD Ratio'].mean()]
    
    # Save complete comparison data
    comparison.to_csv(f'{results_dir}/group_vs_individual_comparison.csv', index=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(PNAS_ONE_HALF_COL, 5.0))
    
    # MAE Comparison
    comparison_data = comparison.iloc[:-1][['Condition', 'Group MAE', 'Individual MAE']].rename(
        columns={'Group MAE': 'Group', 'Individual MAE': 'Individual'}
    ).melt(id_vars='Condition', var_name='Method', value_name='MAE')
    sns.barplot(x='Condition', y='MAE', hue='Method', data=comparison_data, ax=axes[0, 0])
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=35)
    
    # KS Comparison
    ks_comparison_data = comparison.iloc[:-1][['Condition', 'Group KS', 'Individual KS']].rename(
        columns={'Group KS': 'Group', 'Individual KS': 'Individual'}
    ).melt(id_vars='Condition', var_name='Method', value_name='KS Statistic')
    sns.barplot(x='Condition', y='KS Statistic', hue='Method', data=ks_comparison_data, ax=axes[0, 1])
    axes[0, 1].set_ylabel('KS Statistic')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=35)
    
    # Bias Strength Comparison
    bias_comparison = pd.DataFrame({
        'Condition': ['healthy', 'unhealthy', 'balanced'],
        'Group': [best_params[c] for c in ['healthy', 'unhealthy', 'balanced']],
        'Individual': [individual_df[individual_df['condition'] == c]['fitted_bias_strength'].mean() 
                          for c in ['healthy', 'unhealthy', 'balanced']]
    })
    bias_data = bias_comparison.melt(id_vars='Condition', var_name='Method', value_name='Bias Strength')
    sns.barplot(x='Condition', y='Bias Strength', hue='Method', data=bias_data, ax=axes[1, 0])
    axes[1, 0].set_ylabel('Bias Strength')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=35)
    
    # SD Ratio Comparison
    sd_ratio_data = comparison.iloc[:-1][['Condition', 'Group SD Ratio', 'Individual SD Ratio']].rename(
        columns={'Group SD Ratio': 'Group', 'Individual SD Ratio': 'Individual'}
    ).melt(id_vars='Condition', var_name='Method', value_name='SD Ratio')
    sns.barplot(x='Condition', y='SD Ratio', hue='Method', data=sd_ratio_data, ax=axes[1, 1])
    axes[1, 1].set_ylabel('SD Ratio')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=35)
    
    for ax in axes.flatten():
        ax.legend().set_visible(False)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'{results_dir}/group_vs_individual_comparison.pdf', bbox_inches='tight', format='pdf')
    plt.close()
    
    return comparison