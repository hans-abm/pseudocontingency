import pandas as pd
import numpy as np
from model import LearningModel
from agent import LearningAgent
import plots
import os
from time import strftime
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_study1_data():
    """Load and preprocess Study 1 data"""
    try:
        emp_data = pd.read_csv('paper_1\data\Study1_data_processed.csv', delimiter=';')
        emp_data = emp_data.map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
        return emp_data
    except FileNotFoundError:
        print("Error: Study1_data_processed.csv not found in data/ directory")
        return None

def convert_coef_to_prob(coef_values):
    """
    Convert from [-0.5, 0.5] to [0, 1]
    """
    return coef_values + 0.5

def load_calibrated_parameters():
    """Load the best parameters from calibration study"""
    try:
        # Try to load from the most recent calibration results
        fitted_individuals = pd.read_csv('paper_1/results/c2_0_1_bias_range_0_1_10runs/fitted_individual_bias.csv')
        individual_biases = fitted_individuals['fitted_bias_strength'].values
        
        best_group_params = {
            'healthy': 1.0,
            'unhealthy': 1.0, 
            'balanced': 0 
        }
        
        return best_group_params, individual_biases
        
    except FileNotFoundError:
        print("Warning: Could not load calibrated parameters. Using default values.")
        print("Please run calibration first or update the file paths.")
        return {'healthy': 0.5, 'unhealthy': 0.5, 'balanced': 0.0}, np.array([0.5])
    

def run_group_simulation_for_validation(group_params, num_simulations=100):
    """
    Run group-level simulations using calibrated parameters for validation
    Study 1 uses healthy_b/unhealthy_b environments with different step counts
    
    If num_simulations == 1: Returns individual agent beliefs from that single run
    If num_simulations > 1: Returns averaged distributions across all simulation runs
    """
    results = {}
    
    # Map environment names to bias strengths and step counts for Study 1
    env_config = {
        'unhealthy_b': {'bias': group_params['unhealthy'], 'steps': 45, 'n_agents': 49},
        'healthy_b': {'bias': group_params['healthy'], 'steps': 45, 'n_agents': 34},
        'balanced': {'bias': group_params['balanced'], 'steps': 44, 'n_agents': 51}
    }
    
    for env_name, config in env_config.items():
        if num_simulations == 1:
            # For single simulation, return all individual agent beliefs
            model = LearningModel(N=config['n_agents'], env_type=env_name, bias_strength=config['bias'])
            
            # Run for the appropriate number of steps
            for _ in range(config['steps']):
                model.step()
            
            # Extract final beliefs for all agents
            df = model.datacollector.get_agent_vars_dataframe().reset_index()
            final_step_data = df[df['Step'] == df['Step'].max()]
            final_ht = final_step_data['ht'].values
            
            results[env_name] = final_ht
            print(f"Environment {env_name}: 1 run, {len(final_ht)} agents, mean = {final_ht.mean():.3f}, std = {final_ht.std():.3f}")
            
        else:
            # For multiple simulations, store results from each simulation run
            all_run_results = []
            
            for sim in range(num_simulations):
                # Create model with appropriate parameters for Study 1 validation
                model = LearningModel(N=config['n_agents'], env_type=env_name, bias_strength=config['bias'])
                
                # Run for the appropriate number of steps
                for _ in range(config['steps']):
                    model.step()
                
                # Extract final beliefs for this run
                df = model.datacollector.get_agent_vars_dataframe().reset_index()
                final_step_data = df[df['Step'] == df['Step'].max()]
                final_ht = final_step_data['ht'].values
                
                # Store the mean of this run
                all_run_results.append(final_ht.mean())
            
            # Convert to array for easier manipulation
            results[env_name] = np.array(all_run_results)
            print(f"Environment {env_name}: {num_simulations} runs, mean = {results[env_name].mean():.3f}, std = {results[env_name].std():.3f}")
    
    return results

def run_individual_simulation_for_validation(emp_data, individual_biases, num_runs=100):
    """
    Run individual-level simulations matching each participant with multiple runs per participant
    Study 1 conditions: 1=unhealthy_b, 2=healthy_b, 3=balanced
    """
    all_sim_results = []
    
    # Environment mapping for Study 1 (swapped from Study 2)
    condition_to_env = {1: 'unhealthy_b', 2: 'healthy_b', 3: 'balanced'}
    
    # Step counts for each environment
    env_steps = {'unhealthy_b': 45, 'healthy_b': 45, 'balanced': 44}
    
    print(f"Running {num_runs} simulations per participant for individual validation...")
    
    for _, row in emp_data.iterrows():
        participant_runs = []
        env_type = condition_to_env[row['condition']]
        
        # Run multiple simulations for this participant
        for run in range(num_runs):
            # Randomly sample a bias strength from calibrated individual biases
            bias = np.random.choice(individual_biases)
            
            # Run single-agent simulation
            model = LearningModel(N=1, env_type=env_type, bias_strength=bias)
            
            # Use appropriate step count for this environment
            for _ in range(env_steps[env_type]):
                model.step()
            
            # Extract final belief
            agent = next(a for a in model.schedule.agents if isinstance(a, LearningAgent))
            
            participant_runs.append({
                'participant_id': row['id'],
                'condition': row['condition'],
                'condition_name': env_type,
                'run': run,
                'empirical_ht': convert_coef_to_prob(row['belief_ht_R']),
                'simulated_ht': agent.ht_belief,
                'used_bias': bias
            })
        
        # Calculate mean across runs for this participant
        run_results = pd.DataFrame(participant_runs)
        mean_simulated_ht = run_results['simulated_ht'].mean()
        mean_used_bias = run_results['used_bias'].mean()
        
        # Store the averaged result for this participant
        all_sim_results.append({
            'participant_id': row['id'],
            'condition': row['condition'],
            'condition_name': env_type,
            'empirical_ht': convert_coef_to_prob(row['belief_ht_R']),
            'simulated_ht': mean_simulated_ht,
            'used_bias': mean_used_bias,
            'num_runs': num_runs
        })
    
    return pd.DataFrame(all_sim_results)

def plot_group_distribution_comparison(group_params, num_simulations=100, results_dir=None, 
                                     emp_healthy=None, emp_unhealthy=None, emp_balanced=None):
    """
    Plot the comparison of simulated distributions with empirical data for all environments.
    Now properly shows the distribution of run averages across multiple simulations.
    """
    # Set PNAS style
    plots.set_pnas_style()
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    if results_dir is None:
        results_dir = f'paper_1/results/results_{strftime("%y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)

    # Run group simulations - now returns averaged results
    group_sim_results = run_group_simulation_for_validation(group_params, num_simulations)
    
    # Extract simulated data (now these are averages across runs)
    sim_healthy_ht = group_sim_results['healthy_b']
    sim_unhealthy_ht = group_sim_results['unhealthy_b'] 
    sim_balanced_ht = group_sim_results['balanced']

    # Calculate standard deviations
    sd_healthy_emp = emp_healthy.std()
    sd_unhealthy_emp = emp_unhealthy.std()
    sd_balanced_emp = emp_balanced.std()
    
    sd_healthy_sim = sim_healthy_ht.std()
    sd_unhealthy_sim = sim_unhealthy_ht.std()
    sd_balanced_sim = sim_balanced_ht.std()

    # Prepare data for histogram plotting
    data_sets = [
        (emp_healthy, sim_healthy_ht, 'Healthy', group_params['healthy'], sd_healthy_emp, sd_healthy_sim),
        (emp_unhealthy, sim_unhealthy_ht, 'Unhealthy', group_params['unhealthy'], sd_unhealthy_emp, sd_unhealthy_sim),
        (emp_balanced, sim_balanced_ht, 'Balanced', group_params['balanced'], sd_balanced_emp, sd_balanced_sim)
    ]

    bins = 5
    alpha = 0.6
    colors = {'empirical': '#1f77b4', 'simulated': '#d62728'}

    # Determine common y-limit without plotting
    max_y = 0
    for emp, sim, *_ in data_sets:
        counts_emp, _ = np.histogram(emp, bins=bins)
        counts_sim, _ = np.histogram(sim, bins=bins)
        max_y = max(max_y, counts_emp.max(), counts_sim.max())

    # Plotting - use PNAS two-column width
    fig, axes = plt.subplots(1, 3, figsize=(plots.PNAS_TWO_COL, 3.5))

    for ax, (emp_data, sim_data, title, bias_val, sd_emp, sd_sim) in zip(axes, data_sets):
        # Convert pandas Series to numpy array if needed
        emp_values = emp_data.values if hasattr(emp_data, 'values') else emp_data
        sim_values = sim_data.values if hasattr(sim_data, 'values') else sim_data
        
        # Plot histograms
        ax.hist(emp_values, bins=bins, alpha=alpha, color=colors['empirical'], label='Empirical', density=False)
        ax.hist(sim_values, bins=bins, alpha=alpha, color=colors['simulated'], label='Simulated', density=False)
        
        # Plot mean lines
        ax.axvline(emp_values.mean(), color=colors['empirical'], linestyle='--', linewidth=2.5)
        ax.axvline(sim_values.mean(), color=colors['simulated'], linestyle='--', linewidth=2.5)
        
        # Add SD text directly on the plot with corresponding colors
        ax.text(0.05, 0.92, f'SD = {sd_emp:.2f}', transform=ax.transAxes, 
                color=colors['empirical'], fontsize=9, fontweight='bold')
        ax.text(0.05, 0.85, f'SD = {sd_sim:.2f}', transform=ax.transAxes, 
                color=colors['simulated'], fontsize=9, fontweight='bold')
        
        title_text = f'{title}\nBias = {bias_val:.2f}'
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel('Health-Taste Belief')
        ax.set_ylim(0, max_y * 1.1)

    axes[0].set_ylabel('Frequency')
    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

    # Save and tidy
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
                      sd_balanced_emp - sd_balanced_sim],
        'Empirical Mean': [emp_healthy.mean(), emp_unhealthy.mean(), emp_balanced.mean()],
        'Simulated Mean': [sim_healthy_ht.mean(), sim_unhealthy_ht.mean(), sim_balanced_ht.mean()],
        'Empirical N': [len(emp_healthy), len(emp_unhealthy), len(emp_balanced)],
        'Simulation Runs': [len(sim_healthy_ht), len(sim_unhealthy_ht), len(sim_balanced_ht)]
    })
    sd_table.to_csv(f'{results_dir}/standard_deviation_comparison.csv', index=False)
    
    return fig, group_sim_results

def plot_individual_distribution_comparison(individual_sim_results, results_dir=None, 
                                          emp_healthy=None, emp_unhealthy=None, emp_balanced=None):
    """
    Plot the comparison of individually fitted model distributions with empirical data for all environments.
    Now uses averaged results across multiple runs per participant.
    """
    # Set PNAS style
    plots.set_pnas_style()
    
    if results_dir is None:
        results_dir = f'paper_1/results/results_{strftime("%y%m%d_%H%M%S")}'
        os.makedirs(results_dir, exist_ok=True)
    
    # Calculate standard deviations for empirical data
    sd_healthy_emp = emp_healthy.std()
    sd_unhealthy_emp = emp_unhealthy.std()
    sd_balanced_emp = emp_balanced.std()
    
    sim_healthy = individual_sim_results[individual_sim_results['condition'] == 2]['simulated_ht']    # condition 2 = healthy_b
    sim_unhealthy = individual_sim_results[individual_sim_results['condition'] == 1]['simulated_ht']  # condition 1 = unhealthy_b
    sim_balanced = individual_sim_results[individual_sim_results['condition'] == 3]['simulated_ht']   # condition 3 = balanced
    
    # Calculate standard deviations for simulated data
    sd_healthy_sim = sim_healthy.std()
    sd_unhealthy_sim = sim_unhealthy.std()
    sd_balanced_sim = sim_balanced.std()
    
    # Calculate mean bias for each condition
    mean_bias_healthy = individual_sim_results[individual_sim_results['condition'] == 2]['used_bias'].mean()
    mean_bias_unhealthy = individual_sim_results[individual_sim_results['condition'] == 1]['used_bias'].mean()
    mean_bias_balanced = individual_sim_results[individual_sim_results['condition'] == 3]['used_bias'].mean()
    
    # Get number of runs used
    num_runs = individual_sim_results['num_runs'].iloc[0] if 'num_runs' in individual_sim_results.columns else 1
    
    # Prepare data for histogram plotting
    data_sets = [
        (emp_healthy, sim_healthy, 'Healthy', mean_bias_healthy, sd_healthy_emp, sd_healthy_sim),
        (emp_unhealthy, sim_unhealthy, 'Unhealthy', mean_bias_unhealthy, sd_unhealthy_emp, sd_unhealthy_sim),
        (emp_balanced, sim_balanced, 'Balanced', mean_bias_balanced, sd_balanced_emp, sd_balanced_sim)
    ]
    
    bins = 5
    alpha = 0.6
    colors = {'empirical': '#1f77b4', 'simulated': '#d62728'}
    
    # Determine common y-limit without plotting
    max_y = 0
    for emp, sim, *_ in data_sets:
        counts_emp, _ = np.histogram(emp, bins=bins)
        counts_sim, _ = np.histogram(sim, bins=bins)
        max_y = max(max_y, counts_emp.max(), counts_sim.max())
    
    # Plotting - use PNAS two-column width
    fig, axes = plt.subplots(1, 3, figsize=(plots.PNAS_TWO_COL, 3.5))
    
    for ax, (emp_data, sim_data, title, mean_bias, sd_emp, sd_sim) in zip(axes, data_sets):
        ax.hist(emp_data, bins=bins, alpha=alpha, color=colors['empirical'], label='Empirical')
        ax.hist(sim_data, bins=bins, alpha=alpha, color=colors['simulated'], label='Simulated')
        ax.axvline(emp_data.mean(), color=colors['empirical'], linestyle='--', linewidth=2.5)
        ax.axvline(sim_data.mean(), color=colors['simulated'], linestyle='--', linewidth=2.5)
        
        # Add SD text directly on the plot with corresponding colors
        ax.text(0.05, 0.92, f'SD = {sd_emp:.2f}', transform=ax.transAxes, 
                color=colors['empirical'], fontsize=9, fontweight='bold')
        ax.text(0.05, 0.85, f'SD = {sd_sim:.2f}', transform=ax.transAxes, 
                color=colors['simulated'], fontsize=9, fontweight='bold')
        
        title_text = f'{title}\nBias = {mean_bias:.2f}'
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel('Health-Taste Belief')
        ax.set_ylim(0, max_y * 1.1)
    
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
                      sd_balanced_emp - sd_balanced_sim],
        'Empirical Mean': [emp_healthy.mean(), emp_unhealthy.mean(), emp_balanced.mean()],
        'Simulated Mean': [sim_healthy.mean(), sim_unhealthy.mean(), sim_balanced.mean()],
        'Runs per Participant': [num_runs, num_runs, num_runs]
    })
    sd_table.to_csv(f'{results_dir}/individual_standard_deviation_comparison.csv', index=False)
    
    plt.close()
    
    return fig

def plot_validation_comparison(validation_metrics_df, results_dir):
    """
    Create comparison plots between group-level and individual-level validation performance
    using data from validation_metrics.csv
    
    Parameters:
    -----------
    validation_metrics_df : pd.DataFrame
        DataFrame loaded from validation_metrics.csv
    results_dir : str
        Directory to save the comparison plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plots.set_pnas_style()
    
    # Separate group and individual validation results
    group_metrics = validation_metrics_df[validation_metrics_df['validation_type'] == 'group'].copy()
    individual_metrics = validation_metrics_df[validation_metrics_df['validation_type'] == 'individual'].copy()
    
    # Map condition names to consistent format
    condition_mapping = {
        'healthy_b': 'Healthy',
        'unhealthy_b': 'Unhealthy', 
        'balanced': 'Balanced'
    }
    
    group_metrics['condition_clean'] = group_metrics['condition'].map(condition_mapping)
    individual_metrics['condition_clean'] = individual_metrics['condition'].map(condition_mapping)
    
    # Create comparison dataframe
    comparison_data = []
    
    for condition in ['Healthy', 'Unhealthy', 'Balanced']:
        group_row = group_metrics[group_metrics['condition_clean'] == condition].iloc[0]
        individual_row = individual_metrics[individual_metrics['condition_clean'] == condition].iloc[0]
        
        comparison_data.append({
            'Condition': condition,
            'Group MAE': group_row['mean_absolute_error'],
            'Individual MAE': individual_row['mean_absolute_error'],
            'Group KS': group_row['ks_statistic'],
            'Individual KS': individual_row['ks_statistic'],
            'Group Mean': group_row['simulated_mean'],
            'Individual Mean': individual_row['simulated_mean'],
            'Empirical Mean': group_row['empirical_mean'],
            'Group N': group_row['simulated_n'],
            'Individual N': individual_row['simulated_n']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison data
    comparison_df.to_csv(f'{results_dir}/validation_comparison.csv', index=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(plots.PNAS_TWO_COL, 3.5))
    
    # MAE Comparison
    mae_data = comparison_df[['Condition', 'Group MAE', 'Individual MAE']].melt(
        id_vars='Condition', 
        value_vars=['Group MAE', 'Individual MAE'],
        var_name='Method', 
        value_name='MAE'
    )
    mae_data['Method'] = mae_data['Method'].str.replace(' MAE', '')
    
    sns.barplot(data=mae_data, x='Condition', y='MAE', hue='Method', ax=axes[0])
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_xlabel('')
    
    # KS Comparison
    ks_data = comparison_df[['Condition', 'Group KS', 'Individual KS']].melt(
        id_vars='Condition',
        value_vars=['Group KS', 'Individual KS'], 
        var_name='Method',
        value_name='KS Statistic'
    )
    ks_data['Method'] = ks_data['Method'].str.replace(' KS', '')
    
    sns.barplot(data=ks_data, x='Condition', y='KS Statistic', hue='Method', ax=axes[1])
    axes[1].set_ylabel('KS Statistic')
    axes[1].set_xlabel('')
    
    # Remove individual legends and rotate x-axis labels
    for ax in axes:
        ax.legend().set_visible(False)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35)
    
    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(f'{results_dir}/validation_comparison.pdf', bbox_inches='tight', format='pdf')
    plt.close()
    
    # Print summary statistics
    print("\n=== Validation Comparison Summary ===")
    print("Mean Absolute Error (lower is better):")
    print(f"  Group average: {comparison_df['Group MAE'].mean():.4f}")
    print(f"  Individual average: {comparison_df['Individual MAE'].mean():.4f}")
    
    print("\nKS Statistic (lower is better):")
    print(f"  Group average: {comparison_df['Group KS'].mean():.4f}")
    print(f"  Individual average: {comparison_df['Individual KS'].mean():.4f}")
    
    # Determine which method performs better
    group_better_mae = (comparison_df['Group MAE'] < comparison_df['Individual MAE']).sum()
    group_better_ks = (comparison_df['Group KS'] < comparison_df['Individual KS']).sum()
    
    print(f"\nGroup method performs better in {group_better_mae}/3 conditions (MAE)")
    print(f"Group method performs better in {group_better_ks}/3 conditions (KS)")
    
    return comparison_df

def calculate_validation_metrics(emp_healthy, emp_unhealthy, emp_balanced, 
                               group_sim_results, individual_sim_results):
    """
    Calculate validation metrics comparing empirical and simulated data
    """
    from scipy.stats import ks_2samp
    
    metrics = []
    
    # Group-level metrics - map environment names to empirical data
    env_to_emp = {
        'healthy_b': emp_healthy,
        'unhealthy_b': emp_unhealthy, 
        'balanced': emp_balanced
    }
    
    for env_name, emp_data in env_to_emp.items():
        sim_data = group_sim_results[env_name]
        
        # Mean absolute error
        mae = abs(emp_data.mean() - sim_data.mean())
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_2samp(emp_data, sim_data)
        
        metrics.append({
            'validation_type': 'group',
            'condition': env_name,
            'empirical_mean': emp_data.mean(),
            'simulated_mean': sim_data.mean(),
            'mean_absolute_error': mae,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'empirical_n': len(emp_data),
            'simulated_n': len(sim_data)
        })
    
    # Individual-level metrics - map condition numbers to empirical data
    condition_to_emp = {
        1: ('unhealthy_b', emp_unhealthy),  # condition 1 = unhealthy_b
        2: ('healthy_b', emp_healthy),     # condition 2 = healthy_b  
        3: ('balanced', emp_balanced)      # condition 3 = balanced
    }
    
    for condition_num, (env_name, emp_data) in condition_to_emp.items():
        # Get individual simulation data for this condition
        ind_sim_data = individual_sim_results[
            individual_sim_results['condition'] == condition_num
        ]['simulated_ht']
        
        if len(ind_sim_data) > 0:
            mae = abs(emp_data.mean() - ind_sim_data.mean())
            ks_stat, ks_pval = ks_2samp(emp_data, ind_sim_data)
            
            metrics.append({
                'validation_type': 'individual',
                'condition': env_name,
                'empirical_mean': emp_data.mean(),
                'simulated_mean': ind_sim_data.mean(),
                'mean_absolute_error': mae,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'empirical_n': len(emp_data),
                'simulated_n': len(ind_sim_data)
            })
    
    return pd.DataFrame(metrics)

def validate_model_performance(emp_data, group_params, individual_biases, results_dir, 
                             individual_num_runs=100, group_num_runs=100):
    """
    Main validation function that compares model predictions to Study 1 data
    Study 1 conditions: 1=unhealthy_b, 2=healthy_b, 3=balanced
    
    Parameters:
    -----------
    individual_num_runs : int, default=100
        Number of simulation runs per participant for individual validation
    group_num_runs : int, default=100
        Number of simulation runs for group-level validation
    """
    print("=== Starting Model Validation ===")
    
    # Preprocess empirical data by condition (Study 1 mapping)
    emp_unhealthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 1, 'belief_ht_R'])  # condition 1 = unhealthy_b
    emp_healthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 2, 'belief_ht_R'])    # condition 2 = healthy_b
    emp_balanced = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 3, 'belief_ht_R'])   # condition 3 = balanced
    
    print(f"Empirical data loaded (Study 1):")
    print(f"  Unhealthy_b condition (1): {len(emp_unhealthy)} participants, mean = {emp_unhealthy.mean():.3f}")
    print(f"  Healthy_b condition (2): {len(emp_healthy)} participants, mean = {emp_healthy.mean():.3f}")  
    print(f"  Balanced condition (3): {len(emp_balanced)} participants, mean = {emp_balanced.mean():.3f}")
    
    # Run individual-level validation with multiple runs per participant
    print(f"\n--- Running individual-level validation ({individual_num_runs} runs per participant) ---")
    individual_sim_results = run_individual_simulation_for_validation(
        emp_data, individual_biases, num_runs=individual_num_runs
    )
    print(f"Individual simulation completed for {len(individual_sim_results)} participants")
    
    # Create plots and comparisons
    print("\n--- Creating validation plots ---")
    
    # Group-level distribution comparison
    group_fig, group_sim_results = plot_group_distribution_comparison(
        group_params,
        num_simulations=group_num_runs,
        results_dir=results_dir,
        emp_healthy=emp_healthy,
        emp_unhealthy=emp_unhealthy,
        emp_balanced=emp_balanced
    )
    
    print("Group simulation results:")
    for env, results in group_sim_results.items():
        print(f"  {env}: mean = {results.mean():.3f}, std = {results.std():.3f}")
    
    # Individual-level distribution comparison
    individual_fig = plot_individual_distribution_comparison(
        individual_sim_results,
        results_dir=results_dir,
        emp_healthy=emp_healthy,
        emp_unhealthy=emp_unhealthy,
        emp_balanced=emp_balanced
    )
    
    # Save individual results
    individual_sim_results.to_csv(f'{results_dir}/individual_validation_results.csv', index=False)
    
    # Calculate and save validation metrics
    validation_metrics = calculate_validation_metrics(
        emp_healthy, emp_unhealthy, emp_balanced,
        group_sim_results, individual_sim_results
    )
    
    validation_metrics.to_csv(f'{results_dir}/validation_metrics.csv', index=False)
    print(f"\nValidation metrics saved to {results_dir}/validation_metrics.csv")
    
    # Print summary of improvements
    print(f"\n=== Individual Validation Summary ===")
    print(f"• Running {individual_num_runs} simulations per participant")
    print(f"• Averaging results across runs for more stable estimates")
    print(f"• Total simulations: {len(individual_sim_results) * individual_num_runs}")
    
    return validation_metrics

def main():
    """
    Main function to run the validation analysis
    """
    # Create results directory
    timestamp = strftime("%y%m%d_%H%M%S")
    results_dir = f'paper_1/results/validation_study1_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Load Study 1 data
    emp_data = load_and_preprocess_study1_data()
    if emp_data is None:
        return
    
    # Load calibrated parameters
    group_params, individual_biases = load_calibrated_parameters()
    
    print(f"Using group parameters: {group_params}")
    print(f"Using {len(individual_biases)} individual bias values")
    
    # Run validation with multiple runs for individual validation
    validation_metrics = validate_model_performance(
        emp_data, group_params, individual_biases, results_dir,
        individual_num_runs=10,  # Number of runs per participant
        group_num_runs=10        # Number of runs for group validation
    )
    
    print("\n=== Validation Summary ===")
    print(validation_metrics.round(3))
    
    # Add comparison plot
    print("\n--- Creating validation comparison plot ---")
    comparison_df = plot_validation_comparison(validation_metrics, results_dir)
    
    print(f"\nValidation completed. All results saved to: {results_dir}")
    print(f"Comparison plot saved as: {results_dir}/validation_comparison.pdf")
    print(f"Comparison data saved as: {results_dir}/validation_comparison.csv")

if __name__ == "__main__":
    main()