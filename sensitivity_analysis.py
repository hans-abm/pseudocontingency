from model import LearningModel
from agent import LearningAgent
import pandas as pd
import os
from time import strftime
import numpy as np
from multiprocessing import Pool, cpu_count

from calibrate import (
    convert_coef_to_prob, 
    error, 
    ks_stat
)

# Load and preprocess empirical data
emp_data = pd.read_csv('data/Study2_data_processed.csv', delimiter=';')
emp_data = emp_data.map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

# Subset empirical data by experimental condition
emp_healthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 1, 'ht_rel_c2'])
emp_unhealthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 2, 'ht_rel_c2'])
emp_balanced = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 3, 'ht_rel_c2'])


def run_simulation_with_learning_rate(x, learning_rate, num_runs=1):
    """
    Run the simulation with a specific learning rate in all environments.
    
    Parameters:
    x (list): Model parameters [bias_strength]
    learning_rate (float): Learning rate parameter
    num_runs (int): Number of times to run the simulation
    
    Returns:
    tuple: (healthy_results, unhealthy_results, balanced_results)
    """
    bias_strength = round(x[0], 2)
    
    healthy_results = []
    unhealthy_results = []
    balanced_results = []
    
    for _ in range(num_runs):
        # Run healthy environment simulation
        healthy_model = LearningModel(
            N=52, 
            env_type='healthy', 
            bias_strength=bias_strength,
            learning_rate=learning_rate
        )
        for i in range(24):
            healthy_model.step()
        
        healthy_data = healthy_model.datacollector.get_agent_vars_dataframe().reset_index()
        healthy_data = healthy_data[healthy_data['Step'] == healthy_data['Step'].iloc[-1]][['ht']]
        healthy_results.append(healthy_data)
        
        # Run unhealthy environment simulation
        unhealthy_model = LearningModel(
            N=52, 
            env_type='unhealthy', 
            bias_strength=bias_strength,
            learning_rate=learning_rate
        )
        for i in range(24):
            unhealthy_model.step()
        
        unhealthy_data = unhealthy_model.datacollector.get_agent_vars_dataframe().reset_index()
        unhealthy_data = unhealthy_data[unhealthy_data['Step'] == unhealthy_data['Step'].iloc[-1]][['ht']]
        unhealthy_results.append(unhealthy_data)

        # Run balanced environment simulation
        balanced_model = LearningModel(
            N=52, 
            env_type='balanced', 
            bias_strength=bias_strength,
            learning_rate=learning_rate
        )
        for i in range(24):
            balanced_model.step()

        balanced_data = balanced_model.datacollector.get_agent_vars_dataframe().reset_index()
        balanced_data = balanced_data[balanced_data['Step'] == balanced_data['Step'].iloc[-1]][['ht']]
        balanced_results.append(balanced_data)
    
    return healthy_results, unhealthy_results, balanced_results


def run_simulation_wrapper_with_lr(params):
    """
    Wrapper function for multiprocessing that includes learning rate parameter
    """
    bias_strength, learning_rate, num_runs = params
    return bias_strength, learning_rate, run_simulation_with_learning_rate([bias_strength], learning_rate, num_runs)


def group_level_sensitivity_analysis(learning_rates, bias_range=None, num_runs=5, verbose=True, num_processes=None):
    """
    Run group-level sensitivity analysis across different learning rates.
    
    Parameters:
    learning_rates (list): List of learning rate values to test
    bias_range (list): List of bias strength values to evaluate
    num_runs (int): Number of simulation runs per parameter set
    verbose (bool): Whether to print progress information
    num_processes (int): Number of processes to use
    
    Returns:
    dict: Results for each learning rate
    """
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)]
    
    if num_processes is None:
        num_processes = cpu_count()
    
    if verbose:
        print(f"Running sensitivity analysis with {len(learning_rates)} learning rates")
        print(f"Learning rates: {learning_rates}")
        print(f"Using {num_processes} processes across {len(bias_range)} bias values")
    
    all_results = {}
    
    for lr in learning_rates:
        if verbose:
            print(f"\n--- Testing learning rate: {lr} ---")
        
        # Prepare parameters for this learning rate
        params = [(bias, lr, num_runs) for bias in bias_range]
        
        # Initialize results storage for this learning rate
        results = {
            'learning_rate': lr,
            'bias_strength': [],
            'error_healthy': [],
            'error_unhealthy': [],
            'error_balanced': [],
            'ks_healthy': [],
            'ks_unhealthy': [],
            'ks_balanced': []
        }
        
        # Track best parameters for each environment
        best_params = {
            'healthy': {'value': None, 'ks': float('inf')},
            'unhealthy': {'value': None, 'ks': float('inf')},
            'balanced': {'value': None, 'ks': float('inf')}
        }
        
        # Run simulations in parallel for this learning rate
        with Pool(processes=num_processes) as pool:
            for bias_strength, learning_rate, (healthy_results, unhealthy_results, balanced_results) in pool.imap_unordered(run_simulation_wrapper_with_lr, params):
                
                # Calculate errors
                errors = error(healthy_results, unhealthy_results, balanced_results)
                
                # Calculate KS statistics
                ks_stats = ks_stat(healthy_results, unhealthy_results, balanced_results)
                
                # Store results
                results['bias_strength'].append(bias_strength)
                results['error_healthy'].append(errors['healthy'])
                results['error_unhealthy'].append(errors['unhealthy'])
                results['error_balanced'].append(errors['balanced'])
                results['ks_healthy'].append(ks_stats['healthy'])
                results['ks_unhealthy'].append(ks_stats['unhealthy'])
                results['ks_balanced'].append(ks_stats['balanced'])
                
                # Update best parameters based on KS statistic
                if ks_stats['healthy'] < best_params['healthy']['ks']:
                    best_params['healthy']['ks'] = ks_stats['healthy']
                    best_params['healthy']['value'] = bias_strength

                if ks_stats['unhealthy'] < best_params['unhealthy']['ks']:
                    best_params['unhealthy']['ks'] = ks_stats['unhealthy']
                    best_params['unhealthy']['value'] = bias_strength

                if ks_stats['balanced'] < best_params['balanced']['ks']:
                    best_params['balanced']['ks'] = ks_stats['balanced']
                    best_params['balanced']['value'] = bias_strength
        
        # Sort results by bias_strength
        sort_indices = np.argsort(results['bias_strength'])
        for key in results.keys():
            if key not in ['learning_rate', 'best_params']:
                results[key] = [results[key][i] for i in sort_indices]
        
        # Add best parameters to results
        results['best_params'] = {
            'healthy': best_params['healthy']['value'],
            'unhealthy': best_params['unhealthy']['value'],
            'balanced': best_params['balanced']['value']
        }
        
        if verbose:
            print(f"Best parameters for lr={lr}: {results['best_params']}")
        
        all_results[lr] = results
    
    return all_results


def fit_individual_bias_strength_with_lr_worker(params):
    """
    Worker function for fitting individual bias strengths with specific learning rate.
    
    Parameters:
    params: tuple containing (participant_row, conditions, bias_range, learning_rate, num_runs)
    
    Returns:
    dict: Fitted results for this participant with this learning rate
    """
    participant, conditions, bias_range, learning_rate, num_runs = params
    participant_id = participant['id']
    condition = conditions.get(participant['condition'])
    target_ht = convert_coef_to_prob(participant['belief_ht_overall_R'])

    # Grid search for best bias_strength
    best_error = float('inf')
    best_bias = None
    best_sim_ht = None

    for bias in bias_range:
        run_errors = []

        for _ in range(num_runs):
            # Run model with this bias strength and learning rate
            model = LearningModel(
                N=1, 
                env_type=condition, 
                bias_strength=bias,
                learning_rate=learning_rate
            )
            for _ in range(24):
                model.step()

            # Get final belief
            agent = next(a for a in model.schedule.agents if isinstance(a, LearningAgent))
            sim_ht = agent.ht_belief

            # Calculate error
            run_errors.append(abs(sim_ht - target_ht))

        # Average error across runs
        avg_error = np.mean(run_errors)

        # Update if better
        if avg_error < best_error:
            best_error = avg_error
            best_bias = bias
            best_sim_ht = sim_ht

    # Return result for this participant
    return {
        'participant_id': participant_id,
        'condition': condition,
        'learning_rate': learning_rate,
        'empirical_ht': target_ht,
        'simulated_ht': best_sim_ht,
        'fitted_bias_strength': best_bias,
        'fit_error': best_error
    }


def individual_level_sensitivity_analysis(learning_rates, empirical_data, bias_range=None, num_processes=None):
    """
    Run individual-level sensitivity analysis across different learning rates.
    
    Parameters:
    learning_rates (list): List of learning rate values to test
    empirical_data (DataFrame): DataFrame with empirical data
    bias_range (list): List of bias strength values to evaluate
    num_processes (int): Number of processes to use
    
    Returns:
    DataFrame: Combined results for all learning rates
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)]
    
    print(f"Fitting individual bias strengths across {len(learning_rates)} learning rates using {num_processes} processes")
    
    all_fitted_results = []
    conditions = {1: 'healthy', 2: 'unhealthy', 3: 'balanced'}
    
    for lr in learning_rates:
        print(f"Processing learning rate: {lr}")
        
        # Prepare parameters for this learning rate
        params = [(row, conditions, bias_range, lr, 1) for _, row in empirical_data.iterrows()]
        
        # Run fitting in parallel for this learning rate
        with Pool(processes=num_processes) as pool:
            results = list(pool.imap_unordered(fit_individual_bias_strength_with_lr_worker, params))
        
        all_fitted_results.extend(results)
    
    return pd.DataFrame(all_fitted_results)


def analyze_sensitivity_results(group_results, individual_results, results_dir):
    """
    Analyze and visualize sensitivity analysis results.
    
    Parameters:
    group_results (dict): Results from group-level sensitivity analysis
    individual_results (DataFrame): Results from individual-level sensitivity analysis
    results_dir (str): Directory to save results
    """
    
    # Create summary DataFrame for group results
    group_summary = []
    for lr, results in group_results.items():
        summary_row = {
            'learning_rate': lr,
            'best_bias_healthy': results['best_params']['healthy'],
            'best_bias_unhealthy': results['best_params']['unhealthy'],
            'best_bias_balanced': results['best_params']['balanced'],
            'min_ks_healthy': min(results['ks_healthy']),
            'min_ks_unhealthy': min(results['ks_unhealthy']),
            'min_ks_balanced': min(results['ks_balanced'])
        }
        group_summary.append(summary_row)
    
    group_summary_df = pd.DataFrame(group_summary)
    group_summary_df.to_csv(f'{results_dir}/group_sensitivity_summary.csv', index=False)
    
    # Create summary for individual results
    individual_summary = individual_results.groupby(['learning_rate', 'condition']).agg({
        'fitted_bias_strength': ['mean', 'std', 'min', 'max'],
        'fit_error': ['mean', 'std']
    }).round(4)
    
    individual_summary.to_csv(f'{results_dir}/individual_sensitivity_summary.csv')
    
    # Print summary statistics
    print("\n=== GROUP-LEVEL SENSITIVITY SUMMARY ===")
    print(group_summary_df)
    
    print("\n=== INDIVIDUAL-LEVEL SENSITIVITY SUMMARY ===")
    print(individual_summary)
    
    # Analyze stability of optimal parameters across learning rates
    print("\n=== PARAMETER STABILITY ANALYSIS ===")
    for condition in ['healthy', 'unhealthy', 'balanced']:
        col_name = f'best_bias_{condition}'
        stability = group_summary_df[col_name].std()
        mean_param = group_summary_df[col_name].mean()
        print(f"{condition.capitalize()} environment:")
        print(f"  Mean optimal bias: {mean_param:.3f}")
        print(f"  Standard deviation: {stability:.3f}")
        print(f"  Coefficient of variation: {stability/mean_param:.3f}")

    # Add this after the existing individual_summary calculation:

    # Individual-level stability across learning rates
    print("\n=== INDIVIDUAL-LEVEL STABILITY ACROSS LEARNING RATES ===")
    individual_stability = []

    for condition in ['healthy', 'unhealthy', 'balanced']:
        # Get mean fitted bias for each learning rate in this condition
        condition_means_by_lr = []
        
        for lr in individual_results['learning_rate'].unique():
            lr_condition_data = individual_results[
                (individual_results['learning_rate'] == lr) & 
                (individual_results['condition'] == condition)
            ]['fitted_bias_strength']
            condition_means_by_lr.append(lr_condition_data.mean())
        
        # Calculate stability metrics
        mean_across_lr = np.mean(condition_means_by_lr)
        std_across_lr = np.std(condition_means_by_lr)
        cv_across_lr = std_across_lr / mean_across_lr
        
        individual_stability.append({
            'condition': condition,
            'mean_bias_across_lr': mean_across_lr,
            'std_bias_across_lr': std_across_lr,
            'cv_bias_across_lr': cv_across_lr
        })
        
        print(f"{condition.capitalize()}:")
        print(f"  Mean bias across LRs: {mean_across_lr:.3f}")
        print(f"  Standard deviation: {std_across_lr:.3f}")
        print(f"  Coefficient of variation: {cv_across_lr:.3f}")

    # Save individual stability metrics
    individual_stability_df = pd.DataFrame(individual_stability)
    individual_stability_df.to_csv(f'{results_dir}/individual_stability_metrics.csv', index=False)
    
    return group_summary_df, individual_summary


def run_sensitivity_analysis(learning_rates=None, bias_range=None):
    """
    Run complete sensitivity analysis for both group and individual levels.
    
    Parameters:
    learning_rates (list): List of learning rate values to test
    bias_range (list): List of bias strength values to evaluate
    """
    # Set default parameters
    if learning_rates is None:
        learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    if bias_range is None:
        bias_range = [round(x * 0.05, 2) for x in range(0, 21)]  # 0.0 to 1.0 in 0.05 increments
    
    # Create results directory
    timestamp = strftime("%y%m%d_%H%M%S")
    results_dir = f'paper_1/results/sensitivity_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Sensitivity analysis results will be saved to: {results_dir}")
    
    # Step 1: Group-level sensitivity analysis
    print("\n=== Starting group-level sensitivity analysis ===")
    group_results = group_level_sensitivity_analysis(
        learning_rates=learning_rates,
        bias_range=bias_range,
        num_runs=1,
        verbose=True
    )
    
    # Save group results
    import pickle
    with open(f'{results_dir}/group_sensitivity_results.pkl', 'wb') as f:
        pickle.dump(group_results, f)
    
    # Step 2: Individual-level sensitivity analysis
    print("\n=== Starting individual-level sensitivity analysis ===")
    individual_results = individual_level_sensitivity_analysis(
        learning_rates=learning_rates,
        empirical_data=emp_data,
        bias_range=bias_range
    )
    
    # Save individual results
    individual_results.to_csv(f'{results_dir}/individual_sensitivity_results.csv', index=False)
    
    # Step 3: Analyze and summarize results
    print("\n=== Analyzing sensitivity results ===")
    group_summary, individual_summary = analyze_sensitivity_results(
        group_results, 
        individual_results, 
        results_dir
    )
    
    print(f"\nSensitivity analysis completed. All results saved to {results_dir}")
    
    return group_results, individual_results, group_summary, individual_summary


if __name__ == "__main__":
    # Define parameters for sensitivity analysis
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    bias_range = [round(x * 0.05, 2) for x in range(0, 21)]  # 0.0 to 1.0 in 0.05 increments
    
    # Run the complete sensitivity analysis
    group_results, individual_results, group_summary, individual_summary = run_sensitivity_analysis(
        learning_rates=learning_rates,
        bias_range=bias_range
    )