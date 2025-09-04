from model import LearningModel
from agent import LearningAgent
import pandas as pd
from scipy.stats import ks_2samp
import os
from time import strftime
import numpy as np
from multiprocessing import Pool, cpu_count
import plots

# Load empirical data
emp_data = pd.read_csv('data\Study2_data_processed.csv', delimiter = ';')
emp_data = emp_data.map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

def convert_coef_to_prob(coef_values):
    """
    Convert from [-1, 1] to [0, 1]
    
    """
    return (coef_values + 1)/2

# Subset and preprocess empirical data by experimental condition
emp_healthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 1,   'ht_rel_c2'])
emp_unhealthy = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 2, 'ht_rel_c2'])
emp_balanced = convert_coef_to_prob(emp_data.loc[emp_data['condition'] == 3,  'ht_rel_c2'])

def run_simulation(x, num_runs=1):
    """
    Run the simulation in both healthy, unhealthy, and balanced environments.
    
    Parameters:
    x (list): Model parameters [bias_strength]
    num_runs (int): Number of times to run the simulation
    
    Returns:
    tuple: (healthy_results, unhealthy_results, balanced_results) each containing a list of DataFrames
    """
    bias_strength = round(x[0], 2)
    
    healthy_results = []
    unhealthy_results = []
    balanced_results = []
    

    for _ in range(num_runs):
        # Run healthy environment simulation
        healthy_model = LearningModel(N=52, env_type='healthy', bias_strength=bias_strength)
        for i in range(24): 
            healthy_model.step()
        
        healthy_data = healthy_model.datacollector.get_agent_vars_dataframe().reset_index()
        healthy_data = healthy_data[healthy_data['Step'] == healthy_data['Step'].iloc[-1]][['ht']]
        healthy_results.append(healthy_data)
        
        # Run unhealthy environment simulation
        unhealthy_model = LearningModel(N=52, env_type='unhealthy', bias_strength=bias_strength)
        for i in range(24): 
            unhealthy_model.step()
        
        unhealthy_data = unhealthy_model.datacollector.get_agent_vars_dataframe().reset_index()
        unhealthy_data = unhealthy_data[unhealthy_data['Step'] == unhealthy_data['Step'].iloc[-1]][['ht']]
        unhealthy_results.append(unhealthy_data)

        balanced_model = LearningModel(N=52, env_type='balanced', bias_strength=bias_strength)
        for i in range(24):
            balanced_model.step()

        balanced_data = balanced_model.datacollector.get_agent_vars_dataframe().reset_index()
        balanced_data = balanced_data[balanced_data['Step'] == balanced_data['Step'].iloc[-1]][['ht']]
        balanced_results.append(balanced_data)
    
    return healthy_results, unhealthy_results, balanced_results


def run_simulation_wrapper(params):
    """
    Wrapper function for multiprocessing that unpacks parameters and calls run_simulation
    """
    bias_strength, num_runs = params
    return bias_strength, run_simulation([bias_strength], num_runs)


def error(healthy_results, unhealthy_results, balanced_results):
    """
    Calculate error between simulation results and empirical data for both environments.
    
    Parameters:
    healthy_results (list): List of DataFrames from healthy environment simulations
    unhealthy_results (list): List of DataFrames from unhealthy environment simulations
    return_dict (bool): If True, return a dictionary with errors per environment
    
    Returns:
    float or dict: Combined error or dictionary of errors per environment
    """
    # Average the simulation results
    avg_healthy_ht = pd.concat(healthy_results)['ht'].mean()
    avg_unhealthy_ht = pd.concat(unhealthy_results)['ht'].mean()
    avg_balanced_ht = pd.concat(balanced_results)['ht'].mean()
    
    # Calculate errors
    error_healthy = abs(emp_healthy.mean() - avg_healthy_ht)
    error_unhealthy = abs(emp_unhealthy.mean() - avg_unhealthy_ht)
    error_balanced = abs(emp_balanced.mean() - avg_balanced_ht)
    

    return {'healthy': error_healthy, 'unhealthy': error_unhealthy, 'balanced': error_balanced}
    

def ks_stat(healthy_results, unhealthy_results, balanced_results):
    """
    Calculate average KS statistics between simulation results and empirical data over multiple runs.
    
    Parameters:
    healthy_results (list): List of DataFrames from healthy environment simulations
    unhealthy_results (list): List of DataFrames from unhealthy environment simulations
    balanced_results (list): List of DataFrames from balanced environment simulations
    return_dict (bool): If True, return a dictionary with KS stats per environment
    
    Returns:
    float or dict: Combined average KS statistic or dictionary of average KS stats per environment
    """   
    # Calculate KS statistics for each run and then average
    healthy_ks_values = []
    unhealthy_ks_values = []
    balanced_ks_values = []
    
    for i in range(len(healthy_results)):
        # Get individual run results
        sim_healthy_ht = healthy_results[i]['ht']
        sim_unhealthy_ht = unhealthy_results[i]['ht']
        sim_balanced_ht = balanced_results[i]['ht']
        
        # Calculate KS statistics for this run
        ks_healthy, _ = ks_2samp(sim_healthy_ht, emp_healthy)
        ks_unhealthy, _ = ks_2samp(sim_unhealthy_ht, emp_unhealthy)
        ks_balanced, _ = ks_2samp(sim_balanced_ht, emp_balanced)
        
        # Store the values
        healthy_ks_values.append(ks_healthy)
        unhealthy_ks_values.append(ks_unhealthy)
        balanced_ks_values.append(ks_balanced)
    
    # Calculate average KS statistics
    avg_ks_healthy = sum(healthy_ks_values) / len(healthy_ks_values)
    avg_ks_unhealthy = sum(unhealthy_ks_values) / len(unhealthy_ks_values)
    avg_ks_balanced = sum(balanced_ks_values) / len(balanced_ks_values)
    

    return {'healthy': avg_ks_healthy, 'unhealthy': avg_ks_unhealthy, 'balanced': avg_ks_balanced}


def group_level_parallel(bias_range=None, num_runs=5, verbose=True, num_processes=None):
    """
    Parallel version of optimize_and_evaluate_parameters using multiprocessing.
    
    Parameters:
    bias_range (list): List of bias strength values to evaluate
    num_runs (int): Number of simulation runs per parameter set
    verbose (bool): Whether to print progress information
    num_processes (int): Number of processes to use, defaults to number of CPU cores
    
    Returns:
    dict: Dictionary with performance metrics for each bias strength and environment
    """
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)] #[round(x * 0.01, 2) for x in range(0, 101)]  # 0.0 to 1.0 in 0.01 increments
    
    if num_processes is None:
        num_processes = cpu_count()
    
    if verbose:
        print(f"Running optimization with {num_processes} processes across {len(bias_range)} parameter values")
    
    # Prepare parameters for parallel processing
    params = [(bias, num_runs) for bias in bias_range]
    
    # Initialize results storage
    results = {
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
    
    # Run simulations in parallel
    with Pool(processes=num_processes) as pool:
        for bias_strength, (healthy_results, unhealthy_results, balanced_results) in pool.imap_unordered(run_simulation_wrapper, params):
            
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
        if key != 'best_params':
            results[key] = [results[key][i] for i in sort_indices]
    
    # Add best parameters to results
    results['best_params'] = {
        'healthy': best_params['healthy']['value'],
        'unhealthy': best_params['unhealthy']['value'],
        'balanced': best_params['balanced']['value']
    }
    
    return results


def fit_individual_bias_strength_worker(params):
    """
    Worker function for parallelizing individual bias strength fitting with multiple runs.

    Parameters:
    params: tuple containing (participant_row, conditions, bias_range, num_runs)

    Returns:
    dict: Fitted results for this participant
    """
    participant, conditions, bias_range, num_runs = params
    participant_id = participant['id']
    condition = conditions.get(participant['condition'])
    target_ht = convert_coef_to_prob(participant['belief_ht_overall_R'])

    # Grid search for best bias_strength
    best_error = float('inf')
    best_bias = None
    best_sim_ht = None

    # Use provided bias_range
    for bias in bias_range:
        run_errors = []

        for _ in range(num_runs):
            # Run model with this bias strength
            model = LearningModel(N=1, env_type=condition, bias_strength=bias)
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
        'empirical_ht': target_ht,
        'simulated_ht': best_sim_ht,
        'fitted_bias_strength': best_bias,
        'fit_error': best_error
    }


def fit_individual_bias_strengths_parallel(empirical_data, bias_range=None, num_processes=None):
    """
    Parallel version of fit_individual_bias_strengths using multiprocessing.
    
    Parameters:
    empirical_data (DataFrame): DataFrame with empirical data
    bias_range (list): List of bias strength values to evaluate
    num_processes (int): Number of processes to use, defaults to number of CPU cores
    
    Returns:
    DataFrame: DataFrame with fitted bias_strength values for each participant ID
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)]
    
    print(f"Fitting individual bias strengths using {num_processes} processes")
    
    # Group by condition
    conditions = {1: 'healthy', 2: 'unhealthy', 3: 'balanced'}
    
    # Prepare parameters for parallel processing
    params = [(row, conditions, bias_range, 10) for _, row in empirical_data.iterrows()]
    
    # Run fitting in parallel
    with Pool(processes=num_processes) as pool:
        results = list(pool.imap_unordered(fit_individual_bias_strength_worker, params))
    
    return pd.DataFrame(results)


def run(bias_range=None):
    """
    Run group-level and individual-level analyses of bias_strength calibration to empirical data.
    
    Parameters:
    bias_range (list): List of bias strength values to evaluate, default is 0.0 to 1.0 in 0.1 increments
    """
    # Set default bias_range if not provided
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)]
    
    # Create a single results directory for all outputs
    timestamp = strftime("%y%m%d_%H%M%S")
    results_dir = f'paper_1/results/calibration_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    print(f"All results will be saved to: {results_dir}")
    
    # Step 1: Group-level optimization
    print("\n=== Starting group-level parameter optimization ===")
    group_results = group_level_parallel(
        bias_range=bias_range,
        num_runs=10,
        verbose=True
    )
    
    # Extract and report best parameters
    best_params = [
        group_results['best_params']['healthy'],
        group_results['best_params']['unhealthy'],
        group_results['best_params']['balanced']
    ]
    print(f"\nBest Group Parameters: {best_params}")
    
    # Plot group-level results 
    plots.plot_performance_vs_bias_strength(group_results, results_dir=results_dir)
    plots.plot_distribution_comparison(
        best_params, 
        num_simulations=10, 
        results_dir=results_dir,
        run_simulation=run_simulation,
        emp_healthy=emp_healthy,
        emp_unhealthy=emp_unhealthy,
        emp_balanced=emp_balanced
    )
    
    # Step 2: Individual-level analysis
    print("\n=== Starting individual-level parameter fitting ===")
    fitted_df = fit_individual_bias_strengths_parallel(emp_data, bias_range=bias_range)
    fitted_df.to_csv(f'{results_dir}/fitted_individual_bias.csv', index=False)
    
    # Plot individual-level distribution comparison - updated to use the imported functions
    print("\n=== Plotting individual-level distribution comparison ===")
    plots.analyze_fitted_distribution_ridge(fitted_df, results_dir=results_dir)
    plots.plot_individual_distribution_comparison(
        fitted_df, 
        results_dir=results_dir,
        emp_healthy=emp_healthy,
        emp_unhealthy=emp_unhealthy,
        emp_balanced=emp_balanced
    )
        
    # Step 3: Compare group vs individual fits - updated to use the imported functions
    print("\n=== Comparing group vs individual performance ===")
    comparison = plots.compare_group_vs_individual_performance(
        group_results, 
        fitted_df, 
        results_dir,
        emp_healthy=emp_healthy,
        emp_unhealthy=emp_unhealthy,
        emp_balanced=emp_balanced
    )
    
    print("\nGroup vs Individual Performance Comparison:")
    print(comparison)
    comparison.to_csv(f'{results_dir}/group_vs_individual_comparison.csv', index=False)
    
    print(f"\nAll analyses completed. Results saved to {results_dir}")


if __name__ == "__main__":

    bias_range = [round(x * 0.05, 2) for x in range(0, 21)] # [round(-1 + x * 0.05, 2) for x in range(0, 61)] [round(x * 0.05, 2) for x in range(-40, 41)]
    run(bias_range = bias_range)