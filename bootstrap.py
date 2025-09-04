"""Bootstrap calibration analysis for bias parameter estimation."""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.stats import ks_2samp
import os
from time import strftime
from model import LearningModel
from agent import LearningAgent

PRACTICAL_SETTINGS = {
    'development': {
        'n_bootstrap': 100,
        'bias_range_step': 0.05,
        'description': 'Fast testing'
    },
    'moderate': {
        'n_bootstrap': 1000,
        'bias_range_step': 0.1,
        'description': 'Moderate analysis'
    },
    'full': {
        'n_bootstrap': 1000,
        'bias_range_step': 0.05,
        'description': 'Full analysis'
    }
}

class OptimizedLearningModel(LearningModel):
    """Optimized model for bootstrap calibration."""
    
    def __init__(self, N=156, bias_strength=0.5, env_type='unhealthy', 
                 noise=0.05, seed=None, learning_rate=0.9, steps=24, bias_strengths=None):
        
        width = steps  
        height = N
        
        self.bias_strengths = bias_strengths if bias_strengths is not None else [bias_strength] * N
        
        super().__init__(N, width, height, bias_strength, env_type, noise, seed, learning_rate)
        
        if bias_strengths is not None:
            for i, agent in enumerate(self.schedule.agents):
                if isinstance(agent, LearningAgent) and i < len(bias_strengths):
                    agent.bias_strength = bias_strengths[i]
        
        self.datacollector = None

    def step(self):
        """Optimized step without data collection."""
        self.schedule.step()

def bootstrap_resample(data, seed=None):
    """Create bootstrap resample of data."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(data)
    bootstrap_indices = np.random.choice(n, size=n, replace=True)
    bootstrap_sample = data.iloc[bootstrap_indices].copy()
    bootstrap_sample.reset_index(drop=True, inplace=True)
    
    return bootstrap_sample

def fast_group_calibration(bootstrap_data, bias_range):
    """Fast group-level calibration."""
    def convert_coef_to_prob(coef_values):
        return (coef_values + 1) / 2
    
    emp_healthy = convert_coef_to_prob(bootstrap_data.loc[bootstrap_data['condition'] == 1, 'ht_rel_c2'])
    emp_unhealthy = convert_coef_to_prob(bootstrap_data.loc[bootstrap_data['condition'] == 2, 'ht_rel_c2'])
    emp_balanced = convert_coef_to_prob(bootstrap_data.loc[bootstrap_data['condition'] == 3, 'ht_rel_c2'])
    
    best_params = {
        'healthy': {'bias': None, 'ks': float('inf')},
        'unhealthy': {'bias': None, 'ks': float('inf')},
        'balanced': {'bias': None, 'ks': float('inf')}
    }
    
    for bias in bias_range:
        env_results = {}
        
        for env_name, env_type in [('healthy', 'healthy'), ('unhealthy', 'unhealthy'), ('balanced', 'balanced')]:
            run_results = []
            
            model = OptimizedLearningModel(N=52, env_type=env_type, bias_strength=bias, steps=24)
            
            for _ in range(24):
                model.step()
            
            final_beliefs = [agent.ht_belief for agent in model.schedule.agents 
                           if isinstance(agent, LearningAgent)]
            run_results.extend(final_beliefs)
            
            env_results[env_name] = np.array(run_results)
        
        ks_healthy, _ = ks_2samp(env_results['healthy'], emp_healthy)
        ks_unhealthy, _ = ks_2samp(env_results['unhealthy'], emp_unhealthy)
        ks_balanced, _ = ks_2samp(env_results['balanced'], emp_balanced)
        
        if ks_healthy < best_params['healthy']['ks']:
            best_params['healthy'] = {'bias': bias, 'ks': ks_healthy}
        
        if ks_unhealthy < best_params['unhealthy']['ks']:
            best_params['unhealthy'] = {'bias': bias, 'ks': ks_unhealthy}
            
        if ks_balanced < best_params['balanced']['ks']:
            best_params['balanced'] = {'bias': bias, 'ks': ks_balanced}
    
    return {
        'healthy': best_params['healthy']['bias'],
        'unhealthy': best_params['unhealthy']['bias'],
        'balanced': best_params['balanced']['bias']
    }

def fast_individual_calibration(bootstrap_data, bias_range):
    """Fast individual-level calibration using multi-agent approach."""
    def convert_coef_to_prob(coef_values):
        return (coef_values + 1) / 2
    
    conditions = {1: 'healthy', 2: 'unhealthy', 3: 'balanced'}
    
    fitted_biases_by_condition = {
        'healthy': [],
        'unhealthy': [],
        'balanced': []
    }
    
    for condition_num, condition_name in conditions.items():
        condition_data = bootstrap_data[bootstrap_data['condition'] == condition_num]
        
        if len(condition_data) == 0:
            continue
        
        for _, row in condition_data.iterrows():
            target_ht = convert_coef_to_prob(row['belief_ht_overall_R'])
            
            best_error = float('inf')
            best_bias = None
            
            model = OptimizedLearningModel(
                N=len(bias_range), 
                env_type=condition_name, 
                bias_strengths=bias_range,
                steps=24
            )
            
            for _ in range(24):
                model.step()
            
            final_beliefs = [agent.ht_belief for agent in model.schedule.agents 
                           if isinstance(agent, LearningAgent)]
            
            errors = [abs(belief - target_ht) for belief in final_beliefs]
            min_error_idx = np.argmin(errors)
            
            if errors[min_error_idx] < best_error:
                best_error = errors[min_error_idx]
                best_bias = bias_range[min_error_idx]
            
            fitted_biases_by_condition[condition_name].append(best_bias)
    
    return fitted_biases_by_condition

def bootstrap_calibration_worker(args):
    """Worker function for parallel bootstrap calibration."""
    bootstrap_iter, original_data, bias_range = args
    
    bootstrap_data = bootstrap_resample(original_data, seed=bootstrap_iter)
    
    group_params = fast_group_calibration(bootstrap_data, bias_range)
    
    individual_biases_by_condition = fast_individual_calibration(bootstrap_data, bias_range)
    
    individual_stats = {}
    all_individual_biases = []
    
    for condition, biases in individual_biases_by_condition.items():
        if len(biases) > 0:
            individual_stats[f'individual_{condition}_mean'] = np.mean(biases)
            individual_stats[f'individual_{condition}_std'] = np.std(biases)
            all_individual_biases.extend(biases)
    
    individual_stats['individual_overall_mean'] = np.mean(all_individual_biases) if all_individual_biases else np.nan
    individual_stats['individual_overall_std'] = np.std(all_individual_biases) if all_individual_biases else np.nan
    
    result = {
        'bootstrap_iter': bootstrap_iter,
        'group_healthy': group_params['healthy'],
        'group_unhealthy': group_params['unhealthy'],
        'group_balanced': group_params['balanced'],
    }
    
    result.update(individual_stats)
    
    return result

def run_bootstrap_calibration(original_data, n_bootstrap=1000, bias_range=None, 
                              num_processes=None, results_dir=None):
    """Run bootstrap calibration analysis."""
    if bias_range is None:
        bias_range = [round(x * 0.1, 1) for x in range(0, 11)]
    
    if num_processes is None:
        num_processes = cpu_count()
    
    if results_dir is None:
        timestamp = strftime("%y%m%d_%H%M%S")
        results_dir = f'paper_1/results/bootstrap_calibration_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)
    
    print(f"Running bootstrap calibration with {n_bootstrap} iterations using {num_processes} processes")
    print(f"Results will be saved to: {results_dir}")
    
    args_list = [(i, original_data, bias_range) 
                 for i in range(n_bootstrap)]
    
    with Pool(processes=num_processes) as pool:
        bootstrap_results = []
        
        chunk_size = max(1, min(100, n_bootstrap // 10))
        for i in range(0, len(args_list), chunk_size):
            chunk = args_list[i:i+chunk_size]
            chunk_results = pool.map(bootstrap_calibration_worker, chunk)
            bootstrap_results.extend(chunk_results)
            
            progress = (i + len(chunk)) / len(args_list) * 100
            print(f"Progress: {progress:.1f}% ({i + len(chunk)}/{len(args_list)} iterations)")
    
    results_df = pd.DataFrame(bootstrap_results)
    summary_stats = calculate_bootstrap_summary(results_df)
    
    results_df.to_csv(f'{results_dir}/bootstrap_raw_results.csv', index=False)
    summary_stats.to_csv(f'{results_dir}/bootstrap_summary_statistics.csv', index=False)
    
    print("\n=== Bootstrap Calibration Summary ===")
    print(summary_stats)
    
    return results_df, summary_stats

def calculate_bootstrap_summary(bootstrap_results):
    """Calculate summary statistics from bootstrap results."""
    summary_data = []
    
    for condition in ['healthy', 'unhealthy', 'balanced']:
        col_name = f'group_{condition}'
        if col_name in bootstrap_results.columns:
            values = bootstrap_results[col_name].dropna()
            
            summary_data.append({
                'parameter': f'Group {condition.capitalize()}',
                'median': values.median(),
                'mean': values.mean(),
                'std': values.std(),
                'ci_2.5': values.quantile(0.025),
                'ci_97.5': values.quantile(0.975),
                'n_bootstrap': len(values)
            })
    
    for condition in ['healthy', 'unhealthy', 'balanced']:
        mean_col = f'individual_{condition}_mean'
        if mean_col in bootstrap_results.columns:
            values = bootstrap_results[mean_col].dropna()
            
            summary_data.append({
                'parameter': f'Individual {condition.capitalize()} Mean',
                'median': values.median(),
                'mean': values.mean(),
                'std': values.std(),
                'ci_2.5': values.quantile(0.025),
                'ci_97.5': values.quantile(0.975),
                'n_bootstrap': len(values)
            })
    
    if 'individual_overall_mean' in bootstrap_results.columns:
        values = bootstrap_results['individual_overall_mean'].dropna()
        summary_data.append({
            'parameter': 'Individual Overall Mean',
            'median': values.median(),
            'mean': values.mean(),
            'std': values.std(),
            'ci_2.5': values.quantile(0.025),
            'ci_97.5': values.quantile(0.975),
            'n_bootstrap': len(values)
        })
    
    return pd.DataFrame(summary_data)

def estimate_runtime_optimized(setting='moderate'):
    """Estimate runtime for bootstrap analysis."""
    config = PRACTICAL_SETTINGS[setting]
    
    seconds_per_model_run = 0.05
    
    try:
        emp_data = pd.read_csv('data/Study2_data_processed.csv', delimiter=';')
        n_participants = len(emp_data)
    except:
        n_participants = 156
    
    step = config['bias_range_step']
    n_bias_values = int(1.0/step) + 1
    
    group_runs = config['n_bootstrap'] * n_bias_values
    individual_runs = config['n_bootstrap'] * n_participants
    total_runs = group_runs + individual_runs
    
    estimated_seconds = total_runs * seconds_per_model_run
    estimated_hours = estimated_seconds / 3600
    
    print(f"Runtime Estimate for '{setting}' setting:")
    print(f"  Total model runs: {total_runs:,}")
    print(f"  Estimated time: {estimated_hours:.1f} hours")
    print(f"  With parallel processing ({cpu_count()} cores): {estimated_hours/cpu_count():.1f} hours")
    
    return estimated_hours

def run_complete_bootstrap_analysis(setting='moderate'):
    """Complete bootstrap analysis workflow."""
    emp_data = pd.read_csv('data/Study2_data_processed.csv', delimiter=';')
    emp_data = emp_data.map(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    
    config = PRACTICAL_SETTINGS[setting]
    print(f"Using '{setting}' setting: {config['description']}")
    
    step = config['bias_range_step']
    bias_range = [round(x * step, 2) for x in range(0, int(1.0/step) + 1)]
    
    estimate_runtime_optimized(setting)
        
    bootstrap_results, bootstrap_summary = run_bootstrap_calibration(
        original_data=emp_data,
        n_bootstrap=config['n_bootstrap'],
        bias_range=bias_range
    )
    
    return bootstrap_results, bootstrap_summary

if __name__ == "__main__":
    print("Bootstrap Calibration Analysis")
    print("=" * 45)
    
    print("\nRuntime estimates:")
    for setting in ['development', 'moderate', 'full']:
        estimate_runtime_optimized(setting)
        print()
    
    print("Starting analysis...")
    results = run_complete_bootstrap_analysis(setting='full')