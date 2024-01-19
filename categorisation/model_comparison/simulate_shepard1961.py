import numpy as np
import torch
import pandas as pd
import sys
SYS_PATH = '/u/ajagadish/vanilla-llama' #'/raven/u/ajagadish/vanilla-llama/'
sys.path.append(f'{SYS_PATH}/categorisation/')
sys.path.append(f'{SYS_PATH}/categorisation/data')
sys.path.append(f'{SYS_PATH}/categorisation/rl2')
from plots import simulate_shepard1961

def compute_distance_between_model_and_humans(models):
    num_blocks = 15 # 16
    num_trials_per_block = 16
    num_runs = 1
    min_mse_distance = np.inf
    mse_distances = []
    beta_range = np.arange(0., 1.0, 0.1)
    for beta in beta_range:
        betas = [None, beta]
        mse_distance_beta = simulate_shepard1961(models=models, betas=betas, num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks)
        if mse_distance_beta[1] <= min_mse_distance:
            min_mse_distance = mse_distance_beta[1]
            best_beta = beta
        mse_distances.append(mse_distance_beta[1])

    print(f'best beta: {best_beta}, min_mse_distance: {min_mse_distance}')
    # save results
    model_name = 'ermi' if 'claude' in models[1] else 'rmc' if 'rmc' in models[1] else 'pfn' if 'syntheticnonlinear' in models[1] else 'mi'
    np.save(f'{SYS_PATH}/categorisation/data/fitted_simulation/shepard1961_{model_name}_num_runs={num_runs}_num_blocks={num_blocks}_num_trials_per_block={num_trials_per_block}.npy', [np.array(mse_distances), beta_range])

if __name__ == '__main__':

    models = ['humans',\
          'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic',\
          ]
    compute_distance_between_model_and_humans(models)

    models = ['humans',\
          'env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0',
          ]
    compute_distance_between_model_and_humans(models)

    models = ['humans',\
          'env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc',
          ]
    compute_distance_between_model_and_humans(models)

    models = ['humans',\
          'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_syntheticnonlinear',\
          ]
    compute_distance_between_model_and_humans(models)