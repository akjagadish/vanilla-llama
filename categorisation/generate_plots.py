import numpy as np
import torch
import pandas as pd
import sys
SYS_PATH = '/u/ajagadish/vanilla-llama' #'/raven/u/ajagadish/vanilla-llama/'
sys.path.append(f'{SYS_PATH}/categorisation/')
sys.path.append(f'{SYS_PATH}/categorisation/data')
sys.path.append(f'{SYS_PATH}/categorisation/rl2')


# #### Data Statistics
from plots import label_imbalance, plot_mean_number_tasks, plot_data_stats, plot_trial_by_trial_performance, plot_burstiness_training_curriculum

## load and filter data
#dim4: 'claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1'
#dim6: 'claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2'
#dim3: 'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4'
# env_name='claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4'
# data = pd.read_csv(f'{SYS_PATH}/categorisation/data/{env_name}.csv') 
# data = data.groupby(['task_id']).filter(lambda x: len(x['target'].unique()) == 2) # check if data has only two values for target in each task
# data.input = data['input'].apply(lambda x: np.array(eval(x)))

# ## analyse llm generated data
# # label_imbalance(data)
# # plot_mean_number_tasks(data)
# # plot_burstiness_training_curriculum(data)#, num_tasks=100
# # min_trials, burn_in = 50, 1
# # df = data.groupby('task_id').filter(lambda x: len(x)>=min_trials)
# # data = data[data.trial_id<=min_trials] # keep only min_trials for all tasks for model fitting
# # plot_trial_by_trial_performance(df, burn_in, min_trials-burn_in, min_trials)

# plot_data_stats(data, poly_degree=2) #TODO: extend this function to work for data with more than 3 dims

#--------------------------- 
from plots import replot_nosofsky1988, replot_nosofsky1994, replot_levering2020

## replot experimental plots
# replot_nosofsky1988()
# replot_nosofsky1994()
# replot_levering2020()

#---------------------------
from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994, evaluate_smith1998

# ## meta-learner trained on original llm data
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# beta=0.3
# evaluate_nosofsky1988(env_name=env_name, experiment=2, beta=beta, noises=[0.0], shuffles=[False], num_runs=50, num_blocks=1, num_eval_tasks=64)
# evaluate_levering2020(env_name=env_name, noises=[0.0],  beta=beta, shuffles=[False], num_runs=50, num_eval_tasks=64, num_trials=150)
# evaluate_nosofsky1994(env_name=env_name, tasks=np.arange(1,7),  beta=beta, noises=[0.0], shuffles=[False], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64)

## meta-leaner trained on shuffled llm data
# env_model_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# ## 'claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# ## 'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# dim = int(env_model_name.split('dim')[1].split('_')[0]) 
# beta = 0.1856 if dim==3 else 0.095 #0.3
# if dim==3:
#     run, num_runs = 1, 50 
#     num_trials_per_block = 16
#     num_blocks = 10
#     evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), beta=beta, noises=[0.0], shuffles=[True], shuffle_evals=[False],\
#                            experiment='shepard_categorisation', num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks,\
#                              run=run, num_eval_tasks=64)
# elif dim==6:
#  run, num_runs = 1, 50
#  num_trials = 616
#  num_blocks = 11
#  evaluate_smith1998(env_name=env_model_name, noises=[0.0], beta=beta, shuffles=[True], num_runs=num_runs, num_eval_tasks=64, num_trials=num_trials, num_blocks=num_blocks, run=run)
# evaluate_nosofsky1988(env_name=env_model_name, experiment=2, beta=beta, noises=[0.0], shuffles=[True], num_runs=50, num_blocks=1, num_eval_tasks=64)
# evaluate_levering2020(env_name=env_model_name, noises=[0.0], beta=beta,  shuffles=[True], num_runs=50, num_eval_tasks=64, num_trials=150)

#---------------------------
# ## meta-leaner trained on synthetic data (note that for synthetic nonlinear shuffles is set to True)
# env_model_name = 'dim6synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# #'transformer_metalearning_synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=128_num_head=8'
# #'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# #'transformer_metalearning_synthetic_model=transformer_num_episodes500000_num_hidden=128_lr0.0003_num_layers=6_d_model=64_num_head=4
# nonlinear, run, beta, num_blocks, num_trials_per_block, num_runs = True, 0, 0.3, 10, 16, 50
# shuffle = True if nonlinear else False
# dim = 6
# # evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), beta=beta, noises=[0.0], shuffles=[False], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks, num_eval_tasks=64, synthetic=True, nonlinear=nonlinear, run=run)
# # evaluate_nosofsky1988(env_name=env_model_name, experiment=2, beta=beta, noises=[0.0], shuffles=[True], num_runs=50, num_blocks=1, num_eval_tasks=64, synthetic=True, nonlinear=nonlinear,run=run)
# # evaluate_levering2020(env_name=env_model_name, noises=[0.0], beta=beta, shuffles=[True], num_runs=50, num_eval_tasks=64, num_trials=150, synthetic=True, nonlinear=nonlinear, run=run)
# if dim==6:
#     num_trials = 616
#     num_blocks = 11
#     evaluate_smith1998(env_name=env_model_name, noises=[0.0], beta=beta, shuffles=[True], num_runs=num_runs, num_eval_tasks=64, num_trials=num_trials, num_blocks=num_blocks, synthetic=True, nonlinear=nonlinear, run=run)

## RMC model simulations
# env_model_name = 'env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8s'
# dim = 3 
# beta = 0.6340 # mean fitted values from Badham et al. 2017
# run, num_runs = 1, 50 
# num_trials_per_block = 16
# num_blocks = 16
# shuffle = True 
# rmc = True
# evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), beta=beta, noises=[0.0], shuffles=[shuffle], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks, num_eval_tasks=64, rmc=rmc, run=run)

#---------------------------
from plots import compare_metalearners
## compare meta-learners for different noises and shuffles
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000'
# model_env = 'num_episodes500000_num_hidden=128_lr0.0003'
# compare_metalearners(env_name, model_env, noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[False], num_runs=10)

#---------------------------
from plots import plot_data_stats_synthetic

# ##synthetic dim3: 'synthetic_tasks_dim3_data100_tasks5000_nonlinearFalse'
# ##synthetic dim3: 'synthetic_tasks_dim3_data100_tasks5000_nonlinearTrue'
# ##rmc: 'rmc_tasks_dim3_data100_tasks5000'
# env_name='rmc_tasks_dim3_data100_tasks5000'
# data = pd.read_csv(f'{SYS_PATH}/categorisation/data/{env_name}.csv') 
# data = data.groupby(['task_id']).filter(lambda x: len(x['target'].unique()) == 2) # check if data has only two values for target in each task
# data.input = data['input'].apply(lambda x: np.array(eval(x)))
# synthetic_type = 'rmc' if 'rmc' in env_name else 'nonlinear' if env_name.split('nonlinear')[1]=='True' else 'linear'
# dim = int(env_name.split('dim')[1].split('_')[0])

# plot_data_stats_synthetic(data, poly_degree=2, synthetic_type=synthetic_type, dim=dim)
# min_trials, burn_in = 90, 1
# df = data.groupby('task_id').filter(lambda x: len(x)>=min_trials)
# data = data[data.trial_id<=min_trials] # keep only min_trials for all tasks for model fitting
# plot_trial_by_trial_performance(df, burn_in, min_trials-burn_in, min_trials)

#---------------------------
from plots import model_comparison_badham2017, gcm_pm_fitted_simulations, model_comparison_devraj2022, simulate_shepard1961

model_comparison_badham2017()
# gcm_pm_fitted_simulations()
model_comparison_devraj2022()
models = ['humans',\
          'env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=3',\
          'env=dim3synthetic_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_synthetic',\
          #'env=rmc_tasks_dim3_data100_tasks11499_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_rmc',
          ]
betas = [None, 0.1503, 0.1314] #ermi: 0.1856 (run1); ermi: 0.150285 (run3) rmc: 0.6340 mi: 0.1314
num_trials_per_block = 16
num_blocks = 16
num_runs = 50
simulate_shepard1961(models=models, betas=betas, num_runs=num_runs, num_blocks=num_blocks, num_trials=num_trials_per_block*num_blocks)