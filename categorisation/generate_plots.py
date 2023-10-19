import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
import numpy as np
import torch
import pandas as pd


# load and filter data
env_name='claude_generated_tasks_paramsNA_dim3_data100_tasks4000_pversion4' ##_pversion1
data = pd.read_csv(f'/raven/u/ajagadish/vanilla-llama/categorisation/data/{env_name}.csv') 
data = data.groupby(['task_id']).filter(lambda x: len(x['target'].unique()) == 2) # check if data has only two values for target in each task
data.input = data['input'].apply(lambda x: np.array(eval(x)))

## analyse llm generated data
from plots import label_imbalance, plot_mean_number_tasks, plot_data_stats, plot_trial_by_trial_performance, plot_burstiness_training_curriculum
# label_imbalance(data)
# plot_mean_number_tasks(data)
plot_data_stats(data, poly_degree=2)
# plot_burstiness_training_curriculum(data, num_tasks=100)

# plot trial by trial performance
# min_trials, burn_in = 50, 1
# df = data.groupby('task_id').filter(lambda x: len(x)>=min_trials)
# data = data[data.trial_id<=min_trials] # keep only min_trials for all tasks for model fitting
# plot_trial_by_trial_performance(df, burn_in, min_trials-burn_in, min_trials)

## replot experimental plots
# from plots import replot_nosofsky1988, replot_nosofsky1994, replot_levering2020
# replot_nosofsky1988()
# replot_nosofsky1994()
# replot_levering2020()

## meta-learner trained on original llm data
# from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# # env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000_num_episodes500000_num_hidden=128_lr0.0003'
# evaluate_nosofsky1988(env_name=env_name, experiment=2, noises=[0.0], shuffles=[False], num_runs=50, num_blocks=1, num_eval_tasks=64)
# evaluate_levering2020(env_name=env_name, noises=[0.0], shuffles=[False], num_runs=50, num_eval_tasks=64, num_trials=150)
# evaluate_nosofsky1994(env_name=env_name, tasks=np.arange(1,7), noises=[0.0], shuffles=[False], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64)

## meta-leaner trained on shuffled llm data
# from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# env_model_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
# evaluate_nosofsky1988(env_name=env_model_name, experiment=2, noises=[0.0], shuffles=[True], num_runs=50, num_blocks=1, num_eval_tasks=64)
# evaluate_levering2020(env_name=env_model_name, noises=[0.0], shuffles=[True], num_runs=50, num_eval_tasks=64, num_trials=150)
# evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), noises=[0.0], shuffles=[True], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=10, num_eval_tasks=64)

## meta-leaner trained on synthetic data
# from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# env_model_name = 'transformer_metalearning_synthetic_model=transformer_num_episodes500000_num_hidden=128_lr0.0003_num_layers=6_d_model=64_num_head=4'
# evaluate_nosofsky1988(env_name=env_model_name, experiment=2, beta=1., noises=[0.0], shuffles=[False], num_runs=50, num_blocks=1, num_eval_tasks=64, synthetic=True)
# evaluate_levering2020(env_name=env_model_name, noises=[0.0], beta=1., shuffles=[False], num_runs=50, num_eval_tasks=64, num_trials=150, synthetic=True)
# evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), beta=2., noises=[0.0], shuffles=[False], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64, synthetic=True)

## compare meta-learners for different noises and shuffles
# from plots import compare_metalearners
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000'
# model_env = 'num_episodes500000_num_hidden=128_lr0.0003'
# compare_metalearners(env_name, model_env, noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[False], num_runs=10)