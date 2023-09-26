import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/rl2')
import numpy as np
import torch
import pandas as pd

#-----------------------------------------------------------
## analyse llm generated data
env_name='claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1'
data = pd.read_csv(f'/raven/u/ajagadish/vanilla-llama/categorisation/data/{env_name}.csv') #_pversion1
data = data.query('target == "A" or target == "B"')

from plots import label_imbalance
label_imbalance(data)

from plots import plot_mean_number_tasks
plot_mean_number_tasks(data)

from plots import plot_data_stats
plot_data_stats(data)

#-----------------------------------------------------------

## replot experimental plots

from plots import replot_nosofsky1988, replot_nosofsky1994, replot_levering2020
replot_nosofsky1988()
replot_nosofsky1994()
replot_levering2020()

#------------------------------------------------------------

## meta-learner trained on original llm data
from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000'
evaluate_nosofsky1988(env_name=env_name, experiment=2, noises=[0.0], shuffles=[False], num_runs=50, num_blocks=1, num_eval_tasks=64)
evaluate_levering2020(env_name=env_name, noises=[0.0], shuffles=[False], num_runs=50, num_eval_tasks=64, num_trials=150)
evaluate_nosofsky1994(env_name=env_name, tasks=np.arange(1,7), noises=[0.0], shuffles=[False], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64)

#------------------------------------------------------------

## meta-leaner trained on shuffled llm data
from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000'
evaluate_nosofsky1988(env_name=env_name, experiment=2, noises=[0.0], shuffles=[True], num_runs=50, num_blocks=1, num_eval_tasks=64)
evaluate_levering2020(env_name=env_name, noises=[0.0], shuffles=[True], num_runs=50, num_eval_tasks=64, num_trials=150)
evaluate_nosofsky1994(env_name=env_name, tasks=np.arange(1,7), noises=[0.0], shuffles=[True], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64)

#------------------------------------------------------------

## meta-leaner trained on synthetic data
from plots import evaluate_nosofsky1988, evaluate_levering2020, evaluate_nosofsky1994
# env_name = 'claude_generated_tasks_paramsNA_dim3_data100_tasks14000'
evaluate_nosofsky1988(env_name=env_name, experiment=2, noises=[0.0], shuffles=[True], num_runs=50, num_blocks=1, num_eval_tasks=64, synthetic=True)
evaluate_levering2020(env_name=env_name, noises=[0.0], shuffles=[True], num_runs=50, num_eval_tasks=64, num_trials=150, synthetic=True)
evaluate_nosofsky1994(env_name=env_name, tasks=np.arange(1,7), noises=[0.0], shuffles=[True], shuffle_evals=[False], experiment='shepard_categorisation', num_runs=50, num_eval_tasks=64, synthetic=True)

#------------------------------------------------------------
