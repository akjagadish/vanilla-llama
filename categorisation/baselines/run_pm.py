import pandas as pd
from pm import PrototypeModel
import numpy as np

# df = pd.read_csv('exp1.csv')
# pm = PrototypeModel(num_features=3, distance_measure=1, num_iterations=1)
# ll, r2 = pm.fit_participants(df)

# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')


## benchmarking pm model
# df_train = pd.read_csv('../data/human/akshay-benchmark-across-languages-train.csv')
# df_transfer = pd.read_csv('../data/human/akshay-benchmark-across-languages-transfer.csv')
# pm = PrototypeModel(num_features=2, distance_measure=1, num_iterations=1, prototypes=[[4.766967, 2.243946],[2.292030, 4.593165]])
# params = pm.benchmark(df_train, df_transfer)
# print('fitted parameters: c {}, bias {}, w1 {}, w2 {}'.format(*params))
# true_params = pd.read_csv('../data/human/akshay-benchmark-across-languages-params.csv')
# print('true parameters: ', true_params)

## fit gcm model to meta-learning model choices
# shpeards task
# df = pd.read_csv('../data/meta_learner/shepard_categorisation_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_beta=0.3_num_trials=96_num_runs=1.csv') 
# pm =  PrototypeModel(num_features=3, distance_measure=1, num_iterations=1, learn_prototypes=True, prototypes=None)# prototypes='from_data'
# # pm.burn_in = True
# ll, r2 = pm.fit_metalearner(df, num_blocks=6)
# print(ll, r2)
# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

# smiths task
df = pd.read_csv('../data/meta_learner/smithstask_env=claude_generated_tasks_paramsNA_dim6_data500_tasks12911_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta=0.3_num_trials=300_num_runs=1.csv')
opt_method = 'minimize'
num_runs, num_blocks, NUM_TASKS = 5, 6, 2
lls, r2s = np.zeros((num_runs, NUM_TASKS, num_blocks)), np.zeros((num_runs, NUM_TASKS, num_blocks))
params_list = []
for idx in range(num_runs):
    pm =  PrototypeModel(num_features=6, distance_measure=1, num_iterations=1, learn_prototypes=False, prototypes='from_data')# prototypes='from_data'
    lls[idx], r2s[idx], params = pm.fit_metalearner(df, num_blocks=num_blocks)
    params_list.append(params)
    print(lls[idx], r2s[idx])
    print(f'mean log-likelihood across blocks: {lls[idx].mean()} \n')
    print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

# save the r2 and ll values
np.savez(f'../data/meta_learner/pm_smithstask_runs={num_runs}_blocks={num_blocks}_tasks={NUM_TASKS}'\
         , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)