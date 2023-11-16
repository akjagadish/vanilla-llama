import pandas as pd
from pm import PrototypeModel


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
df = pd.read_csv('../data/meta_learner/shepard_categorisation_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_beta=0.3_num_trials=96_num_runs=1.csv') 
pm =  PrototypeModel(num_features=3, distance_measure=1, num_iterations=1,learn_prototypes=True)# prototypes='from_data'
pm.burn_in = True
ll, r2 = pm.fit_metalearner(df)
print(ll, r2)
print(f'mean log-likelihood across participants: {ll.mean()} \n')
print(f'mean pseudo-r2 across participants: {r2.mean()}')