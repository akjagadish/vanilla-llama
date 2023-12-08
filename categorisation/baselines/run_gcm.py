import pandas as pd
from gcm import GeneralizedContextModel
import sys
import numpy as np
import argparse
sys.path.append('../')

## badham et al.
# df = pd.read_csv('../data/human/exp1.csv')
# gcm = GeneralizedContextModel(num_features=3, distance_measure=1, num_iterations=1)
# gcm.burn_in = True
# ll, r2 = gcm.fit_participants(df)
# print(ll, r2)
# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

## speekenbrink et al. 
# df = pd.read_csv('../data/human/exp2.csv')
# gcm = GeneralizedContextModel(num_features=4, distance_measure=1, num_iterations=1)
# gcm.burn_in = True
# ll, r2 = gcm.fit_participants(df)
# print(ll, r2)
# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

## devraj et al. 2022
# df = pd.read_csv('../data/human/devraj2022rational.csv')
# df = df[df['condition'] == 'control'] # only pass 'control' condition
# ##### REMOVE THIS LINE AFTER TESTING #####
# # df = df[df['participant']==0] # keep only first participant for testing
# ##### REMOVE ABOVE LINE AFTER TESTING #####
# num_runs, num_blocks, num_iter = 1, 11, 10
# loss = 'mse_transfer'
# opt_method = 'minimize'
# NUM_TASKS, NUM_FEATURES = 1, 6
# lls, r2s, params_list = [], [], []
# for idx in range(num_runs):
#     gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, opt_method=opt_method, loss=loss)
#     ll, r2, params = gcm.fit_participants(df, num_blocks=num_blocks, reduce='sum')
#     params_list.append(params)
#     lls.append(ll)
#     r2s.append(r2)
#     print(lls[idx], r2s[idx])
#     print(f'mean mse across blocks: {lls[idx].mean()} \n')
#     print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

# # save the r2 and ll values
# lls = np.array(lls)
# r2s = np.array(r2s)
# np.savez(f'../data/meta_learner/gcm_humans_devrajstask_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}'\
#          , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)
 
## benchmarking gcm model
# df_train = pd.read_csv('../data/human/akshay-benchmark-across-languages-train.csv')
# df_transfer = pd.read_csv('../data/human/akshay-benchmark-across-languages-transfer.csv')
# gcm = GeneralizedContextModel(num_features=2, distance_measure=1, num_iterations=1)
# params = gcm.benchmark(df_train, df_transfer)
# print('fitted parameters: c {}, bias {}, w1 {}, w2 {}'.format(*params))
# true_params = pd.read_csv('../data/human/akshay-benchmark-across-languages-params.csv')
# print('true parameters: ', true_params)

## fit gcm model to meta-learning model choices
# shpeards task
# df = pd.read_csv('../data/meta_learner/shepard_categorisation_env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0_beta=0.3_num_trials=96_num_runs=1.csv') 
# gcm = GeneralizedContextModel(num_features=3, distance_measure=1, num_iterations=1)
# # gcm.burn_in = True
# ll, r2 = gcm.fit_metalearner(df, num_blocks=6, reduce='sum')
# print(ll, r2)
# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')

# smiths task
# df = pd.read_csv('../data/meta_learner/smithstask_env=claude_generated_tasks_paramsNA_dim6_data500_tasks12911_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta=0.3_num_trials=300_num_runs=1.csv')
# opt_method = 'minimize' #'minimize' or 'differential_evolution
# num_runs, num_blocks, NUM_TASKS = 5, 6, 2
# lls, r2s = np.zeros((num_runs, NUM_TASKS, num_blocks)), np.zeros((num_runs, NUM_TASKS, num_blocks))
# params_list = []
# for idx in range(num_runs):
#     gcm = GeneralizedContextModel(num_features=6, distance_measure=1, num_iterations=1, opt_method=opt_method)
#     lls[idx], r2s[idx], params = gcm.fit_metalearner(df, num_blocks=num_blocks, reduce='sum')
#     params_list.append(params)
#     print(lls[idx], r2s[idx])
#     print(f'mean log-likelihood across blocks: {lls[idx].mean()} \n')
#     print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

# # save the r2 and ll values
# np.savez(f'../data/meta_learner/gcm_simulations_smithstask_runs={num_runs}_blocks={num_blocks}_tasks={NUM_TASKS}'\
#          , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)



def fit_gcm_to_metalearner(beta, num_runs=1, num_blocks=11, num_iter=1, num_tasks=1, num_features=6, opt_method='minimize', loss='mse_transfer'):
    # beta=0.1
    df = pd.read_csv(f'../data/meta_learner/smith_categorisation__tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1_beta={beta}_num_trials=616_num_runs=10.csv')
    # num_runs, num_blocks, num_iter = 1, 11, 1
    # loss = 'mse_transfer'
    # opt_method = 'minimize'
    NUM_TASKS, NUM_FEATURES = 1, 6
    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        gcm = GeneralizedContextModel(num_features=NUM_FEATURES, distance_measure=1, num_iterations=num_iter, opt_method=opt_method, loss=loss)
        ll, r2, params = gcm.fit_metalearner(df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(lls[idx], r2s[idx])
        print(f'mean mse across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'../data/meta_learner/gcm_metalearner_devrajstask_beta={beta}_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_tasks={NUM_TASKS}'\
            , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit gcm to meta-learner choices')
    parser.add_argument('--beta', type=float, required=True, help='beta value')
    parser.add_argument('--num-iter', type=int, required=True, default=1, help='number of iterations')
    parser.add_argument('--num-runs', type=int, required=False,  default=11, help='number of runs')
    parser.add_argument('--num-blocks', type=int,required=False, default=11, help='number of blocks')
    parser.add_argument('--num-tasks', type=int, required=False, default=1, help='number of tasks')
    parser.add_argument('--num-features', type=int, required=False, default=6, help='number of features')
    parser.add_argument('--opt-method', type=str, required=False, default='minimize', help='optimization method')
    parser.add_argument('--loss', type=str, required=False, default='mse_transfer', help='loss function')
    args = parser.parse_args()

    fit_gcm_to_metalearner(beta=args.beta, num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter, num_tasks=args.num_tasks, num_features=args.num_features, opt_method=args.opt_method, loss=args.loss)
