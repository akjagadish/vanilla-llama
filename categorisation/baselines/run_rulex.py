import pandas as pd
from rulex import RulExModel
import sys
import numpy as np
import argparse
import sys
SYS_PATH = '/u/ajagadish/vanilla-llama' #'/raven/u/ajagadish/vanilla-llama/'
sys.path.append(f'{SYS_PATH}/categorisation/data')

def fit_rulex_to_humans(num_runs, num_blocks, num_iter, num_tasks, num_features, opt_method, loss, exception):
    #TODO: in devraj every participant does only one condition so need to select only one condition
    df = pd.read_csv('../data/human/devraj2022rational.csv')
    df = df[df['condition'] == 'control'] # only pass 'control' condition
    # num_runs, num_blocks, num_iter = 1, 11, 10
    # loss = 'mse_transfer'
    # opt_method = 'minimize'
    exceptions = [[1, 1, 1 , 1, 0, 1], [0, 0, 0, 1, 0, 0]]
    NUM_TASKS, NUM_FEATURES = 1, 6
    lls, r2s, params_list = [], [], []
    for idx in range(num_runs):
        rulex = RulExModel(num_features=NUM_FEATURES, opt_method=opt_method, loss=loss, exception=exception, exceptions=exceptions)
        ll, r2, params = rulex.fit_participants(df, num_blocks=num_blocks, reduce='sum')
        params_list.append(params)
        lls.append(ll)
        r2s.append(r2)
        print(lls[idx], r2s[idx])
        print(f'mean mse across blocks: {lls[idx].mean()} \n')
        print(f'mean pseudo-r2 across blocks: {r2s[idx].mean()}')

    # save the r2 and ll values
    lls = np.array(lls)
    r2s = np.array(r2s)
    np.savez(f'{SYS_PATH}/categorisation/data/model_comparison/devraj2022_rulex_runs={num_runs}_iters={num_iter}_blocks={num_blocks}_loss={loss}_exception={exception}'\
             , r2s=r2s, lls=lls, params=np.stack(params_list), opt_method=opt_method)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit rulex to meta-learner or human choices')
    parser.add_argument('--exception', action='store_true', help='use exception term')
    parser.add_argument('--num-iter', type=int, required=True, default=1, help='number of iterations')
    parser.add_argument('--num-runs', type=int, required=False,  default=1, help='number of runs')
    parser.add_argument('--num-blocks', type=int,required=False, default=11, help='number of blocks')
    parser.add_argument('--num-tasks', type=int, required=False, default=1, help='number of tasks')
    parser.add_argument('--num-features', type=int, required=False, default=6, help='number of features')
    parser.add_argument('--opt-method', type=str, required=False, default='minimize', help='optimization method')
    parser.add_argument('--loss', type=str, required=False, default='mse_transfer', help='loss function')
    parser.add_argument('--fit-human-data', action='store_true', help='fit gcm to human choices')
    args = parser.parse_args()

    if args.fit_human_data:
        fit_rulex_to_humans(num_runs=args.num_runs, num_blocks=args.num_blocks, num_iter=args.num_iter,\
                           num_tasks=args.num_tasks, num_features=args.num_features, \
                              opt_method=args.opt_method, loss=args.loss, exception=args.exception)
    else:   
        raise NotImplementedError