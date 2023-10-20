import pickle
import re
import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')
from utils import parse_generated_tasks, return_generated_task
import ipdb
import argparse

def parse_and_pool_generated_tasks(path, gpt, models, dims, data, tasks, runs, proc_ids, prompt_version, use_generated_tasklabels):
    ''' 
    Parse (if not parsed) and pool the generated tasks from LLMs into a single dataframe
    Args:
        path: path to the folder containing the generated tasks
        gpt: gpt model used to generate the tasks
        models: list of models used to generate the tasks
        dims: list of dimensions used to generate the tasks
        data: list of number of datapoints used to generate the tasks
        tasks: list of number of tasks used to generate the tasks
        runs: list of runs used to generate the tasks
        proc_ids: dictionary of process ids used to generate the tasks
        prompt_version: version of the prompt used to generate the tasks
    Returns:
        df: dataframe containing the pooled generated tasks
    '''

    df = None
    for model in models:
        for dim in dims:
            for num_data in data:
                last_task_id=0
                for num_tasks in tasks:
                    for run in runs:
                        for proc_id in proc_ids[num_tasks]:
                            
                            filename = f'{gpt}_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
                            #if os.path.exists(f"{path}/{filename}.csv"): 
                            last_task_id = parse_generated_tasks(path+'/parsed', filename, gpt, num_data, last_task_id, use_generated_tasklabels, prompt_version)
                            print(f'parsed: {filename}')
                           
                            # load llm generated tasks which were successfully regex parsed
                            df = return_generated_task(path+'/parsed', gpt, model, dim, num_data, num_tasks, run, proc_id, prompt_version) if df is None else pd.concat([df, \
                                    return_generated_task(path+'/parsed', gpt, model, dim, num_data, num_tasks, run, proc_id, prompt_version)], ignore_index=True)
                            print(f'pooled: {filename}')
                
                # save the pooled dataframe to csv
                df = df if use_generated_tasklabels else df.query('target == "A" or target == "B"')
                total_tasks = df.task_id.max()+1
                df.to_csv(f"{path}/{gpt}_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{int(total_tasks)}_pversion{prompt_version}.csv")
                print(f'saved: {gpt}_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{int(total_tasks)}_pversion{prompt_version}.csv')
    return df


if __name__ == '__main__':
    ## take arguments from command line and arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/raven/u/ajagadish/vanilla-llama/categorisation/data', help='path to the folder containing the generated tasks')
    parser.add_argument('--gpt', type=str, default='claude', help='gpt model used to generate the tasks')
    parser.add_argument('--models', nargs='+', default=['NA'], help='versions of gpt model used to generate the tasks')
    parser.add_argument('--dims', nargs='+', type=int,  default=[3], help='list of dimensions used to generate the tasks')
    parser.add_argument('--num-data', nargs='+', type=int,  default=[100], help='list of number of datapoints used to generate the tasks')
    parser.add_argument('--tasks', type=int, default=2000, help='list of number of tasks used to generate the tasks')
    parser.add_argument('--runs', nargs='+', type=int, default=[0], help='list of runs used to generate the tasks')
    parser.add_argument('--proc_ids', nargs='+', type=int, default=[0], help='dictionary of process ids used to generate the tasks')
    parser.add_argument('--prompt_version', type=int, default=4, help='version of the prompt used to generate the tasks')
    parser.add_argument('--use_generated_tasklabels', action='store_true', help='whether to use the task labels generated by the LLMs or the ground truth labels')
    
    args = parser.parse_args()
    path = args.path
    gpt = args.gpt
    models = args.models
    dims = args.dims
    num_data_points = args.num_data
    tasks = args.tasks
    runs = args.runs
    proc_ids = {tasks: args.proc_ids}
    prompt_version = args.prompt_version
    use_generated_tasklabels = args.use_generated_tasklabels

    data = parse_and_pool_generated_tasks(path, gpt, models, dims, num_data_points, [tasks], runs, proc_ids, prompt_version, use_generated_tasklabels)