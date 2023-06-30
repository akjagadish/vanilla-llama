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

def parse_and_pool_generated_tasks(path, gpt, models, dims, data, tasks, runs, proc_ids):
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
    Returns:
        df: dataframe containing the pooled generated tasks
    '''

    df = None
    total_tasks = 0
    for model in models:
        for dim in dims:
            for num_data in data:
                last_task_id=0
                for num_tasks in tasks:
                    for run in runs:
                        for proc_id in proc_ids[num_tasks]:
                            total_tasks += num_tasks
                            
                            filename = f'{gpt}_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}'
                            #if os.path.exists(f"{path}/{filename}.csv"): 
                            last_task_id = parse_generated_tasks(path, filename, gpt, num_data, last_task_id)
                            print(f'parsed: {filename}')
                           
                            # load llama generated tasks which were successfully regex parsed
                            df = return_generated_task(path, gpt, model, dim, num_data, num_tasks, run, proc_id) if df is None else pd.concat([df, \
                                    return_generated_task(path, gpt, model, dim, num_data, num_tasks, run, proc_id)], ignore_index=True)
                            print(f'pooled: {filename}')
                # save the pooled dataframe to csv
                print(df)
                df = df.query('target == "A" or target == "B"')
                #df['task_id'] = np.int64(np.arange(len(df))/num_data) #+ 1 
                df.to_csv(f"{path}/{gpt}_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{total_tasks}.csv")
    
    return df


if __name__ == '__main__':

    path ='/raven/u/ajagadish/vanilla-llama/categorisation/data'
    gpt = 'gpt3' #'llama',
    models = ['NA']#['65B']
    dims = [3]
    num_data_points = [100]
    tasks = [100] #1000, 2000, 500, 1500]
    runs = [0] #{1000: 0, 2000: 0}
    proc_ids = {10: [999, 998, 997], 5: [999], 100: [0]} #1000: range(0, 8), 2000: range(0,2), 500: range(0,2), 1500: range(0,1)} #format is {num_tasks: proc_ids}
    
    data = parse_and_pool_generated_tasks(path, gpt, models, dims, num_data_points, tasks, runs, proc_ids)


# for model in models:
#     for dims in dims:
#         for num_data in num_data_points:
#             for task in tasks:
#                 for run in runs:
#                     for proc_id in proc_ids[task]:
#                         parse_generated_tasks(path,\
#                                             f'llama_generated_tasks_params{model}_dim{dims}_data{num_data}_tasks{task}_run{run}_procid{proc_id}',\
#                                             num_data_points)
# file_name='llama_generated_tasks_65B_3'
# # load llama generated tasks which were successfully regex parsed
# with open(f"data/{file_name}.txt", "rb") as fp:   
#     datasets = pickle.load(fp)

# # regular expression pattern to extract input values from the stored inputs and targets
# pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'
# # ((?:\s|\[)?\s*[\d.]+) (?:,|;|\s) ?\s*([\d.]+) (?:,|;|\s) ?\s*([\d.]+)'

# # make a pandas dataframe for the parsed data
# df = df.read_csv('data/{file_name}.csv') if else None
# task_id=1
# # parse the list using regex
# for task, data in enumerate(datasets):
#     # initialize lists to store parsed values
#     inputs = []
#     targets = []
#     for item in data:
#         match = re.match(pattern, item[0])
#         if match:
#             try:
#                 inputs.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
#                 targets.append(item[1])
#             except:
#                 print(f'error parsing {task, item[0]}')
#         else:
#             print(f'error parsing {task, item[0]}')
#     if len(inputs) == 8:
#         # create a DataFrame from inputs and targets
#         df = pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)}) if df is None else pd.concat([df, \
#         pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)})], ignore_index=True)
#         task_id+=1

# # print the DataFrame
# print(df)
# #save data frame to csv
# df.to_csv('data/llama_generated_tasks_65B_3.csv')