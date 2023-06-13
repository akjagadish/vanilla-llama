import pickle
import re
import pandas as pd
import numpy as np
import sys
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/')
sys.path.append('/raven/u/ajagadish/vanilla-llama/categorisation/data')

def parse_generated_tasks(path, file_name, num_datapoints=8):
   
    # load llama generated tasks which were successfully regex parsed
    with open(f"{path}/{file_name}.txt", "rb") as fp:   
        datasets = pickle.load(fp)

    # regular expression pattern to extract input values from the stored inputs and targets
    pattern = r'((?: )?\s*[\d.]+)(?:,|;|\s)?\s*([\d.]+)(?:,|;|\s)?\s*([\d.]+)'

    # make a pandas dataframe for the parsed data
    df = None 
    task_id = 1

    # parse the list using regex
    for task, data in enumerate(datasets):
        # initialize lists to store parsed values
        inputs, targets = [], []

        # load each input-target pair
        for item in data:
            match = re.match(pattern, item[0])
            if match:
                try:
                    inputs.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
                    targets.append(item[1])
                except:
                    print(f'no match')
            else:
                print(f'error parsing {task, item[0]}')

        # if the number of datapoints is equal to the number of inputs, add to dataframe
        if len(inputs) == num_datapoints:
            df = pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)}) if df is None else pd.concat([df, \
                 pd.DataFrame({'input': inputs, 'target': targets, 'task_id': np.ones((len(inputs),))*(task_id)})], ignore_index=True)
            task_id+=1

    # save data frame to csv
    df.to_csv(f'{path}/{file_name}.csv')

def return_generated_task(path, model, num_dim, num_data, num_tasks, run, proc_id):
    return pd.read_csv(f"{path}/temp/llama_generated_tasks_params{model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}.csv")

def pool_generated_tasks(path, models, dims, data, tasks, runs, proc_ids):
    ''' 
    Pool the generated tasks from different processes into a single dataframe
    Args:
        path: path to the folder containing the generated tasks
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
                for num_tasks in tasks:
                    for run in runs:
                        for proc_id in proc_ids[num_tasks]:
                            total_tasks += num_tasks
                            df = return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id) if df is None else pd.concat([df, \
                                return_generated_task(path, model, dim, num_data, num_tasks, run, proc_id)], ignore_index=True)
                # save the pooled dataframe to csv
                df = df.query('target == "A" or target == "B"')
                df['task_id'] = np.int64(np.arange(len(df))/num_data) + 1 
                df.to_csv(f"{path}/llama_generated_tasks_params{model}_dim{dim}_data{num_data}_tasks{total_tasks}.csv")
    
    return df