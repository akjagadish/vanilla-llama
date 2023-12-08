import numpy as np
import torch
from envs import CategorisationTask, ShepardsTask, NosofskysTask, LeveringsTask, SyntheticCategorisationTask, SmithsTask
import pandas as pd
import argparse
from tqdm import tqdm

def simulate(env_name=None, model_path=None, experiment='categorisation', env=None, model=None, mode='val', shuffle_trials=True, policy='binomial', beta=1., max_steps=100, batch_size=1, device='cpu'):
    
    if env is None:
        # load environment
        if experiment == 'synthetic':
            env = SyntheticCategorisationTask(max_steps=max_steps, shuffle_trials=shuffle_trials)
        if experiment == 'categorisation':
            env = CategorisationTask(data=env_name, mode=mode, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'shepard_categorisation':
            env = ShepardsTask(task=env_name, return_prototype=True, batch_size=batch_size, max_steps=max_steps, shuffle_trials=shuffle_trials)
        elif experiment == 'nosofsky_categorisation':
            env = NosofskysTask(task=env_name)
        elif experiment == 'levering_categorisation':
            env = LeveringsTask(task=env_name)
        elif experiment == 'smith_categorisation':
            env = SmithsTask(rule=env_name, return_prototype=True, batch_size=batch_size, max_steps=max_steps, shuffle_trials=shuffle_trials)

    if model is None: # load model
        model = torch.load(model_path)[1].to(device) if device=='cuda' else torch.load(model_path, map_location=torch.device('cpu'))[1].to(device)
        
    with torch.no_grad():
        model.eval()
        outputs = env.sample_batch()
        if (env.return_prototype is True) and hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, targets, stacked_prototypes = outputs
        else:
            packed_inputs, sequence_lengths, targets = outputs
        model.beta = beta  # model beta is adjustable at test time
        model.device = device
        model_choices = model(packed_inputs.float().to(device), sequence_lengths) 
        
        # sample from model choices probs using binomial distribution
        if policy=='binomial':
            model_choices = torch.distributions.Binomial(probs=model_choices).sample()
        elif policy=='greedy':
            model_choices = model_choices.round()

        model_choices = np.stack([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)]).squeeze() if batch_size>1 else np.stack([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)])
        true_choices = np.stack(targets).squeeze() if batch_size>1 else np.stack(targets)
        prototypes = np.stack(stacked_prototypes) if env.return_prototype else None
        input_features = packed_inputs[..., :-1]
        category_labels = torch.concat(env.stacked_labels, axis=0).float() if experiment=='nosofsky_categorisation' else None
        task_feature = env_name
        
    return model_choices, true_choices, sequence_lengths, category_labels, prototypes, input_features, task_feature
     
def simulate_metalearners_choices(env_name, model_path, experiment='categorisation', mode='test', shuffle_trials=False, beta=1., num_trials=100, num_runs=1, batch_size=1, device='cpu'):
    
    for run_idx in range(num_runs):

        model_choices, true_choices, sequences, category_labels,  prototypes, input_features, task_feature = simulate(env_name=env_name,\
                    model_path=model_path, experiment=experiment, mode=mode, shuffle_trials=shuffle_trials, \
                    beta=beta, batch_size=batch_size, max_steps=num_trials, device=device)
        last_task_trial_idx = 0

        # loop over batches, indexing them as tasks in the pd data frame
        for task_idx, (model_choices_task, true_choices_task, sequence_lengths_task, prototypes_task, input_features_task) in enumerate(zip(model_choices, true_choices, sequences, prototypes, input_features)):
            # loop over trials in each batch
            for trial_idx, (model_choice, true_choice, input_feature) in enumerate(zip(model_choices_task, true_choices_task, input_features_task)):
                
                data = {'run': run_idx, 'task': task_idx, 'trial': trial_idx + last_task_trial_idx, 'task_feature': task_feature, 'choice': int(model_choice), 'correct_choice': int(true_choice), \
                        **{f'feature{i+1}': input_feature[i].numpy() for i in range(len(input_feature))},\
                        **{f'prototype_feature{i+1}': prototypes_task[int(true_choice)][i] for i in range(len(prototypes_task[0]))}}

                # make a pandas data frame
                df = pd.DataFrame(data, index=[0]) if run_idx==0 and task_idx==0 and trial_idx==0 else pd.concat([df, pd.DataFrame(data, index=[0])])
            
            last_task_trial_idx = 0 #trial_idx + 1
    return df

def simulate_task(model_name=None, task_name=None, tasks=[None], beta=1., num_runs=1, num_trials=100, batch_size=1, device='cpu'):

    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
    for task in tasks:
        df = simulate_metalearners_choices(task, model_path, task_name, \
                                    beta=beta, shuffle_trials=True, num_runs=num_runs, \
                                    batch_size=batch_size, num_trials=num_trials, device=device)
        
        # concate into one csv
        if task == tasks[0]:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    # save to csv
    df_all.to_csv(f'/u/ajagadish/vanilla-llama/categorisation/data/meta_learner/{task_name}_{model_name[48:]}_beta={beta}_num_trials={num_trials}_num_runs={num_runs}.csv', index=False)

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--task-name', type=str, required=True, help='task name')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    parser.add_argument('--beta', type=float, default=1., help='beta value for softmax')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    beta = args.beta

    if args.task_name == 'shepard_categorisation':
        simulate_task(model_name=args.model_name, task_name=args.task_name, tasks=np.arange(1,7), beta=beta, num_runs=1, batch_size=1, num_trials=100, device=device)#1000
    elif args.task_name == 'smith_categorisation':
        simulate_task(model_name=args.model_name,  task_name=args.task_name, tasks=['linear', 'nonlinear'], beta=beta, num_runs=1, batch_size=1, num_trials=616, device=device)#300