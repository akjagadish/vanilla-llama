import numpy as np
import torch
from envs import CategorisationTask, ShepardsTask, NosofskysTask, LeveringsTask, SyntheticCategorisationTask, SmithsTask
import pandas as pd
import argparse
from tqdm import tqdm

def evaluate_single_run(env_name=None, model_path=None, experiment='categorisation', env=None, model=None, mode='val', shuffle_trials=False, policy='binomial', beta=1., max_steps=70, device='cpu', batch_size=10):
    
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

       
def save_metalearners_choices(env_name, model_path, experiment='categorisation', mode='test', shuffle_trials=False, beta=1., num_trials=96, num_runs=5, batch_size=10):
    
    for run_idx in range(num_runs):

        model_choices, true_choices, sequences, category_labels,  prototypes, input_features, task_feature = evaluate_single_run(env_name=env_name,\
                    model_path=model_path, experiment=experiment, mode=mode, shuffle_trials=shuffle_trials, \
                    beta=beta, batch_size=batch_size, max_steps=num_trials)
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
        
        
def evaluate_nosofsky1994(env_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, batch_size=10, num_eval_tasks=1113, synthetic=False):

    model_name = f"env={env_name}_noise{0.}_shuffle{True}_run=0.pt"
    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"
    for task in tasks:
        df = save_metalearners_choices(task, model_path, 'shepard_categorisation', \
                                    beta=beta, shuffle_trials=True, num_runs=num_runs, \
                                    batch_size=batch_size, num_trials=num_trials)
        
        # concate into one csv
        if task == tasks[0]:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    # save to csv
    df_all.to_csv(f'/u/ajagadish/vanilla-llama/categorisation/data/meta_learner/shepard_categorisation_{model_name[:-3]}_beta={beta}_num_trials={num_trials}_num_runs={num_runs}.csv', index=False)

def evaluate_smith1998(env_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, batch_size=10, num_eval_tasks=1113, synthetic=False, run=0):

    model_name = f"env={env_name}_noise{0.}_shuffle{True}_run={run}.pt"
    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}"

    for task in tasks:
        df = save_metalearners_choices(task, model_path, 'smith_categorisation', \
                                    beta=beta, shuffle_trials=True, num_runs=num_runs, \
                                    batch_size=batch_size, num_trials=num_trials)
        
        # concate into one csv
        if task == tasks[0]:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    # save to csv
    df_all.to_csv(f'/u/ajagadish/vanilla-llama/categorisation/data/meta_learner/smithstask_{model_name[:-3]}_beta={beta}_num_trials={num_trials}_num_runs={num_runs}.csv', index=False)


if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--task-name', type=str, required=True, help='task name')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    
    env_model_name = args.model_name
    #'claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8'
    if args.task_name == 'nosofsky1994':
        evaluate_nosofsky1994(env_name=env_model_name, tasks=np.arange(1,7), beta=0.3, noises=[0.0], shuffles=[True], shuffle_evals=[False], num_runs=1, batch_size=1, num_trials=1000)
    elif args.task_name == 'smith1998':
        evaluate_smith1998(env_name=env_model_name, tasks=['linear', 'nonlinear'], beta=0.3, noises=[0.0], shuffles=[True], shuffle_evals=[False], num_runs=1, batch_size=1, num_trials=300, run=1)