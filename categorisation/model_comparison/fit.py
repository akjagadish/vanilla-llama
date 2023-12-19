import numpy as np
import torch
from human_envs import Badham2017, Devraj2022
import argparse
from tqdm import tqdm
import sys
sys.path.insert(0, '/u/ajagadish/vanilla-llama/categorisation/rl2')

def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, beta=1., epsilon=0., method='soft_sigmoid', device='cpu', **kwargs):
    
    # load model
    model = torch.load(model_path)[1].to(device) if device=='cuda' else torch.load(model_path, map_location=torch.device('cpu'))[1].to(device)
           
    with torch.no_grad():
        
        # model setup: eval mode and set beta
        model.eval()
        model.beta = beta 
        model.device = device

        # env setup: sample batch from environment and unpack
        outputs = env.sample_batch(participant)
        if (env.return_prototype is True) and hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, correct_choices, human_choices, stacked_prototypes = outputs
        else:
            packed_inputs, sequence_lengths, correct_choices, human_choices = outputs

        # get model choices
        model_choice_probs = model(packed_inputs.float().to(device), sequence_lengths) 
       
        
        if method == 'eps_greedy': 
            assert beta == 1., "beta must be 1 for eps_greedy"
            # make a new tensor containing model_choice_probs for each trial for option 1 and 1-model_choice_probs for option 2
            #probs = torch.stack([model_choice_probs[i, :sequence_lengths[i]] for i in range(len(model_choice_probs))])
            probs = torch.cat([1-model_choice_probs, model_choice_probs], axis=2)
            # keep only the probabilities for the chosen option from human_choices
            probs = torch.vstack([probs[batch, i, human_choices[batch, i, 0].long()] for batch in range(probs.shape[0]) for i in range(sequence_lengths[batch])])
            probs_with_guessing = probs * (1 - epsilon) + epsilon * 0.5 
            loglikehoods = torch.log(probs_with_guessing)
            summed_loglikelihoods = loglikehoods.sum()
        elif method == 'soft_sigmoid':
            # compute log likelihoods of human choices under model choice probs (binomial distribution)
            loglikehoods = torch.distributions.Binomial(probs=model_choice_probs).log_prob(human_choices)
            summed_loglikelihoods = torch.vstack([loglikehoods[idx, :sequence_lengths[idx]].sum() for idx in range(len(loglikehoods))]).sum()
        
        
        # sum log likelihoods only for unpadded trials per condition and compute chance log likelihood
        chance_loglikelihood = sum(sequence_lengths) * np.log(0.5)

        # task performance
        model_choices = torch.distributions.Binomial(probs=model_choice_probs).sample()
        model_choices = torch.concat([model_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)], axis=0).squeeze().float()
        correct_choices = torch.concat([correct_choices[i, :seq_len] for i, seq_len in enumerate(sequence_lengths)], axis=0).squeeze().float()
        correct_choices = correct_choices.reshape(-1).float().to(device)
        model_accuracy = (model_choices==correct_choices).sum()/(model_choices.shape[0])

    return summed_loglikelihoods, chance_loglikelihood, model_accuracy 

              
def evaluate_model(env=None, model_name=None, beta=1., epsilon=0., method='soft_sigmoid', num_runs=5, **task_features):
    '''  compute log likelihoods of human choices under model choice probs based on binomial distribution
    '''

    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
    participants = env.data.participant.unique()
    loglikelihoods, p_r2, model_acc = [], [], []
    for participant in participants:
        ll, chance_ll, acc  = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, experiment='badham2017deficits', shuffle_trials=True,\
                                                                            beta=beta, epsilon=epsilon, method=method, **task_features)
        loglikelihoods.append(ll)
        p_r2.append(1 - (ll/chance_ll))
        model_acc.append(acc)
    
    loglikelihoods = np.array(loglikelihoods)
    p_r2 = np.array(p_r2)
    model_acc = np.array(model_acc)
    
    return -loglikelihoods, p_r2, model_acc

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--task-name', type=str, required=True, help='task name')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    parser.add_argument('--method', type=str, default='soft_sigmoid', help='method for computing model choice probabilities')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    
    if args.method == 'soft_sigmoid':
        # beta sweep
        betas = np.arange(0., 10., 0.05)
        parameters = betas
    elif args.method == 'eps_greedy':
        # epsilon sweep
        epsilons = np.arange(0., 1., 0.05)
        parameters = epsilons
    elif args.method == 'both':
        epsilons = np.arange(0., 1., 0.05)
        betas = np.arange(0., 10., 0.05)
        parameters = epsilons

    else:
        raise NotImplementedError
    
    nlls, pr2s, accs = [], [], []
    for idx, param in enumerate(parameters):
        epsilon = param if args.method == 'eps_greedy' else 0.
        beta = param if args.method == 'soft_sigmoid' else 1.
        if args.task_name == 'badham2017':
            env = Badham2017() 
            #TODO: for task in tasks:
            task_features = {}
            nll_per_beta, pr2_per_beta, model_acc_per_beta = evaluate_model(env=env, model_name=args.model_name, epsilon=epsilon, beta=beta, method=args.method, num_runs=1, **task_features)
        elif args.task_name == 'devraj2022':
            env = Devraj2022()
            nll_per_beta, pr2_per_beta, model_acc_per_beta = evaluate_model(env=env, model_name=args.model_name, epsilon=epsilon, beta=beta, method=args.method, num_runs=1)
        else:
            raise NotImplementedError
        nlls.append(nll_per_beta)
        pr2s.append(pr2_per_beta)
        accs.append(model_acc_per_beta)

    pr2s = np.array(pr2s)
    min_nll_index = np.argmin(np.stack(nlls), 0)
    pr2s_min_nll = np.stack([pr2s[min_nll_index[idx], idx] for idx in range(pr2s.shape[1])])
    # print(f"beta with min nll: {betas[min_nll_index]}")
    # print(f"beta with min nll: {parameters[min_nll_index]}" if args.method == 'soft_sigmoid' else f"epsilon with min nll: {parameters[min_nll_index]}")

    # save list of results
    save_path = f"/u/ajagadish/vanilla-llama/categorisation/data/model_comparison/{args.task_name}_{args.model_name}_{args.method}"
    np.savez(save_path, betas=parameters, nlls=nlls, pr2s=pr2s, accs=accs)

#python model_comparison/fit.py --model-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8 --task-name badham2017

