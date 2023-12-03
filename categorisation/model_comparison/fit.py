import numpy as np
import torch
from human_envs import Badham2017, Devraj2022
import argparse
from tqdm import tqdm
import sys
sys.path.insert(0, '/u/ajagadish/vanilla-llama/categorisation/rl2')

def compute_loglikelihood_human_choices_under_model(env=None, model_path=None, participant=0, experiment='shepard_categorisation', shuffle_trials=False, policy='binomial', beta=1., device='cpu', batch_size=10, **kwargs):
    
    # load model
    model = torch.load(model_path)[1].to(device) if device=='cuda' else torch.load(model_path, map_location=torch.device('cpu'))[1].to(device)
           
    with torch.no_grad():
 
        model.eval() # set model to eval mode
        # sample batch from environment
        outputs = env.sample_batch(participant)
        
        if (env.return_prototype is True) and hasattr(env, 'return_prototype'):
            packed_inputs, sequence_lengths, targets, human_targets, stacked_prototypes = outputs
        else:
            packed_inputs, sequence_lengths, targets, human_targets = outputs
        
        # set beta
        model.beta = beta 
        model.device = device

        # get model choices
        model_choice_probs = model(packed_inputs.float().to(device), sequence_lengths) 
       
        # compute log likelihoods of human choices under model choice probs (binomial distribution)
        loglikehoods = torch.distributions.Binomial(probs=model_choice_probs).log_prob(human_targets)
        
        # sum log likelihoods only for unpadded trials per condition
        summed_loglikehoods = torch.stack([loglikehoods[idx][:sequence_lengths[idx]].sum() for idx in range(len(loglikehoods))]).sum()
        chance_loglikelihood = sum(sequence_lengths) * np.log(0.5)

    return summed_loglikehoods, chance_loglikelihood 

              
def evaluate_badham2017(model_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, batch_size=10, num_eval_tasks=1113, synthetic=False):

    # setup model params
    #model_name = f"env={env_name}_noise{0.}_shuffle{True}_run=0.pt"
    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
    
    # load environment
    env = Badham2017() #load human data

    #TODO: for task in tasks: (all tasks or just one task)
    task_features = {'task':0, 'all_tasks': True}
    participants = env.data.participant.unique()
    loglikelihoods, p_r2 = [], []
    for participant in participants:
        # compute log likelihoods of human choices under model choice probs (binomial distribution)
        ll, chance_ll = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, experiment='badham2017deficits', shuffle_trials=True,\
                                                                            beta=beta, batch_size=batch_size, max_steps=num_trials, **task_features)
        loglikelihoods.append(ll)
        p_r2.append(1 - (ll/chance_ll))
    
    loglikelihoods = np.array(loglikelihoods)
    
    return -loglikelihoods, p_r2

def evaluate_devraj2022(model_name=None, experiment=None, tasks=[None], beta=1., noises=[0.05, 0.1, 0.0], shuffles=[True, False], shuffle_evals=[True, False], num_runs=5, num_trials=96, batch_size=10, num_eval_tasks=1113, synthetic=False):

    # setup model params
    #model_name = f"env={env_name}_noise{0.}_shuffle{True}_run=0.pt"
    model_path = f"/u/ajagadish/vanilla-llama/categorisation/trained_models/{model_name}.pt"
    
    # load environment
    env = Devraj2022() #load human data

    #TODO: for task in tasks: (all tasks or just one task)
    task_features = {'task':0, 'all_tasks': True}
    participants = env.data.participant.unique()
    loglikelihoods, p_r2 = [], []
    for participant in participants:

        # compute log likelihoods of human choices under model choice probs (binomial distribution)
        ll, chance_ll = compute_loglikelihood_human_choices_under_model(env=env, model_path=model_path, participant=participant, experiment='badham2017deficits', shuffle_trials=True,\
                                                                            beta=beta, batch_size=batch_size, max_steps=num_trials, **task_features)
        loglikelihoods.append(ll)
        p_r2.append(1 - (ll/chance_ll))
    
    loglikelihoods = np.array(loglikelihoods)
    
    return -loglikelihoods, p_r2

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='save meta-learner choices on different categorisation tasks')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--task-name', type=str, required=True, help='task name')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    
    betas = np.arange(0., 10., 0.05)
    nlls, pr2s = [], []
    
    for idx, beta in enumerate(betas):
        if args.task_name == 'badham2017':
            nll_per_beta, pr2_per_beta = evaluate_badham2017(model_name=args.model_name, tasks=np.arange(1,7), beta=0.3, noises=[0.0], shuffles=[True], shuffle_evals=[False], num_runs=1, batch_size=1, num_trials=1000)
        elif args.task_name == 'devraj2022':
            nll_per_beta, pr2_per_beta = evaluate_devraj2022(model_name=args.model_name, tasks=np.arange(1,7), beta=0.3, noises=[0.0], shuffles=[True], shuffle_evals=[False], num_runs=1, batch_size=1, num_trials=1000)
        else:
            raise NotImplementedError
        nlls.append(nll_per_beta)
        pr2s.append(pr2_per_beta)

    pr2s = np.array(pr2s)
    min_nll_index = np.argmin(np.stack(nlls), 0)
    pr2s_min_nll = np.stack([pr2s[min_nll_index[idx], idx] for idx in range(pr2s.shape[1])])
    print(f"beta with min nll: {betas[min_nll_index]}")

    # save list of results
   
    save_path = f"/u/ajagadish/vanilla-llama/categorisation/model_comparison/{args.task_name}_{args.model_name}_beta_sweep"
    np.savez(save_path, betas=betas, nlls=nlls, pr2s=pr2s)

#python model_comparison/fit.py --model-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8 --task-name badham2017

