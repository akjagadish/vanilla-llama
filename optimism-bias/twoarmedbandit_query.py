
import openai
import gym
import envs.bandits
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
sys.path.insert(1, '/raven/u/ajagadish/vanilla-llama/')
from inference import LLaMAInference
import ipdb

num2words = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
env = gym.make('twoarmedbandit-v0')

# engine = "text-davinci-002"

def act(text=None, run_gpt=False, action_letters=['1', '2']):
    if run_gpt:
        # openai.api_key = "Your Key"
        # response = openai.Completion.create(
        #     engine = engine,
        #     prompt = text,
        #     max_tokens = 1,
        #     temperature = 0.0,
        # )
        raw_response = llama.generate([text], temperature=1., max_length=1)
        response = raw_response[0][0][len(text):].replace(' ', '')
        #import ipdb; ipdb.set_trace()
        if response not in action_letters:   #When answer is not part of the bandit arms
            try:
                text += f" {response}" + f"\nQ: Machine {response} is not part of this casino. Which machine do you choose between machine {action_letters[0]} and machine {action_letters[1]}?\nA: Machine",
                raw_response = llama.generate([text], temperature=1., max_length=1)
                response = raw_response[0][0][len(text):].replace(' ', '')
            except:
                import ipdb; ipdb.set_trace() 
                # response = '1' # action_letters[np.random.choice([1, 2])]
                # print('forced response to be 1')
        return response
    else:

        return NotImplementedError #torch.tensor([np.random.choice([1, 2])]) 


def reset(context, num_trials):

    instructions = "You are going to visit a casino owning two slot machines multiple times. "\
        "You earn money each time you choose a machine in a casino. "\
        #"You are randomly assigned to one of the four slot machines in every trial and you need to pick one of the two options.\n"\
                    
    trials_left = f"Your goal is to maximize the sum of received dollars within the next {num_trials} visits.\n"

    history = ""
    
    question = f"Question: You are now playing in Casino {num2words[int(context)]}." \
        "Which machine do you choose between Machine 1 and Machine 2?"\
        "Answer: Machine "

    return instructions, history, trials_left, question

def step(history, prev_machine, action, t):
    # get reward and next context
    observation, reward, done, _ = env.step(action-1)
    next_machine = observation[0, 3]
    # print(observation, reward, done)
    
    if t==0:
        history =  "You have received the following amount of dollars when playing in the past: \n"\
    # update history based on current action and trialss
    history += f"- Machine {str(int(action))} in Casino {num2words[int(prev_machine)]} delivered {float(reward)} dollars.\n"
    
    # update trials left
    trials_left = f"Your goal is to maximize the sum of received dollars within the next {env.max_steps-env.t} visit(s).\n"

    # prepare next question
    # question = f"Q: Which option do you choose for machine {num2words[int(next_machine)]}?\n"\
    #     "A: Option"
    question = f"Question: You are now playing in Casino {num2words[int(next_machine)]}. " \
        "Which machine do you choose between Machine 1 and Machine 2?\n"\
        "Answer: Machine "
    
    return history, trials_left, next_machine, question, done

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    args = parser.parse_args()

    start_loading = time.time()
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    start_generation = time.time()
    num_runs = 2
    run_gpt = True
    data = []
    for run in range(num_runs):
        done = False
        actions = []
        env.reset()
        num_trials = env.max_steps
        instructions, history, trials_left, question = reset(env.contexts[0, 0], env.max_steps)
        current_machine = env.contexts[0, 0]
        for t in range(num_trials):
            prompt = instructions + trials_left + "\n" + history + "\n"+ question
            print(prompt)
            print("\n")
            # LLM acts
            action = act(prompt, run_gpt)
            action = int(action) #torch.tensor([int(action)])
            # save values
            row = [run, t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), int(action)]
            data.append(row)
            if not done:
                # step into the next trial
                history, trials_left, current_machine, question, done = step(history, current_machine, action, t)

        df = pd.DataFrame(data, columns=['run', 'trial', 'casino', 'mean0', 'mean1', 'reward0', 'reward1', 'choice'])
        #print(df)
        df.to_csv('/u/ajagadish/vanilla-llama/optimism-bias/data/run_' + str(run) + '.csv')
        
        
        

