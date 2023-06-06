import openai
import gym
##import envs.bandits
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
sys.path.insert(1, '/raven/u/ajagadish/vanilla-llama/')
from inference import LLaMAInference
import ipdb
import pickle
import re

# num2words = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
# env = gym.make('palminteri2017-v0')

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
        #TODO: use stop words or maybe try to resample whenever it gives wierd results
        raw_response = llama.generate([text], temperature=1., max_length=300)[0][0]#[len(text):]

        #raw_response = llama.generate([text], temperature=temp, top_p=1, max_length=10, stop_words=["\n", ";"])[0][0][len(text):]
        #response = raw_response.replace(' ', '').replace(',', '').rstrip('\n').rstrip(';')
        #response = raw_response[0][0][len(text):].replace(' ', '')
        # #import ipdb; ipdb.set_trace()
        # if response not in action_letters:   #When answer is not part of the bandit arms
        #     try:
        #         # text += f" {response}" + f"\nQ: Machine {response} is not part of this casino. Which machine do you choose between machine {action_letters[0]} and machine {action_letters[1]}?\nA: Machine",
        #         # raw_response = llama.generate([text], temperature=1., max_length=1)
        #         # response = raw_response[0][0][len(text):].replace(' ', '')
        #     except:
        #         import ipdb; ipdb.set_trace() 
        #         # response = '1' # action_letters[np.random.choice([1, 2])]
        #         # print('forced response to be 1')
        return raw_response
    else:

        return NotImplementedError #torch.tensor([np.random.choice([1, 2])]) 


if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    args = parser.parse_args()
    num_dim = 3
    start_loading = time.time()
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    start_generation = time.time()
    num_runs = 1
    run_gpt = True
    data = []
    for run in range(num_runs):
        done = False
        actions = []
        #env.reset()
        num_trials = 1500 #env.max_steps
        
        # generates new datapoints consistently for 2 dimensional input features and two-category (65B model with temperature 1.) 
        # instructions = f"Each classification problem consists of a collection of input-target pairs."\
        #                 " Each input, x, is a vector of length 2, x=[x1, x2], containing inputs features, with each feature taking continuous values between 0 and 1."\
        #                 " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
        #                 " The following is 10 generated input-target pairs for one such classification problem:\n"\
        #                 "x=["

        # extends above prompt to 5 features
        # instructions = f"Each classification problem consists of a collection of input-target pairs."\
        #                 " Each input, x, is a vector of length 5, x=[x1, x2, x3,...,x5], containing inputs features, with each feature taking continuous values between 0 and 1."\
        #                 " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
        #                 " The following is 10 generated input-target pairs for one such classification problem:\n"\
        #                 "x=["

        instructions = f"A classification problem consists of a set of input-target pairs."\
                        " Each input, x, is a vector of length 3, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        " The following are 8 input-target pairs generated for one such classification problem:\n"\
                        "x=["

        # import ipdb; ipdb.set_trace()
        # print(act(instructions, run_gpt))
        data = []
        unparsable_data = []
        for t in range(num_trials):
            prompt = instructions 
            #print('Prompt:')
            #print(prompt)
            #print("\n")
            #print('LLMA outputs:')
            
            ## LLM acts
            action = act(prompt, run_gpt)

            patterns = [
                r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
                r"x=\[(.*?)\][^\n]*?y=(\w)",
                r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)",
                r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)",
                r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])",
                r"x=(\[.*?\])\s*->\s*([A-Z])",
                r"x=(\[.*?\]),\s*([A-Z])"
            ]

            for pattern in patterns:
                matches = re.findall(pattern, action, re.MULTILINE)
                if len(matches) > 0:
                    data.append(matches)
                    break
            
            if len(matches) == 0:
                unparsable_data.append(action)
            print(f'task {t}: no matches found' if len(matches) == 0 else f'task {t}: match found')


            with open(f"data/llama_generated_tasks_{args.model}_{num_dim}.txt", "wb") as fp:   
                #Pickling
                pickle.dump(data, fp)

            with open(f"data/llama_generated_tasks_{args.model}_{num_dim}_unparsed.txt", "wb") as fp:   
                #Pickling
                pickle.dump(unparsable_data, fp)


            ## save values
            #row = [run, t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), int(action)]
            #data.append(row)
            # if not done:
            #     # step into the next trial
            #     history, trials_left, current_machine, question, done = step(history, current_machine, action, t)

        #df = pd.DataFrame(data, columns=['run', 'example', 'dim0', 'dim1', 'target'])
        #print(df)
        #df.to_csv('/u/ajagadish/vanilla-llama/optimism-bias/data/run_' + str(run) + '.csv')

        # data.append(action)
            # data.append("\n\n\n")

            # matches1 = re.findall(r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])', action)
            # print(f"matches1: {matches1} \n")

            # matches2 = re.findall(r"x=\[(.*?)\][^\n]*?y=(\w)", action) 
            # print(f"matches2: {matches2}\n")
            
            # matches3 = re.findall(r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)", action, re.MULTILINE)
            # print(f"matches3: {matches3}\n")

            # matches4 = re.findall(r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)", action, re.MULTILINE)
            # print(f"matches4: {matches4}\n")

            # matches5 = re.findall(r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])", action, re.MULTILINE)
            # print(f"matches5: {matches5}\n")

            # matches6 = re.findall(r"x=(\[.*?\])\s*->\s*([A-Z])", action, re.MULTILINE)
            # print(f"matches6: {matches6}\n")

            # matches7 = re.findall(r"x=(\[.*?\]),\s*([A-Z])", action, re.MULTILINE)
            # print(f"matches7: {matches7}\n")

            # if len(matches1) > 0:
            #     data.append(matches1)
            # else:
            #     if len(matches2) > 0:
            #         data.append(matches2)
            #     else:
            #         if len(matches3) > 0:
            #             data.append(matches3)
            #         else:
            #             if len(matches4) > 0:
            #                 data.append(matches4)
            #             else:
            #                 if len(matches5) > 0:
            #                     data.append(matches5)
            #                 else:
            #                     if len(matches6) > 0:
            #                         data.append(matches6)
            #                     else:
            #                         if len(matches7) > 0:
            #                             data.append(matches7)
            #                         else:
            #                             data.append([])
        
        

