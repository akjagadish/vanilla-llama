import openai
import gym
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
import os
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

TOKEN_COUNTER = 0
def act(text=None, run_gpt='llama', temperature=1., max_length=300):

    global TOKEN_COUNTER

    if run_gpt=='llama':

        #TODO: use stop words or maybe try to resample whenever it gives wierd results
        raw_response = llama.generate([text], temperature=temperature, max_length=max_length)[0][0]
        return raw_response
    
    elif run_gpt=='gpt4':
        
        openai.api_key = os.getenv("OPENAI_API_KEY_GPT4") # load key from env
        #text = [{"role": "user", "content": text}]  
        text = [{"role": "system", "content": "Do not generate any text other than the input-target pairs in the format specified by the user."}, \
                {"role": "user", "content": text}]
        engine = 'gpt-4'
        try:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = text,
                max_tokens = max_length,
                temperature = temperature,
            )
            #ipdb.set_trace()
            TOKEN_COUNTER += response['usage']['total_tokens'] 
            return response.choices[0].message.content.replace(' ', '')
        except:
            print("Error, trying again...ratelimiterror")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)


    elif run_gpt=='gpt3':
        
        openai.api_key = os.getenv("OPENAI_API_KEY") # load key from env
        engine = "text-davinci-003"
        try:
            response = openai.Completion.create(
                engine = engine,
                prompt = text,
                max_tokens = max_length,
                temperature = temperature,
            )
            #ipdb.set_trace()
            TOKEN_COUNTER += response['usage']['total_tokens'] 
            return response.choices[0].text.strip().replace(' ', '')
        except:
            print("Error")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            #time.sleep(3**iter)
    else:

        return NotImplementedError 


if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B", "NA"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    parser.add_argument("--run-gpt", type=str, required=True, choices=['llama', 'gpt3', 'gpt4'])
    parser.add_argument("--num-tasks", type=int, required=False, default=1000)
    parser.add_argument("--num-dim", type=int, required=False, default=3)
    parser.add_argument("--num-data", type=int, required=False, default=8)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)
    parser.add_argument("--num-runs", type=int, required=False, default=1)

    args = parser.parse_args()
    start_loading = time.time()
    run_gpt = args.run_gpt #True
    assert args.model=='NA'if args.run_gpt=='gpt3' or args.run_gpt=='gpt4' else False, "Only NA model is supported for GPT3"
    # model parameters
    temperature = args.temperature
    max_length = args.max_length
    # instruction parameters
    num_tasks = args.num_tasks
    num_data = args.num_data
    num_dim = args.num_dim
    # runtime parameters
    proc_id = args.proc_id
    num_runs = args.num_runs

    patterns = [
                r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
                r"x=\[(.*?)\][^\n]*?y=(\w)",
                r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)",
                r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)",
                r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])",
                r"x=(\[.*?\])\s*->\s*([A-Z])",
                r"x=(\[.*?\]),\s*([A-Z])",
                r"^([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2}),(A|B)$",
                r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)\]",
                r"n[0-9]+\.\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                r"\[\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(\'A\'|\'B\')\]",
                r"\[([0-9]\.[0-9]{2}),(0\.[0-9]{2}),(0\.[0-9]{2})\],(A|B)",
                ]            

    # load LLaMA model and instructions
    if run_gpt == 'llama':
        llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
        print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")
        instructions = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" The following are {str(num_data)} input-target pairs generated for one such classification problem:\n"\
                        "x=["
    # load GPT-3 specific instructions
    elif run_gpt == 'gpt3':
        # instructions = f"A classification problem consists of a set of input-target pairs."\
        #                 f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
        #                 " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
        #                 f" Please generate a list of {str(num_data)} input-target pairs using the following template for each row:\n"\
        #                 f"- [x1, x2, x3], y"
        instructions = f"A categorisation problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                        " You can choose any naturalistic decision function for the mapping from input to target.  \n\n"\
                        f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                        f"- [x1, x2, x3], y"
    # load GPT-4 specific instructions
    elif run_gpt == 'gpt4':
        # instructions = f"A categorisation problem consists of a set of input-target pairs."\
        #                 f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
        #                 " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
        #                 f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
        #                 f"- [x1, x2, x3], y" ## got code to generate output once but otherwise consistent 
        instructions = f"A categorisation problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {str(num_dim)}, x = [x1, x2, x3], containing feature values (rounded to 2 decimals) that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B."\
                        " You can choose any naturalistic decision function for the mapping from input to target. \n\n"\
                        f" Please generate a list of {str(num_data)} input-target pairs for one such categorisation problem using the following template for each row:\n"\
                        f"- [x1, x2, x3], y"\
                        #f" Do not generate any text but just provide the input-target pairs."
        
                        
    # run gpt models
    for run in range(num_runs):
        data, unparsable_data = [], []
        for t in range(num_tasks):
            print(instructions)
            ## LLM acts
            #ipdb.set_trace()
            action = act(instructions, run_gpt, temperature, max_length)
            print(action)
            for pattern in patterns:
                matches = re.findall(pattern, action, re.MULTILINE)
                if len(matches) > 0:
                    data.append(matches)
                    break
            #ipdb.set_trace()
            if len(matches) == 0:
                unparsable_data.append(action)
            print(f'task {t}: no matches found' if len(matches) == 0 else f'task {t}: match found')

            # save data
            with open(f"data/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}.txt", "wb") as fp:   
                #pickling
                pickle.dump(data, fp)

            with open(f"data/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_unparsed.txt", "wb") as fp:   
                #pickling
                pickle.dump(unparsable_data, fp)

    print(f'total tokens used: {TOKEN_COUNTER}')