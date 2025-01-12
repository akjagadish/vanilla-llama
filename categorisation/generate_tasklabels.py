import openai
import gym
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
sys.path.append("..")
sys.path.insert(1, '/u/ajagadish/vanilla-llama/categorisation/')
sys.path.insert(1, '/raven/u/ajagadish/vanilla-llama/')
from inference import LLaMAInference
from prompts import retrieve_tasklabel_prompt
import ipdb
import pickle
import re
import os
from dotenv import load_dotenv
import anthropic
from utils import pool_tasklabels
load_dotenv() # load environment variables from .env
TOKEN_COUNTER = 0
def act(text=None, run_gpt='llama', temperature=1., max_length=300):

    global TOKEN_COUNTER

    if run_gpt=='llama':

        raw_response = llama.generate([text], temperature=temperature, max_length=max_length)[0][0]
        return raw_response
    
    elif run_gpt=='gpt4':
        
        openai.api_key = os.getenv("OPENAI_API_KEY_GPT4") # load key from env
        text = [{"role": "system", "content": "Do not generate any text other than the list of feature names for the stimuli and their corresponding category label in the format specified by the user."}, \
                {"role": "user", "content": text}]
        engine = 'gpt-4'
        try:
            response = openai.ChatCompletion.create(
                model = engine,
                messages = text,
                max_tokens = max_length,
                temperature = temperature,
            )
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
            TOKEN_COUNTER += response['usage']['total_tokens'] 
            return response.choices[0].text.strip().replace(' ', '')
        except:
            print("Error")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(exc_value)
            #time.sleep(3**iter)
            
    elif run_gpt=='claude':

        client = anthropic.Anthropic()
        response = client.completions.create(
                prompt = anthropic.HUMAN_PROMPT + text + anthropic.AI_PROMPT,
                #stop_sequences=[anthropic.HUMAN_PROMPT],
                model="claude-2",
                temperature=temperature,
                max_tokens_to_sample=max_length,
            ).completion.replace(' ', '')
    
        return response
 
    else:

        return NotImplementedError 


if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B", "NA"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=False, default=None)
    parser.add_argument("--model", type=str, required=False, choices=models)
    parser.add_argument("--run-gpt", type=str, required=True, choices=['llama', 'gpt3', 'gpt4', 'claude'])
    parser.add_argument("--num-tasks", type=int, required=True, default=1000)
    parser.add_argument("--num-dim", type=int, required=True, default=3)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)
    parser.add_argument("--num-runs", type=int, required=False, default=1)
    parser.add_argument('--first-run-id', type=int, default=0, help='id of the first run')
    parser.add_argument("--prompt-version", type=str, required=False, default=None)
    parser.add_argument("--path", type=str, required=False, default='/raven/u/ajagadish/vanilla-llama/categorisation/data/tasklabels')
    parser.add_argument("--pool", action='store_true', required=False, default=False)

    args = parser.parse_args()
    start_loading = time.time()
    run_gpt = args.run_gpt #True
    assert args.model=='NA'if args.run_gpt=='gpt3' or args.run_gpt=='gpt4' or args.run_gpt=='claude' else False, "Only NA model is supported for GPT3"
    # model parameters
    temperature = args.temperature
    max_length = args.max_length
    # instruction parameters
    num_tasks = args.num_tasks
    num_dim = args.num_dim
    # runtime parameters
    proc_id = args.proc_id
    num_runs = args.num_runs
    first_run_id = args.first_run_id
    prompt_version = args.prompt_version

    if args.pool:

        pool_tasklabels(args.path, args.run_gpt, args.model, args.num_dim, args.num_tasks, args.num_runs, args.proc_id, args.prompt_version)

    else:
        patterns = [
                    r'\d+\.(.+?)\n',
                    ]            

        # load LLaMA model and instructions
        if run_gpt == 'llama':
            llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
            raise NotImplementedError

        # load GPT-3 specific instructions
        elif run_gpt == 'gpt3':
            raise NotImplementedError

        # load GPT-4 specific instructions
        elif run_gpt == 'gpt4':
            raise NotImplementedError
        
        # load Claude specific instructions
        elif run_gpt == 'claude':
            instructions = retrieve_tasklabel_prompt('claude', version=f'v{prompt_version}', num_dim=num_dim, num_tasks=num_tasks)
        
        # run gpt models
        for run in range(first_run_id, first_run_id+num_runs):
            stimulus_dimensions, categories = [], []
            ## LLM acts
            # print(instructions)
            action = act(instructions, run_gpt, temperature, max_length)
            # print(action)
            matches = re.findall(patterns[0], action, re.MULTILINE)
 
            if len(matches)>0:
                for match in matches: 
                    categories.append(match.split(',')[-2:]) # last two labels are categories
                    stimulus_dimensions.append(match.split(',')[:-2]) # rest are stimulus dimensions
                # save data
                df = pd.DataFrame({'feature_names': stimulus_dimensions, 'category_names': categories,  'task_id': np.arange(len(stimulus_dimensions)),})
                file_name = f'{run_gpt}_generated_tasklabels_params{args.model}_dim{num_dim}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}'
                df.to_csv(f'{args.path}/{file_name}.csv')
            else:
                print(f'no tasks were successfully parsed')

            if run_gpt=='gpt3' or run_gpt=='gpt4':
                print(f'total tokens used: {TOKEN_COUNTER}')