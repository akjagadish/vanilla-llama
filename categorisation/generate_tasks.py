import openai
import gym
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
sys.path.append("..")
sys.path.insert(1, '/u/ajagadish/vanilla-llama/')
sys.path.insert(1, '/raven/u/ajagadish/vanilla-llama/')
from inference import LLaMAInference
from prompts import retrieve_prompt
from utils import retrieve_features_and_categories
import ipdb
import pickle
import re
import os
from dotenv import load_dotenv
import anthropic

load_dotenv() # load environment variables from .env
TOKEN_COUNTER = 0
def act(text=None, run_gpt='llama', temperature=1., max_length=300):

    global TOKEN_COUNTER

    if run_gpt=='llama':

        #TODO: use stop words or maybe try to resample whenever it gives wierd results
        raw_response = llama.generate([text], temperature=temperature, max_length=max_length)[0][0]
        return raw_response
    
    elif run_gpt=='gpt4':
        
        openai.api_key = os.getenv("OPENAI_API_KEY_GPT4") # load key from env
        text = [{"role": "system", "content": "Do not generate any text other than the list of objects with their feature values and their corresponding category label in the format specified by the user."}, \
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
    parser.add_argument("--num-data", type=int, required=True, default=8)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)
    parser.add_argument("--num-runs", type=int, required=False, default=1)
    parser.add_argument("--prompt-version", type=str, required=False, default=None)
    parser.add_argument("--use-generated-tasklabels", action="store_true", required=False, default=False)
    parser.add_argument("--path-tasklabels", type=str, required=False, default='/raven/u/ajagadish/vanilla-llama/categorisation/data/tasklabels')
    parser.add_argument("--file-name-tasklabels", type=str, required=False, default=None)
    parser.add_argument("--start-task-id", type=int, required=False, default=0)

    args = parser.parse_args()
    start_loading = time.time()
    run_gpt = args.run_gpt #True
    assert args.model=='NA'if args.run_gpt=='gpt3' or args.run_gpt=='gpt4' or args.run_gpt=='claude' else False, "Only NA model is supported for GPT3"
    # model parameters
    temperature = args.temperature
    max_length = args.max_length
    # instruction parameters
    start_task_id = args.start_task_id
    num_tasks = args.num_tasks
    num_data = args.num_data
    num_dim = args.num_dim
    # runtime parameters
    proc_id = args.proc_id
    num_runs = args.num_runs
    prompt_version = args.prompt_version
    num_categories = 2

    patterns = [r'([\d.]+),([\d.]+),([\d.]+),([\w]+)',
                r'([\w\-]+),([\w\-]+),([\w\-]+),([\w]+)',
                r'([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+),([-\w\d,.]+)',
                r'([^,]+),([^,]+),([^,]+),([^,]+)',
                r'([^,\n]+),([^,\n]+),([^,\n]+),([^,\n]+)',
                r'(?:.*?:)?([^,-]+),([^,-]+),([^,-]+),([^,-]+)',
                r'([^,-]+),([^,-]+),([^,-]+),([^,-]+)',] if args.use_generated_tasklabels else \
                        [r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
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
                        r"(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),([A-Z])"
                        ]            

    # load LLaMA model and instructions
    if run_gpt == 'llama':
        llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
        instructions = retrieve_prompt('llama', version='v0', num_dim=num_dim, num_data=num_data)

    # load GPT-3 specific instructions
    elif run_gpt == 'gpt3':
        instructions = retrieve_prompt('gpt3', version='v1', num_dim=num_dim, num_data=num_data)

    # load GPT-4 specific instructions
    elif run_gpt == 'gpt4':
        instructions = retrieve_prompt('gpt4', version='v3', num_dim=num_dim, num_data=num_data)
    
    # load Claude specific instructions
    elif run_gpt == 'claude':
        instructions = retrieve_prompt('claude', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data)

    # run gpt models
    for run in range(num_runs):
        data, unparsable_data, raw_data, task_ids = [], [], [], []
        for t in range(start_task_id, start_task_id+num_tasks):
            #TODO: generate tasks in order or randomly?
            ## LLM acts
            if run_gpt == 'claude' and args.use_generated_tasklabels:
                assert args.file_name_tasklabels is not None, "Please provide a file name for the task labels"
                features, categories = retrieve_features_and_categories(path=args.path_tasklabels,\
                                                                        file_name=args.file_name_tasklabels,\
                                                                        task_id=t)
                assert len(features) == num_dim, "Number of features does not match the number of dimensions"
                assert len(categories) == num_categories, "Number of categories does not match the number of categories"
                instructions = retrieve_prompt('claude', version=f'v{prompt_version}', num_dim=num_dim, num_data=num_data, features=features, categories=categories)

            action = act(instructions, run_gpt, temperature, max_length)
            raw_data.append(action)

            for pattern in patterns:
                matches = re.findall(pattern, action, re.MULTILINE)
                if len(matches) > 0:
                    data.append(matches)
                    task_ids.append(t)
                    break

            if len(matches) == 0:
                unparsable_data.append(action)
            print(f'task {t}: no matches found' if len(matches) == 0 else f'task {t}: match found')
            
            # save data
            with open(f"data/parsed/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}.txt", "wb") as fp:   
                #pickling
                pickle.dump(data, fp)

            with open(f"data/parsed/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}_taskids.txt", "wb") as fp:
                pickle.dump(task_ids, fp)

            with open(f"data/unparsed/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}_unparsed.txt", "wb") as fp:   
                #pickling
                pickle.dump(unparsable_data, fp)

            with open(f"data/raw_data/{run_gpt}_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_pversion{prompt_version}_starttaskid{start_task_id}_lasttaskid{start_task_id+num_tasks}_raw.txt", "wb") as fp:
                #pickling
                pickle.dump(raw_data, fp)

    if run_gpt == 'gpt4':
        print(f'total tokens used: {TOKEN_COUNTER}')