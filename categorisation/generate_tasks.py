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


def act(text=None, run_gpt=False, temperature=1., max_length=300):
    if run_gpt:
        #TODO: use stop words or maybe try to resample whenever it gives wierd results
        raw_response = llama.generate([text], temperature=temperature, max_length=max_length)[0][0]
        return raw_response
    
    else:

        return NotImplementedError 


if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models)
    parser.add_argument("--num-tasks", type=int, required=False, default=1000)
    parser.add_argument("--num-dim", type=int, required=False, default=3)
    parser.add_argument("--num-data", type=int, required=False, default=8)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--max-length", type=int, required=False, default=300)
    parser.add_argument("--proc-id", type=int, required=False, default=0)


    args = parser.parse_args()
    num_dim = args.num_dim
    start_loading = time.time()
    num_tasks = args.num_tasks
    num_data = args.num_data
    temperature = args.temperature
    max_length = args.max_length
    proc_id = args.proc_id
    patterns = [
                r'x=\[(.*?)\][;,]?\s*y\s*=?\s*([AB])',
                r"x=\[(.*?)\][^\n]*?y=(\w)",
                r"x=\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\][^\n]*[y|&]=\s*(A|B)",
                r"x=\[?\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]?\s*(?:,|;|&|->| -> |---)?\s*[y|Y]\s*=\s*(A|B)",
                r"x=(\[.*?\])\s*---\s*y\s*=\s*([A-Z])",
                r"x=(\[.*?\])\s*->\s*([A-Z])",
                r"x=(\[.*?\]),\s*([A-Z])"
            ]
    instructions = f"A classification problem consists of a set of input-target pairs."\
                        f" Each input, x, is a vector of length {num_dim}, x = [x1, x2, x3], containing feature values that range continuously between 0 and 1."\
                        " The target, y, is a function of the input vector and can take on values of either y = A or y = B.\n\n"\
                        f" The following are {num_data} input-target pairs generated for one such classification problem:\n"\
                        "x=["

    # load LLaMA
    llama = LLaMAInference(args.llama_path, args.model, max_batch_size=2)
    print(f"Loaded model {args.model} in {time.time() - start_loading:.2f} seconds")

    start_generation = time.time()
    num_runs = 1
    run_gpt = True
    for run in range(num_runs):
        # import ipdb; ipdb.set_trace()
        # print(act(instructions, run_gpt))
        data = []
        unparsable_data = []
        done = False
        for t in range(num_tasks):
            print(instructions)
            ## LLM acts
            action = act(instructions, run_gpt, temperature, max_length)
            for pattern in patterns:
                matches = re.findall(pattern, action, re.MULTILINE)
                if len(matches) > 0:
                    data.append(matches)
                    break
            
            if len(matches) == 0:
                unparsable_data.append(action)
            print(f'task {t}: no matches found' if len(matches) == 0 else f'task {t}: match found')


            with open(f"data/llama_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}.txt", "wb") as fp:   
                #pickling
                pickle.dump(data, fp)

            with open(f"data/llama_generated_tasks_params{args.model}_dim{num_dim}_data{num_data}_tasks{num_tasks}_run{run}_procid{proc_id}_unparsed.txt", "wb") as fp:   
                #pickling
                pickle.dump(unparsable_data, fp)

