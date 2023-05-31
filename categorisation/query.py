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
        raw_response = llama.generate([text], temperature=1., max_length=250)[0][0]#[len(text):]

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
        num_trials = 1 #env.max_steps
        
        ## original prompt used on GPT-3
        # instructions = 'A classification problem consists of a list of input-target pairs.\
        #                 Each input is a vector of length 2, each entry takes continuous values between 0 and 1. \
        #                 The target is a function of the input vector and takes values of either A or B. \
        #                 Please generate 10 input-target pairs:'

        ## output in the form of tuples (doesn't work)
        # instructions = 'A classification problem consists of a list of input-target pairs.\
        #                 Each input is a vector of length 2, each entry takes continuous values between 0 and 1. \
        #                 The target is a function of the input vector and takes values of either A or B. \
        #                 Please generate 100 example input-target pairs in the form of tuples:'
        
       
        ## WORKING PROMPT, which provides some novel examples
        #instructions = 'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Please generate 10 new input-target pairs in the same format as the example. Here are the input-target pairs for an example classification problem: 1. Input: [0.5, 0.5], Target: A; 2. Input: [0.35, 0.31], Target: B; 3. Input: [0.12, 0.45], Target: B; 10. Input: [0.23, 0.46], Target: A; Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'
        
        ## same as above but with sampled random numbers provided as input pairs
        #instructions = f'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Please generate 10 new input-target pairs in the same format as the example. Here are the input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'
        
        ## reordered the text a bit to see if I reduce the repeats in the sample (does it help?)
        ## tried explicity adding do not include input-output pairs from the example did not help
        #instructions = f'A classification problem consists of a list of input-target pairs. Each input is a vector of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input vector and takes values of either A or B. Here are the input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Please generate 10 new input-target pairs in the same format as the example. Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'

        # reword a bit but still repeats the samples
        #instructions = f'A classification problem consists of input-target pairs. Each input is a list of length 2, each entry takes continuous values between 0 and 1. The target is a function of the input list and takes values of either A or B. Here are some input-target pairs for an example classification problem: 1. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; 2. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 3. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: B; 10. Input: [{round(np.random.rand(1)[0],2)}, {round(np.random.rand(1)[0],2)}], Target: A; Please generate 10 input-target pairs for a new classification problem in the same format as the example. Your output: 1. Input:' # 3. Input: [0.34, 0.56], Target: B;'
        
        # random_samples = np.random.rand(8).round(2)
        # instructions = f'A classification problem consists of a list of input-target pairs.\
        #                     Each input is a list of length 2, each entry takes continuous values between 0 and 1.\
        #                     The target is a function of the input vector and takes values of either A or B.\
        #                     Here are some input-target pairs for an example classification problem: \
        #                     1. Input: [{random_samples[0]}, {random_samples[1]}], Target: A;\
        #                     2. Input: [{random_samples[2]}, {random_samples[3]}], Target: B;\
        #                     3. Input: [{random_samples[4]}, {random_samples[5]}], Target: B;\
        #                     10. Input: [{random_samples[6]}, {random_samples[7]}], Target: A;\
        #                     Please generate 10 new input-target pairs in the same format as the example.\
        #                     Your output:\
        #                     1. Input:\
        #                 '
        
        
        # reformat above the way below doesn't seem yield the same results
        # instructions = """
        #                 A classification problem consists of a list of input-target pairs. 
        #                 Each input is a vector of length 2, each entry takes continuous values between 0 and 1. 
        #                 The target is a function of the input vector and takes values of either A or B. 
        #                 Please generate 10 new input-target pairs in the same format as the example.
        #                 Here are the input-target pairs for an example classification problem: 
        #                 1. Input: [0.5, 0.5], Target: A; 
        #                 2. Input: [0.35, 0.31], Target: B; 
        #                 3. Input: [0.12, 0.45], Target: B; 
        #                 10. Input: [0.23, 0.46], Target: A; 
        #                 Your output: 
        #                 1. Input: 
        #                 """
                        # 3. Input: [0.34, 0.56], Target: B;'
        `` 

        # This one interestingly generates code for generate data points
        # instructions = """ 
        #                 A classification problem consists of a list of input-target pairs. \
        #                 Each input, x, is a vector of length 2 x=[x1, x2], each entry takes continuous values between 0 and 1. \
        #                 The target is a function of the input vector and takes values of either y=A or y=B. \
        #                 Here are the input-target pairs from an example classification problem delimited by triple backticks: 
        #                 ```
        #                    1. x= [
        #                    2. Input: [0.35, 0.31], Target: B; 
        #                    3. Input: [0.12, 0.45], Target: B; 
        #                    …
        #                    10. Input: [0.23, 0.46], Target: A;
        #                 ```
        #                 Please generate 10 new input-target pairs in the same format as the example. 
        #                 Your output: 
        #                 x=[
        #                 """

        ## playing with Julian
        # instructions = f"A classification problem consists of a list of input-target pairs."\
        #                 " Each input, x, is a vector of length 2 x=[x1, x2], each entry takes continuous values between 0 and 1."\
        #                 " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
        #                 "The following is 10 generated input-target pairs:\n"\
        #                 "x=["

        # rewording the above prompt -- this consistently gives new dataponts 
        instructions = f"Each classification problem consists of a collection of input-target pairs."\
                        " Each input, x, is a vector of length 2, x=[x1, x2], containing inputs features, with each feature taking continuous values between 0 and 1."\
                        " The target, y, is a function of the input vector and takes values of either y=A or y=B.\n\n"\
                        " The following is 10 generated input-target pairs for one such classification problem:\n"\
                        "x=["
        import ipdb; ipdb.set_trace()

        for t in range(num_trials):
            prompt = instructions 
            # action = act(instructions, run_gpt)
            print('Prompt:')
            print(prompt)
            print("\n")
            print('LLMA outputs:')
            ## LLM acts
            action = act(prompt, run_gpt)
            print(action)

            ## save values
            #row = [run, t, int(current_machine), env.mean_rewards[0, t, 0].item(), env.mean_rewards[0, t, 1].item(), env.rewards[0, t, 0].item(),  env.rewards[0, t, 1].item(), int(action)]
            #data.append(row)
            # if not done:
            #     # step into the next trial
            #     history, trials_left, current_machine, question, done = step(history, current_machine, action, t)

        #df = pd.DataFrame(data, columns=['run', 'example', 'dim0', 'dim1', 'target'])
        #print(df)
        #df.to_csv('/u/ajagadish/vanilla-llama/optimism-bias/data/run_' + str(run) + '.csv')
        
        

