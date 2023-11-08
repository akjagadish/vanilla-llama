import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
import numpy as np
import torch
import statsmodels.discrete.discrete_model as sm
import ipdb

#TODO: bias term? an additional guessing-rate parameter was added to the exemplar model wihich assumed that some proportion of the time (G) participants simply guessed Category A or B haphazardly.
#TODO: hill-climbing search for the best weights run 4 times
#NOTE: model usually fitted to blocks of trials for example 2 blocks of 28 trials each
    
class GeneralizedContextModel():
    """ Generalized Context Model (GCM) """
    
    def __init__(self, num_features=4, distance_measure=1, num_iterations=1):
        
        self.bounds = [(0, 20), # sensitivity
                       (0, 1), # bias
                       ]     
        self.weight_bound = [(0, 1)] # weights
        self.distance_measure = distance_measure  
        self.num_iterations = num_iterations
        self.num_features = num_features

    def loo_nll(self, df):
        """ compute negative log likelihood for left out participants
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for left out participants"""

        df_train = df.loc[df['is_training']]
        df_test = df.loc[~df['is_training']]
        
        self.bounds.extend(self.weight_bound * self.num_features)
        
        best_params = self.fit_parameters(df_train)
        nll = self.compute_nll(best_params, df_test)
        
        return nll

    def fit_participants(self, df):
        """ fit gcm to individual participants and compute negative log likelihood 
        
        args:
        df: dataframe containing the data
        
        returns:
        nll: negative log likelihood for each participant"""

        num_participants = len(df['participant'].unique())
        log_likelihood, r2 = np.zeros(num_participants), np.zeros(num_participants)
        self.bounds.extend(self.weight_bound * self.num_features)

        for idx, participant_id in enumerate(df['participant'].unique()):
            df_participant = df[(df['participant'] == participant_id)]
            best_params = self.fit_parameters(df_participant)
            log_likelihood[idx] = -self.compute_nll(best_params, df_participant)
            r2[idx] = 1 - (log_likelihood[idx]/(df_participant.trial.max()*np.log(1/2)))
        
        return log_likelihood, r2


    def fit_parameters(self, df):
        """ fit parameters using scipy optimiser 
        
        args:
        df: dataframe containing the data
        
        returns:
        best_params: best parameters found by the optimiser
        """
        
        best_fun = np.inf
        # define the constraint that the weights must sum to 1 as an obj
        constraint_obj = {'type': 'eq', 'fun': self.constraint}
        for _ in range(self.num_iterations):
    
            result = minimize(
                fun=self.compute_nll,
                x0=[np.random.uniform(x, y) for x, y in self.bounds],
                args=(df),
                bounds=self.bounds,
                constraints=constraint_obj
            )
            
            # result = differential_evolution(self.compute_nll, 
            #                     bounds=self.bounds, 
            #                     args=(df),
            #                     maxiter=100)
            
            best_params = result.x

            if result.fun < best_fun:
                best_fun = result.fun
                best_res = result

        if best_res.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")
        
        return best_params

    def constraint(self, params):
        """ define the constraint that the weights must sum to 1 """
        
        return np.sum(params[2:]) - 1

    def compute_nll(self, params, df):
        """ compute negative log likelihood of the data given the parameters 
        
        args:
        params: parameters of the model
        df: dataframe containing the data
        
        returns:
        negative log likelihood of the data given the parameters
        """
   
        ll = 0.
        num_tasks = df['task'].max() + 1
        num_trials = df['trial'].max() + 1
        num_categories = df['choice'].nunique()
        stimuli_seen = [[] for i in range(num_categories)] # list of lists to store objects seen so far within each category
        categories = {'j': 0, 'f': 1}

        for task_id in range(num_tasks):
            df_task = df[(df['task'] == task_id)]

            for trial_id in range(num_trials):
                df_trial = df_task[(df_task['trial'] == trial_id)]
                choice = categories[df_trial.choice.item()] if df_trial.choice.item() in categories else df_trial.choice.item()
            
                true_choice = categories[df_trial.correct_choice.item()] if df_trial.correct_choice.item() in categories else df_trial.correct_choice.item()

                # load num features of the current stimuli
                current_stimuli = df_trial[['feature{}'.format(i+1) for i in range(self.num_features)]].values
                
                # given stimuli and list of objects seen so far within cateogry return probablity the object belongs to each category 
                ll += self.gcm(params, current_stimuli, stimuli_seen, choice)

                # update stimuli seen
                stimuli_seen[true_choice].append(current_stimuli)
    
        return -ll

    def gcm(self, params, current_stimuli, stimuli_seen, choice):
        """ return log likelihood of the choice given the stimuli and stimuli seen so far
         
        args:
        params: parameters of the model
        current_stimuli: features of the current stimuli
        stimuli_seen: list of lists of features of stimuli seen so far within each category
        choice: choice made by the participant
        
        returns:
        log likelihood of the choice given the stimuli and stimuli seen so far
        """
    
        sensitivity, bias = params[:2]
        weights = params[2:]
        category_similarity = np.zeros(len(stimuli_seen))
       
        for idx in range(len(stimuli_seen)):
            if len(stimuli_seen[idx]) == 0:
                # if no stimuli seen yet within category, similarity is set to unseen similarity
                category_similarity[idx] = bias
            else:
                # compute attention weighted similarity measure
                category_similarity[idx] = self.compute_attention_weighted_similarity(current_stimuli, np.stack(stimuli_seen[idx]), (weights, sensitivity))

        # compute category probabilities
        category_probabilities = self.compute_category_probabilities(category_similarity, bias)
        
        # compute log likelihood
        log_likelihood = np.log(category_probabilities[choice])

        return log_likelihood
    
    def compute_attention_weighted_similarity(self, x, y, params):
        """ compute attention weighted similarity between current stimuli and stimuli seen so far 
        args:
        x: current stimuli
        y: stimuli seen so far
        params: attention weights and sensitivity
        
        returns:
        s: similarity between current stimuli and stimuli seen so far 
        """
        weights, sensitivity = params
       
        # compute distance between stimuli vectors with features weighted by attention weights with broadcasting
        d = np.mean(weights.reshape((1,-1)) @ (np.abs(y.squeeze(1)-x) ** self.distance_measure).T, axis=1)
        # take root of the distance measure
        d = d ** (1 / self.distance_measure)
        # compute similarity
        s = np.exp(-sensitivity * d)

        return s
    
    def compute_category_probabilities(self, s, b):
        """ compute probabilities for categories given the similarity and bias 

        args:
        s: similarity
        b: bias
        
        return:
        p: probability of each category
        """
        p = np.exp(b * s) / np.sum(np.exp(b * s))
        
        return p