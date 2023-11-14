import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
import numpy as np
import torch
import statsmodels.discrete.discrete_model as sm
import ipdb

#prototype stimuli are set by the experimenter depending on the type of rule used for categorisation
    
class PrototypeModel():
    """ Prototype model for categorisation task """
    
    def __init__(self, prototypes=None, num_features=3, distance_measure=1, num_iterations=1):
        
        self.bounds = [(0, 20), # sensitivity
                       (0, 1), # bias
                       ]     
        self.weight_bound = [(0, 1)] # weights
        self.distance_measure = distance_measure  
        self.num_iterations = num_iterations
        self.num_features = num_features
        self.prototypes = [np.ones(num_features) * 0.5, np.ones(num_features) * 0.5] if prototypes is None else prototypes

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
                choice = categories[df_trial.choice.item()]
            
                true_choice = categories[df_trial.correct_choice.item()]

                # load num features of the current stimuli
                current_stimuli = df_trial[['feature{}'.format(i+1) for i in range(self.num_features)]].values
                
                # given stimuli and list of objects seen so far within cateogry return probablity the object belongs to each category 
                ll += self.prototype_model(params, current_stimuli, stimuli_seen, choice)

                # update stimuli seen
                stimuli_seen[true_choice].append(current_stimuli)
    
        return -ll
    
    def compute_nll_transfer(self, params, df_train, df_transfer):
        """ compute negative log likelihood of the data given the parameters

        args:
        params: parameters of the model
        df_train: dataframe containing the training data
        df_transfer: dataframe containing the transfer data

        returns:
        negative log likelihood of the data given the parameters
        """
       
        ll = 0.
        num_categories = df_train['category'].nunique()
        stimuli_seen = [df_train[df_train['category'] == i][['x{}'.format(i+1) for i in range(self.num_features)]].values for i in range(num_categories)]
        stimuli_seen = [np.expand_dims(stimuli_seen[i], axis=1) for i in range(num_categories)]

        for trial_id in df_transfer.trial_id.values:
            df_trial = df_transfer[(df_transfer['trial_id'] == trial_id)]
            choice = df_trial['category'].item()
            current_stimuli = df_trial[['x{}'.format(i+1) for i in range(self.num_features)]].values
            ll += self.prototype_model(params, current_stimuli, stimuli_seen, choice)

        return -2*ll
        
    def benchmark(self, df_train, df_transfer):
        """ fit pm to training data and transfer to new data 
        
        args:
        df_train: dataframe containing the training data
        df_transfer: dataframe containing the transfer data
        
        returns:
        nll: negative log likelihood for each participant"""

        self.bounds.extend(self.weight_bound * self.num_features)
        constraint_obj = {'type': 'eq', 'fun': self.constraint}
        result = minimize(
                fun=self.compute_nll_transfer,
                x0=[np.random.uniform(x, y) for x, y in self.bounds],
                args=(df_train, df_transfer),
                bounds=self.bounds,
                constraints=constraint_obj
            )
        if result.success:
            print("The optimiser converged successfully.")
        else:
            Warning("The optimiser did not converge.")

        log_likelihood = -self.compute_nll_transfer(result.x, df_train, df_transfer)/2
        r2 = 1 - (log_likelihood/(len(df_transfer)*np.log(1/2)))
        print(f'fitted log-likelihood: {log_likelihood}')
        print(f'fitted pseudo-r2: {r2} \n')

        return result.x

    def prototype_model(self, params, current_stimuli, stimuli_seen, choice):
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
        num_categories = len(self.prototypes)
        category_similarity = np.zeros(num_categories)
       
        for for_category in range(num_categories):
            if len(stimuli_seen[for_category]) == 0:
                # if no stimuli seen yet within category, similarity is set to unseen similarity
                category_similarity[for_category] = bias
            else:
                # compute attention weighted similarity measure
                
                category_similarity[for_category] = self.compute_attention_weighted_similarity(current_stimuli, np.stack(self.prototypes[for_category]), (weights, sensitivity))

        # compute category probabilities
        category_probabilities = self.compute_category_probabilities(category_similarity, bias)
        
        # compute log likelihood
        epsilon = 1e-10
        log_likelihood = np.log(category_probabilities[choice]+epsilon)

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
        d = np.mean(weights.reshape((1,-1)) @ (np.abs(y-x) ** self.distance_measure).T, axis=1)
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
        
        assert len(s) == 2, "number of categories must be 2"
        weighted_similarities = np.array([b, 1-b]) * s
        epsilon = 1e-10
        sum_weighted_similarities = np.sum(weighted_similarities)
        p = weighted_similarities / (sum_weighted_similarities + epsilon)

        return p