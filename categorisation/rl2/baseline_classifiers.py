import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# binary classifier class for logistic regression using sklearn
class LogisticRegressionModel:
    # take more inputs for different parameters
    def __init__(self, X, y):
        # self.model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        self.model = LogisticRegression(penalty='l2', C=1.0, solver='sag', max_iter=20000)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

# binary classifier class for support vector machines using sklearn
class SVMModel:
    def __init__(self, X, y):
        #self.model = SVC(probability=True)
        # svc model with linear kernel
        #self.model = SVC(kernel='linear', probability=True)
        # svc model with rbf kernel
        self.model = SVC(kernel='rbf', probability=True, class_weight='balanced', shrinking=False, tol=1e-3)
        # poly kernel
        #self.model = SVC(kernel='poly', degree=3, probability=True)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
# benchmark baseline models on 2 dimensional llm generated data
def benchmark_baseline_models_2d(num_tasks, data):
    performance = []
    num_tasks = num_tasks if num_tasks<data.TaskID.max() else data.TaskID.max()
    for task_id in range(1,num_tasks+1):
        X = np.vstack([data[data.TaskID==task_id].Input1, data[data.TaskID==task_id].Input2]).T
        assert X.shape[0]>0, "task {} has no data".format(task_id)
        y = data[data.TaskID==task_id].Target
        lr = LogisticRegressionModel(X, y)
        svm = SVMModel(X,y)
        performance.append([lr.score(X, y), svm.score(X, y)])
    return np.stack(performance)

# benchmark baseline models only regex parsed dataset which might include no samples from other classes or wierd datapoints etc
def benchmark_baseline_models_regex_parsed(num_tasks, data):
    performance = []
    num_tasks = range(1,int(num_tasks)+1) if num_tasks<data.task_id.max() else data.task_id.unique()
    for task_id in num_tasks:
        X = np.stack([eval(val) for val in data[data.task_id==task_id].input.values])
        assert X.shape[0]>0, "task {} has no data".format(task_id)
        y = np.stack([val for val in data[data.task_id==task_id].target.values])
        try:
            lr = LogisticRegressionModel(X, y)
            svm = SVMModel(X,y)
            performance.append([lr.score(X, y), svm.score(X, y)])
        except:
            print("task {} failed".format(task_id))
    return np.stack(performance)

# benchmark baseline models only regex parsed dataset which might include no samples from other classes or wierd datapoints etc
def benchmark_baseline_models_regex_parsed_random_points(data, dim=3):
    performance = []
    num_tasks = int(data.task_id.max()+1)
    num_points  = int(np.array([data[data.task_id==ii].trial_id.max()+1 for ii in np.arange(num_tasks)]).mean())
    for task_id in range(num_tasks):
        X = np.random.rand(num_points, dim) #inputs[samples]
        y = np.random.choice(['A', 'B'], size=num_points) #targets[samples]
        try:
            lr = LogisticRegressionModel(X, y)
            svm = SVMModel(X,y)
            performance.append([lr.score(X, y), svm.score(X, y)])
        except:
            print("task {} failed".format(task_id))
    return np.stack(performance)