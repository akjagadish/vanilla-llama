import openml
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import numpy as np
benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite
Xs = []
ys = []
counter = 0
counter2 = 0
for task_id in  benchmark_suite.tasks:  # iterate over all tasks:
    counter += 1
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    if (len(task.class_labels) == 2):
        features, targets = task.get_X_and_y()  # get the data
        if (features.shape[1] < 100) and (not np.isnan(features).any()):
        # if  not np.isnan(features).any():
            counter += 1
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(features)
            features = scaler.transform(features)
            features = SelectKBest(f_classif, k=4).fit_transform(features, targets)
            # features = PCA(n_components=4).fit_transform(features)
            # clf = LogisticRegression(max_iter=1000).fit(features, targets)
            # y_pred = clf.predict(features)
            # acc = ((targets == y_pred).astype('float')).mean(0)
            # if acc > 0.8:
            print(task)
            Xs.append(features)
            print(features.max())
            print(features.min())
            print(features.shape)
            ys.append(targets)
            print()
            counter2 += 1
print(counter)
print(counter2)
# np.save('data.npy', [Xs, ys], allow_pickle=True)