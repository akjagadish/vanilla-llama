import pandas as pd
from pm import PrototypeModel


df = pd.read_csv('exp1.csv')
pm = PrototypeModel(num_features=3, distance_measure=1, num_iterations=1)
ll, r2 = pm.fit_participants(df)

print(f'mean log-likelihood across participants: {ll.mean()} \n')
print(f'mean pseudo-r2 across participants: {r2.mean()}')