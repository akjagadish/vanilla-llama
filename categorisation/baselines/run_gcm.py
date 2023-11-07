import pandas as pd
from gcm import GeneralizedContextModel


df = pd.read_csv('exp1.csv')
gcm = GeneralizedContextModel(num_features=3, distance_measure=1, num_iterations=1)
ll, r2 = gcm.fit_participants(df)

print(f'mean log-likelihood across participants: {ll.mean()} \n')
print(f'mean pseudo-r2 across participants: {r2.mean()}')