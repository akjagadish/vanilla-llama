import pandas as pd
from pm import PrototypeModel


# df = pd.read_csv('exp1.csv')
# pm = PrototypeModel(num_features=3, distance_measure=1, num_iterations=1)
# ll, r2 = pm.fit_participants(df)

# print(f'mean log-likelihood across participants: {ll.mean()} \n')
# print(f'mean pseudo-r2 across participants: {r2.mean()}')


## benchmarking pm model
df_train = pd.read_csv('../data/human/akshay-benchmark-across-languages-train.csv')
df_transfer = pd.read_csv('../data/human/akshay-benchmark-across-languages-transfer.csv')
pm = PrototypeModel(num_features=2, distance_measure=1, num_iterations=1)
params = pm.benchmark(df_train, df_transfer)
print('fitted parameters: c {}, bias {}, w1 {}, w2 {}'.format(*params))
true_params = pd.read_csv('../data/human/akshay-benchmark-across-languages-params.csv')
print('true parameters: ', true_params)