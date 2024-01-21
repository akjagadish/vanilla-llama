from scipy.spatial.distance import euclidean
import numpy as np

def compute_distance_transfer_stimulus(idx=5):
    A_array = np.array([[0, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0]])

    B_array = np.array([[0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1]])

    T_array= np.array([[0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 1]])


    # Extracting the T5 value
    T5_value = T_array[idx-1]

    # Computing the mean element-by-element Euclidean distance of T5 with elements of A and B
    mean_distance_A = np.mean([euclidean(T5_value, A) for A in A_array])
    mean_distance_B = np.mean([euclidean(T5_value, B) for B in B_array])

    print(mean_distance_A, mean_distance_B)