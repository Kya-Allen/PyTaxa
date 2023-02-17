import numpy as np
import pandas as pd
import math
import itertools

def RMSE(data1, data2) -> float:
    squared_errors = []
    for point1, point2 in zip(data1, data2):
        error = point1 - point2
        squared_error = error ** 2
        squared_errors.append(squared_error)
    
    mse = mean(np.array(squared_errors))
    rmse = math.sqrt(mse)

    return

def categorical_fit(empirical_curve, simulated_curve) -> float:
    return RMSE(empirical_curve, simulated_curve)

def dimensional_fit(empirical_curve, simulated_curve) -> float:
    return  RMSE(empirical_curve, simulated_curve)

def comparison_curve_fit_index(empirical_curve, categorical_simulation, dimensional_simulation) -> float:
    fit_dim = dimensional_fit(empirical_curve, dimensional_simulation)
    fit_cat = categorical_fit(empirical_curve, categorical_simulation)
    index = fit_dim / (fit_dim + fit_cat)

    return index

def mean(single_dimension) -> float:
    n = single_dimension.size
    summation = np.sum(single_dimension)

    return summation / n

def mean_difference(sample1, sample2) -> float:
    return abs(mean(sample1) - mean(sample2))
    

class Model():
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix
        self.curve = []

class MAXEIG(Model):
# Maximum Eigenvalue

    def fit(self, partition_variable, covariance_matrix, num_partitions):
        
        return

class MAMBAC(Model):
# Mean Above Minus Below A cut.

    def fit(self, partition_variable, comparison_variable, num_partitions): #where the number of partitions is the number of divisions, not the number of groups
        partition_space = np.max(partition_variable) - np.min(partition_variable)
        sorted_data = sorted(zip(partition_variable, comparison_variable), key = lambda x: x[0])
        partition_interval = partition_space / num_partitions + 1
        results = []
        partition_pointer: float = 0
        for partition in num_partitions:
            partition_pointer += partition_interval
            index_pointer: int = 0
            while sorted_data[index_pointer][0] <= partition_pointer:
                index_pointer += 1
            result = mean_difference(sorted_data[0:index_pointer][1], sorted_data[index_pointer:][1])
            results.append(result)
        
        self.curve = results
        return self.curve

    def evaluate(self, categorical_simulation, dimensional_simulation):
        self.ccfi = comparison_curve_fit_index(self.curve, categorical_simulation, dimensional_simulation)
        return self.ccfi


def full_mambac(data_matrix, num_partitions):
    indices = [x for x in range(0, len(data_matrix))]
    pairwise_variables = list(itertools.permutations(indices, 2))
    curve_matrix = []
    for pair in pairwise_variables:
        pair_model = MAMBAC(data_matrix)
        pair_model.fit(data_matrix[pair[0]], data_matrix[pair[1]], num_partitions)
        curve_matrix.append(pair_model.curve)

    curve_matrix = np.array(curve_matrix)
    averaged_curve = curve_matrix.mean(axis=0)
    return averaged_curve

def k_fold_maxeig():
    return

def evaluate_model(empirical_curve, categorical_simulation, dimensional_simulation):
    return comparison_curve_fit_index(empirical_curve, categorical_simulation, dimensional_simulation)

plist = [2, 7, 3, 9, 25]
list = list(itertools.permutations(plist, 2))
print(list)