import numpy as np
import pandas as pd
import math

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

class MAMBAC(Model):
# Mean Above Minus Below A cut.

    def fit():
        return

def full_mambac(data_matrix):
    return

def evaluate_model(empirical_curve, categorical_simulation, dimensional_simulation):
    comparison_curve_fit_index(empirical_curve, categorical_simulation, dimensional_simulation)