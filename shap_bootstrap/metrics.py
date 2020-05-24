"""
Author : Koralp Catalsakal
Date : 30/10/2019

"""
import numpy as np

def ensembleRMSE(sample_sizes,rmse_list):
    """
    This metric correctly calculates the overall rmse value for an ensemble of trained models, with each model
    outputting an individual rmse value over it's respective training set.

    Args:
        sample_sizes(list): A list of length of the respective datasets of each individual model
            to be trained on

        rmse_list(list): A list of rmse values of each individual model in the ensemble

    Returns:
        actual_rmse(Double) : Returns the calculated ensemble rmse value.


    """
    total = 0
    for s in range(len(sample_sizes)):
        total += sample_sizes[s] * (rmse_list[s]**2)
    total = total / sum(sample_sizes)
    return np.sqrt(total)
