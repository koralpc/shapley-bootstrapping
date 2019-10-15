"""
Author : Koralp Catalsakal
Date : 15/10/2019
"""


def clusterData(model_func = None, data = None , label = None):

    if label == None:
        model_func(data)
    else:
        model_func(data,label)
