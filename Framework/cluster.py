"""
Author : Koralp Catalsakal
Date : 15/10/2019
"""

import xgboost
import numpy
import pandas

def clusterData(model_func = None, data = None , label = None):

    """
    Model agnostic method to be used in clustering. Can be used to try different models over datasets.
    Default setup accepts unsupervised dataset.

    Args:
        model_func(Model.function) : The specific model function to be used. The result of the function is
            returned in the model

        data(numpy.ndarray or pandas.DataFrame) : Data to be used in training(Without the ground truth data)

        label(numpy.ndarray or pandas.DataFrame) : Labels(Ground truth) of the data argument.

    Returns:
        result(Model) : Returns the fitted/trained model as output


    """
    if label == None:
        result = model_func(data)
    else:
        result = model_func(data,label)
    return result

def splitDataLabeled(nClusters,data,cluster_labels):

    """

    """

    data_dict = {}
    for i in range(nClusters):
        if type(data) == numpy.ndarray:
            data_dict['cluster{0}'.format(i)] = pandas.DataFrame(data[cluster_labels == i])
        else:
            data_dict['cluster{0}'.format(i)] = data[cluster_labels == i]
    return data_dict

def convertOriginalData(data_dict,X,y):
    original_data_split = {}
    for i, (key, val) in enumerate(data_dict.items()):
        #print(val.index.values)
        original_data_split['original_data_cluster{0}'.format(i)] = X.iloc[val.index.values]
        temp = X.loc[original_data_split['original_data_cluster{0}'.format(i)].index]
        original_data_split['original_train_cluster{0}'.format(i)] = temp[temp['instance'] == 'train']
        original_data_split['original_test_cluster{0}'.format(i)] = temp[temp['instance'] == 'test']
        original_data_split['original_train_cluster{0}'.format(i)] = original_data_split['original_train_cluster{0}'.format(i)].drop(columns = 'instance')
        original_data_split['original_test_cluster{0}'.format(i)] = original_data_split['original_test_cluster{0}'.format(i)].drop(columns= 'instance')
        original_data_split['original_label_cluster{0}'.format(i)] = y.iloc[val.index.values]
        temp = y.loc[original_data_split['original_label_cluster{0}'.format(i)].index]
        original_data_split['original_train_label_cluster{0}'.format(i)] = temp[temp['instance'] == 'train']
        original_data_split['original_test_label_cluster{0}'.format(i)] = temp[temp['instance'] == 'test']
        original_data_split['original_train_label_cluster{0}'.format(i)] = original_data_split['original_train_label_cluster{0}'.format(i)].drop(columns = 'instance')
        original_data_split['original_test_label_cluster{0}'.format(i)] = original_data_split['original_test_label_cluster{0}'.format(i)].drop(columns = 'instance')
    return original_data_split

def trainMultipleModels(model_func,data_dict,option,params,**kwargs):

    model_dict = {}
    eval_dict = {}
    if option == 'XGBoost':
        for i in range(len(data_dict.items())//6):
            kwargs['evals_result'] = {}
            dtrain = xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)],label = data_dict['original_train_label_cluster{0}'.format(i)])
            eval = [(xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)], label=data_dict['original_train_label_cluster{0}'.format(i)]), "train"),(xgboost.DMatrix(data_dict['original_test_cluster{0}'.format(i)], label=data_dict['original_test_label_cluster{0}'.format(i)]), "test")]
            model_dict['model{0}'.format(i)] = model_func(params,dtrain,evals = eval,**kwargs)
            eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
    else:
        print( 'eeee')
        #for i, (k,v) in enumerate(data_dict.items()):
            #model_dict['model{0}'.format(i)] = model_func(kwargs['params'],data_dict['original_data_cluster{}'.format(i)], data_dict['original_label_cluster{}'.format(i)],kwargs)
    return model_dict,eval_dict
