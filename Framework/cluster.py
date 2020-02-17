"""
Author : Koralp Catalsakal
Date : 15/10/2019
"""

import xgboost
import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

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
            data_df = pandas.DataFrame(data)
            data_dict['cluster{0}'.format(i)] = data_df[cluster_labels == i]
        else:
            data_dict['cluster{0}'.format(i)] = data[cluster_labels == i]
    return data_dict

def convertOriginalData(data_dict,X,y,no_val = False):
    original_data_split = {}
    if no_val:
        for i, (key, val) in enumerate(data_dict.items()):
            #print(val.index.values)
            original_data_split['cluster{0}'.format(i)] = X.iloc[val.index.values]
            #original_data_split['cluster{0}'.format(i)] = original_data_split['cluster{0}'.format(i)].drop(columns = 'instance')

            original_data_split['label_cluster{0}'.format(i)] = y.iloc[val.index.values]
            #original_data_split['label_cluster{0}'.format(i)] = original_data_split['label_cluster{0}'.format(i)].drop(columns = 'instance')
    else:
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

def trainMultipleModels(model_func,data_dict,option,params,no_val = False,**kwargs):

    model_dict = {}
    eval_dict = {}
    if no_val:
        if option == 'XGBoost':
            for i in range(len(data_dict)//2):
                kwargs['evals_result'] = {}
                dtrain = xgboost.DMatrix(data_dict['cluster{0}'.format(i)],label = data_dict['label_cluster{0}'.format(i)])
                eval = [(xgboost.DMatrix(data_dict['cluster{0}'.format(i)], label=data_dict['label_cluster{0}'.format(i)]), "train")]
                model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
        elif option == 'LinearRegressor':
            for i in range(len(data_dict)//2):
                model_dict['model{0}'.format(i)] = LinearRegression().fit(data_dict['cluster{0}'.format(i)], data_dict['label_cluster{0}'.format(i)],**kwargs)
                eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,data_dict['cluster{0}'.format(i)],data_dict['label_cluster{0}'.format(i)])}}
    else:
        if option == 'XGBoost':
            for i in range(len(data_dict.items())//6):
                kwargs['evals_result'] = {}
                dtrain = xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)],label = data_dict['original_train_label_cluster{0}'.format(i)])
                eval = [(xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)], label=data_dict['original_train_label_cluster{0}'.format(i)]), "train"),(xgboost.DMatrix(data_dict['original_test_cluster{0}'.format(i)], label=data_dict['original_test_label_cluster{0}'.format(i)]), "test")]
                model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
        elif option == 'LinearRegressor':
            for i in range(len(data_dict.items())//6):
                model_dict['model{0}'.format(i)] = LinearRegression().fit(data_dict['original_train_cluster{0}'.format(i)], data_dict['original_train_label_cluster{0}'.format(i)],**kwargs)
                eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,data_dict['original_train_cluster{0}'.format(i)],data_dict['original_train_label_cluster{0}'.format(i)])},
                'test': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,data_dict['original_test_cluster{0}'.format(i)],data_dict['original_test_label_cluster{0}'.format(i)])}}

    return model_dict,eval_dict

def mapDictToArray(cluster_dict,target_array):
    test_dict = {}
    for i in range(len(cluster_dict)):
        test_dict['cluster{0}'.format(i)] = target_array[cluster_dict['cluster{0}'.format(i)].index]
    return test_dict

def mapTestToOriginal(data,labels,test_data,via = 'Shapley'):
    knn = KNeighborsClassifier(n_neighbors=5).fit(data,labels)
    test_labels = knn.predict(test_data)
    return test_labels

def _calculate_accuracy(func,X_test,y_test):
    return numpy.sqrt(numpy.mean((func(X_test) - y_test)**2))
