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
    Concatenates cluster labels as an extra column to the existing DataFrame

    Args:

        data(pandas.DataFrame) : Data to be used in training(Without the ground truth data)

        cluster_labels(numpy.ndarray or pandas.DataFrame) : Cluster assignments of each index stored in array format.

    Returns:
        data_df(pandas.DataFrame) : Returns data with `cluster` column added


    """

    #data_dict = {}
    #for i in range(nClusters):
    if type(data) == numpy.ndarray:
        data_df = pandas.concat((pandas.DataFrame(data),pandas.DataFrame(cluster_labels,columns=['cluster'])),axis = 1)
        #data_df.cluster.mask(cluster_labels == i ,other = i,inplace = True)
        #data_dict['cluster{0}'.format(i)] = data_df[cluster_labels == i]
        #print(cluster_labels == i)
    else:
        data_df = pandas.concat((data,pandas.DataFrame(cluster_labels,columns=['cluster'])),axis = 1)
        #data_dict['cluster{0}'.format(i)] = data[cluster_labels == i]
        #data_df.cluster.loc[cluster_labels == i] = i
        #data_df.cluster.mask(cluster_labels == i ,other = i,inplace = True)
        #print(cluster_labels == i)
    return data_df

def convertOriginalData(data_dict,X,y,no_val = False):

    """
    Adds the train/validation set split assignment to the DataFrame.

    Args:
        data_dict(pandas.DataFrame) : Data with cluster assignments

        X(pandas.DataFrame) : DataFrame that holds indexes of training and validation instances

        y(pandas.DataFrame) : Target labels corresponding to training and validation instances

        no_val(bool) : default is `False`. If `False`, all instances are used as training instances, and no validation split is used. Otherwise data is split

    Returns:
        data_new(pandas.DataFrame) : training data with train/validation split instances

        data_new(pandas.DataFrame) : label data with train/validation split instances
    """



    """
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
    """
    if no_val:
        data_new = X.copy()
        data_new['cluster'] = data_dict['cluster']
        y_new = y.copy()
        y_new['cluster'] = data_dict['cluster']
    else:
        data_new = X.copy()
        data_new['cluster'] = data_dict['cluster']
        #data_dict['instance'] = X['instance']
        y_new = y.copy()
        y_new['cluster'] = data_dict['cluster']
        y_new['instance'] = X['instance']

    return data_new,y_new

def trainMultipleModels(model_func,X,y,option,params,no_val = False,**kwargs):

    model_dict = {}
    eval_dict = {}
    if no_val:
        if option == 'XGBoost':
            kwargs['evals_result'] = {}
            """for i in range(len(data_dict)//2):
                kwargs['evals_result'] = {}
                dtrain = xgboost.DMatrix(data_dict['cluster{0}'.format(i)],label = data_dict['label_cluster{0}'.format(i)])
                eval = [(xgboost.DMatrix(data_dict['cluster{0}'.format(i)], label=data_dict['label_cluster{0}'.format(i)]), "train")]
                model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
            """
            for i in range(len(numpy.unique(X['cluster']))):
                dtrain = xgboost.DMatrix(X[X['cluster'] == i].iloc[:,0:-2],label = y[y['cluster'] == i].iloc[:,0])
                eval = [(xgboost.DMatrix(X[X['cluster'] == i].iloc[:,0:-2],label = y[y['cluster'] == i].iloc[:,0]), "train")]
                model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                eval_dict['eval{0}'.format(i)] = kwargs['evals_result']

        elif option == 'LinearRegressor':
            for i in range(len(numpy.unique(X['cluster']))):
                model_dict['model{0}'.format(i)] = LinearRegression().fit(X[X['cluster'] == i].iloc[:,0:-2],y[y['cluster'] == i].iloc[:,0],**kwargs)
                eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,X[X['cluster'] == i].iloc[:,0:-2],y[y['cluster'] == i].iloc[:,0])}}
    else:
        if option == 'XGBoost':
            """for i in range(len(data_dict.items())//6):
                kwargs['evals_result'] = {}
                dtrain = xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)],label = data_dict['original_train_label_cluster{0}'.format(i)])
                eval = [(xgboost.DMatrix(data_dict['original_train_cluster{0}'.format(i)], label=data_dict['original_train_label_cluster{0}'.format(i)]), "train"),(xgboost.DMatrix(data_dict['original_test_cluster{0}'.format(i)], label=data_dict['original_test_label_cluster{0}'.format(i)]), "test")]
                model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
            """
            kwargs['evals_result'] = {}
            for i in range(len(numpy.unique(X['cluster']))):
                dtrain = xgboost.DMatrix(X[(X['cluster'] == i) & (X['instance'] == 'train')].iloc[:,0:-2],label =  y[(y['cluster'] == i) & (y['instance'] == 'train')].iloc[:,0])
                if not y[(y['cluster'] == i) & (y['instance'] == 'val')].iloc[:,0].empty:
                    eval = [(xgboost.DMatrix(X[(X['cluster'] == i) & (X['instance'] == 'train')].iloc[:,0:-2],label =  y[(y['cluster'] == i) & (y['instance'] == 'train')].iloc[:,0]), "train"),
                        (xgboost.DMatrix(X[(X['cluster'] == i) & (X['instance'] == 'val')].iloc[:,0:-2],label =  y[(y['cluster'] == i) & (y['instance'] == 'val')].iloc[:,0]), "val")]
                    model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                    eval_dict['eval{0}'.format(i)] = kwargs['evals_result']
                else:
                    eval = [(xgboost.DMatrix(X[(X['cluster'] == i) & (X['instance'] == 'train')].iloc[:,0:-2],label =  y[(y['cluster'] == i) & (y['instance'] == 'train')].iloc[:,0]), "train")]
                    model_dict['model{0}'.format(i)] = xgboost.train(params,dtrain,evals = eval,**kwargs)
                    eval_dict['eval{0}'.format(i)] = kwargs['evals_result']

        elif option == 'LinearRegressor':
            for i in range(len(numpy.unique(X['cluster']))):
                model_dict['model{0}'.format(i)] = LinearRegression().fit(X[(X['cluster'] == i) & (X['instance'] == 'train')].iloc[:,0:-2], y[(y['cluster'] == i) & (y['instance'] == 'train')].iloc[:,0],**kwargs)
                eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,X[(X['cluster'] == i) & (X['instance'] == 'train')].iloc[:,0:-2], y[(y['cluster'] == i) & (y['instance'] == 'train')].iloc[:,0])},
                'val': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,X[(X['cluster'] == i) & (X['instance'] == 'val')].iloc[:,0:-2],label =  y[(y['cluster'] == i) & (y['instance'] == 'val')].iloc[:,0])}}

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
