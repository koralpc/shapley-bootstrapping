"""
Author : Koralp Catalsakal
Date : 12/02/2020
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def assign_instances(X,X_train,X_val,y,shapley_values):

    X = pd.DataFrame(X)

    X_instanced = pd.concat((X,pd.DataFrame(columns = ['instance'])),axis = 1)
    X_instanced['instance'].iloc[X_train.index] = 'train'
    X_instanced['instance'].iloc[X_val.index] = 'val'
    y_instanced = pd.concat((pd.DataFrame(y),pd.DataFrame(columns = ['instance'])),axis = 1)
    y_instanced['instance'].iloc[X_train.index] = 'train'
    y_instanced['instance'].iloc[X_val.index] = 'val'


    #Gather shapley values and output values in one dataframe
    shap_instanced = pd.concat((shapley_values,pd.DataFrame(columns = ['instance'])),axis = 1)
    shap_instanced['instance'].loc[X_train.index] = 'train'
    shap_instanced['instance'].loc[X_val.index] = 'val'
    return X_instanced,y_instanced,shap_instanced

def dimensional_reduce(method,data):

    transformed_data = method.fit_transform(data)
    try:
        explained_variance = method.explained_variance_ratio_.sum()
    except AttributeError:
        explained_variance = 0
    return transformed_data,explained_variance

def prepare_pipeline_data(X,y,test_proportion = 0.25):
    X = preprocessing.StandardScaler().fit_transform(X)
    #y = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
    X = pd.DataFrame(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = test_proportion,random_state = 42)
    X_train.reset_index(inplace = True)
    X_train.drop(['index'],axis = 1,inplace = True)
    X_test.reset_index(inplace = True)
    X_test.drop(['index'],axis = 1,inplace = True)

    return X_train,X_test,y_train,y_test


def prepare_pipeline_reduced_data(X,y,method,test_proportion = 0.2,validation_proportion = 0.25):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=0)

    X_train.reset_index(inplace = True)
    X_train.drop(['index'],axis = 1,inplace = True)
    X_test.reset_index(inplace = True)
    X_test.drop(['index'],axis = 1,inplace = True)

    X_train,_ = dimensional_reduce(method,X_train)
    X_test,_ = dimensional_reduce(method,X_test)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))

    scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))

    y_train = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1,)

    X_train_tr,X_train_val,y_train_tr,y_train_val = train_test_split(X_train,y_train,test_size = validation_proportion,random_state = 0)

    return X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test

def reduce_after_clusters(method,data):
    data_attributes = data.iloc[:,:-2]
    transformed_data = method.fit_transform(data_attributes)
    transformed_data = pd.DataFrame(transformed_data)
    try:
        explained_variance = method.explained_variance_ratio_.sum()
    except AttributeError:
        explained_variance = 0
    try:
        transformed_data['instance'] = data['instance']
    except KeyError:
        pass
    transformed_data['cluster'] = data['cluster']
    return transformed_data,explained_variance

def clear_nan(X,y):
    if y is not None:
        X = pd.DataFrame(X)
        X.reset_index(inplace = True)

        X.drop(['index'],axis = 1,inplace = True)
        ## Dropping NA values
        drop_indexes = np.unique(np.where(X.isna())[0])
        X.dropna(inplace = True)
        mask = np.ones(len(y),dtype = bool)
        mask[drop_indexes] = False
        y = y[mask]
        return X,y
    else:
        X = pd.DataFrame(X)
        X.reset_index(inplace = True)

        X.drop(['index'],axis = 1,inplace = True)
        X.dropna(inplace = True)
        return X,None


def get_branch_best(dataframe,*branchs):
    branch_output = []
    for branch in branchs:
        branch_mins = dataframe.groupby('Dataset_name').agg({branch : min})
        branch_output.append(branch_mins)
    return branch_output


def set_branch_best(dataframe,*branchs):
    branch_best_performance = get_branch_best(dataframe,*branchs)
    dataframe_temp = dataframe.copy()
    for i,branch in enumerate(branchs):
        branch_best = branch_best_performance[i]
        for j in range(len(dataframe_temp['Dataset_name'])):
            dataframe_temp[branch].iloc[j] = branch_best.loc[dataframe_temp['Dataset_name'].iloc[j]][0]
    return dataframe_temp


#def calculate_percentage_improvement(dataframe,*branchs)
