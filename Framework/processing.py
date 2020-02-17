"""
Author : Koralp Catalsakal
Date : 12/02/2020
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def dimensional_reduce(method,data):

    transformed_data = method.fit_transform(data)
    explained_variance = method.explained_variance_ratio_.sum()
    return transformed_data,explained_variance



def prepare_pipeline_data(X,y,test_proportion = 0.2,validation_proportion = 0.25):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=0)

    X_train.reset_index(inplace = True)
    X_train.drop(['index'],axis = 1,inplace = True)
    X_test.reset_index(inplace = True)
    X_test.drop(['index'],axis = 1,inplace = True)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train),columns = X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test),columns = X.columns)

    scaler_y = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))

    y_train = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1,)

    X_train_tr,X_train_val,y_train_tr,y_train_val = train_test_split(X_train,y_train,test_size = validation_proportion,random_state = 0)

    return X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test
