
from sklearn.linear_model import LinearRegression
import xgboost
import shap
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing,impute
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator


class ProcessingBlock(BaseEstimator,RegressorMixin):

    def __init__(self):
        #print('Processing Block Constructed')
        self.X_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()
        self.imputer = impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    def fit(self,X,y):
        self.X = X
        self.y = y
        return self

    def split_data(self,X,y = None,test_split = 0.2):

        #X = self.X_scaler.fit_transform(X)
        #y = self.y_scaler.fit_transform(y.reshape(-1,1))

        #X = self.imputer.fit_transform(X)
        #y = self.imputer.fit_transform(y.reshape(-1,1))
        if y is not None:
            X_df =  pd.DataFrame(X)
            X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=test_split, random_state=0)

            self.train_idx = X_train.index
            self.val_idx = X_val.index

            #X_train.reset_index(inplace = True)
            #X_train = X_train.drop(['index'],axis = 1)
            #X_val.reset_index(inplace = True)
            #X_val = X_val.drop(['index'],axis = 1)

            return X_train,X_val,y_train,y_val
        else:
            X_df =  pd.DataFrame(X)
            X_train, X_val = train_test_split(X_df, test_size=test_split, random_state=0)

            self.train_idx = X_train.index
            self.val_idx = X_val.index

            #X_train.reset_index(inplace = True)
            #X_train = X_train.drop(['index'],axis = 1)
            #X_val.reset_index(inplace = True)
            #X_val = X_val.drop(['index'],axis = 1)

            return X_train,X_val

    def transform(self,X,y):
        pass

    def predict(self,X,y):
        pass


    def impute_data(self,X_test,y_test):
        X_test = self.imputer.fit_transform(X_test)
        #X_test = self.X_scaler.transform(X_test)
        y_test = self.imputer.fit_transform(y_test.reshape(-1,1))
        #y_test = self.y_scaler.fit_transform(y_test)
        X_test =  pd.DataFrame(X_test)
        return X_test,y_test


    def impute_test_data(self,X_test):
        X_test = self.imputer.fit_transform(X_test)
        X_test =  pd.DataFrame(X_test)
        return X_test

class ExplainerBlock(BaseEstimator,RegressorMixin):


    def __init__(self,explainer_type,params = None,kwargs = None):

        #print('Shapley Explainer Constructed')
        self.explainer_type = explainer_type
        self.eval_results = {}
        self.base_model = None

        if params is None:
            self.explainer_params = {
                "eta": 0.05,
                "max_depth": 3,
                "objective": "reg:squarederror",
                "subsample": 0.7,
                "eval_metric": "rmse",
                "lambda" : 0.1
            }
        else:
            self.explainer_params = params

        if kwargs is None:
            self.keyword_args =  {
                'num_boost_round':5000,
                'verbose_eval': 0,
                'evals_result' : {},
                'early_stopping_rounds' : 200
            }
        else:
            self.keyword_args = kwargs

    def fit(self,X_exp,y_exp,X_train,y_train,X_val,y_val):

        if self.explainer_type == 'Linear':
            self.base_model = LinearRegression().fit(X_exp, y_exp)
        else:
            eval = [(xgboost.DMatrix(X_train, label=y_train),"train"),(xgboost.DMatrix(X_val, label=y_val),"val"),]
            self.base_model = xgboost.train(self.explainer_params, xgboost.DMatrix(X_train, label=y_train),evals = eval,**self.keyword_args)

        if self.explainer_type == 'Linear':
            self.explainer = shap.LinearExplainer(self.base_model,X_exp,feature_dependence = 'independent')
        else:
            self.explainer = shap.TreeExplainer(self.base_model)

        return self

    def transform(self,X):
        shapley_values = self.explainer.shap_values(X)
        return shapley_values

    def fit_transform(self,X,y,X_train,y_train,X_val,y_val):
        self.fit(X,y,X_train,y_train,X_val,y_val)
        shapley_values = self.transform(X)
        return shapley_values

    def predict(self,X):
        if self.explainer_type == 'Linear':
            y_pred = self.base_model.predict(X)
        if self.explainer_type == 'XGBoost':
            y_pred = self.base_model.predict(xgboost.DMatrix(X))
        return y_pred

class ClusterBlock(BaseEstimator,RegressorMixin):

    def __init__(self,nClusters,training_set_model,test_set_model):
        #print('Clustering Block Constructed')
        self.n_clusters = nClusters
        self.training_set_model = training_set_model
        self.test_set_model = test_set_model

    def fit(self,X,y):
        self.X = X
        self.y = y
        return self

    def transform(self,X):
        print('transform')
        pass

    def cluster_training_instances(self,X):
        self.training_set_model.fit(X)
        return self.training_set_model.labels_

    def cluster_test_instances(self,X,X_test):
        self.test_set_model.fit(X,self.training_set_model.labels_)
        prediction = self.test_set_model.predict(X_test)
        return prediction

class EnsembleBlock(BaseEstimator,RegressorMixin):

    def __init__(self,model_type,params = None,keyword_args = None):
        #print('Ensemble Models Constructed')
        self.eval_dict = {}
        self.model_dict = {}
        self.model_type = model_type

        if params is None:
            self.ensemble_params = {
                    "eta": 0.05,
                    "max_depth": 3,
                    "objective": "reg:squarederror",
                    "subsample": 0.7,
                    "eval_metric": "rmse",
                    "lambda" : 0.1
                }
        else:
            self.ensemble_params = params

        if keyword_args is None:
            self.keyword_args = {
                    'num_boost_round': 5000,
                    'verbose_eval': 0,
                    'evals_result' : {},
                    'early_stopping_rounds' : 200
                }
        else:
            self.keyword_args = keyword_args


    def fit(self,X,y):
        pass

    def train(self,X_train,X_val,y_train,y_val,cluster_labels):
        if self.model_type == 'Linear':
            for i in range(len(np.unique(cluster_labels))):
                c_idx = cluster_labels == i
                X_train_cluster = X_train[c_idx[X_train.index]]
                y_train_cluster = y_train[c_idx[X_train.index]]
                X_val_cluster = X_val[c_idx[X_val.index]]
                y_val_cluster = y_val[c_idx[X_val.index]]
                self.model_dict['model{0}'.format(i)] = LinearRegression().fit(X_train_cluster,y_train_cluster)
                #self.eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,X_val_cluster,y_val_cluster)}}

        if self.model_type == 'XGBoost':
            self.keyword_args['evals_result'] = {}
            for i in range(len(np.unique(cluster_labels))):
                c_idx = cluster_labels == i
                X_train_cluster = X_train[c_idx[X_train.index]]
                y_train_cluster = y_train[c_idx[X_train.index]]
                X_val_cluster = X_val[c_idx[X_val.index]]
                y_val_cluster = y_val[c_idx[X_val.index]]
                if (not y_val_cluster.size == 0):
                    dtrain = xgboost.DMatrix(X_train_cluster,label = y_train_cluster)
                    eval = [(xgboost.DMatrix(X_train_cluster,label =  y_train_cluster), "train"),
                        (xgboost.DMatrix(X_val_cluster,label =  y_val_cluster), "val")]
                    self.model_dict['model{0}'.format(i)] = xgboost.train(self.ensemble_params,dtrain,evals = eval,**self.keyword_args)
                    self.eval_dict['eval{0}'.format(i)] = self.keyword_args['evals_result']
                else:
                    dtrain = xgboost.DMatrix(X_train_cluster,label = y_train_cluster)
                    eval = [(xgboost.DMatrix(X_train_cluster,label =  y_train_cluster), "train")]
                    self.model_dict['model{0}'.format(i)] = xgboost.train(self.ensemble_params,dtrain,evals = eval,**self.keyword_args)
                    self.eval_dict['eval{0}'.format(i)] = self.keyword_args['evals_result']



    def predict(self,X_test,cluster_labels):
        y_pred = np.zeros(shape = (X_test.shape[0],))
        for i in range(len(np.unique(cluster_labels))):
            if self.model_type == 'Linear':
                    y_pred[cluster_labels == i]  = self.model_dict['model{0}'.format(i)].predict(X_test[cluster_labels == i]).reshape(-1,)
            else:
                    y_pred[cluster_labels == i] = self.model_dict['model{0}'.format(i)].predict(xgboost.DMatrix(X_test[cluster_labels == i]))

        return y_pred

class ReduceBlock(BaseEstimator,RegressorMixin):

    def __init__(self,reduce_model):
        #print('Dimensionality Reduction Block Constructed')
        self.reduce_model = reduce_model


    def fit(self,X):
        self.reduce_model.fit(X)

    def transform(self,X):
        X_reduced = self.reduce_model.transform(X)
        return X_reduced


    def fit_transform(self,X):
        self.fit(X)
        X_reduced = self.transform(X)
        return X_reduced
