"""
Author : Koralp Catalsakal
Date : 25/11/2019
"""

import sys
sys.path.append("..")
import shap
import xgboost
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from Framework import cluster
from Framework import metrics
from sklearn.neighbors import KNeighborsClassifier

class ShapleyModel():

    def __init__(self,explainer_type,ensemble_type,nClusters,model_mode):
        self.explainer_type = explainer_type
        self.model_type = ensemble_type
        self.nClusters = nClusters
        self.notebook_mode = model_mode
        self.whole_model = None

    def explainShapley(self,X_exp,y_exp,X_train,y_train,X_val,y_val):
        params = {
            "eta": 0.01,
            "max_depth": 1,
            "objective": "reg:squarederror",
            "subsample": 0.5,
            "eval_metric": "rmse"
        }

        eval_results = {}
        kwargs = {
            'num_boost_round':500,
            'verbose_eval': 500,
            'evals_result' : {},
            'early_stopping_rounds' : 100
        }
        if self.explainer_type == 'Linear':
            whole_model = LinearRegression().fit(X_exp, y_exp)
        else:
            eval = [(xgboost.DMatrix(X_train, label=y_train),"train"),(xgboost.DMatrix(X_val, label=y_val),"test"),]
            whole_model = xgboost.train(params, xgboost.DMatrix(X_train, label=y_train),evals = eval,**kwargs)

        if self.explainer_type == 'Linear':
            self.explainer = shap.LinearExplainer(whole_model,X_exp)
        else:
            self.explainer = shap.TreeExplainer(whole_model)

        shap_values = self.explainer.shap_values(X_exp)

        return shap_values

    def clusterData(self,X,shap_dataframe):

        kmeans = cluster.clusterData(KMeans(n_clusters=self.nClusters, random_state=0).fit,shap_dataframe)
        kmeans_original = cluster.clusterData(KMeans(n_clusters=self.nClusters, random_state=0).fit,X)
        data_dict = cluster.splitDataLabeled(self.nClusters,shap_dataframe,kmeans.labels_)
        data_dict_original = cluster.splitDataLabeled(self.nClusters,X,kmeans_original.labels_)

        return data_dict,data_dict_original,kmeans,kmeans_original

    def clusterDataTest(self,data,labels,test_data):

        test_labels = cluster.mapTestToOriginal(data,labels,test_data)
        test_dict = cluster.splitDataLabeled(self.nClusters,test_data,test_labels)

        return test_dict

    def prepareTrainData(self,data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,no_val = True):

        if self.notebook_mode == 'Original' or self.notebook_mode == 'Original-PCA':
            split_data_original,y_org = cluster.convertOriginalData(data_dict_original,X_instanced,y_instanced,no_val)
            split_data_shapley,y_shap = cluster.convertOriginalData(data_dict,X_instanced,y_instanced,no_val)
        elif self.notebook_mode =='Shapley' or self.notebook_mode == 'Shapley-PCA':
            split_data_original,y_org = cluster.convertOriginalData(data_dict_original,X_instanced,y_instanced,no_val)
            split_data_shapley,y_shap = cluster.convertOriginalData(data_dict,shap_instanced,y_instanced,no_val)
        else:
            print(self.notebook_mode == 'Original')

        return split_data_original,y_org,split_data_shapley,y_shap

    def trainPredictor(self,X,y,no_val = True):

        params = {
            "eta": 0.01,
            "max_depth": 1,
            "objective": "reg:squarederror",
            "subsample": 0.5,
            "eval_metric": "rmse"
        }

        eval_results = {}
        kwargs = {
            'num_boost_round': 20000,
            'verbose_eval': 5000,
            'evals_result' : {},
            'early_stopping_rounds' : 100
        }

        if self.model_type == 'Linear':
            model_dict,eval_results = cluster.trainMultipleModels(LinearRegression().fit,X,y,'LinearRegressor',params,no_val = False)
        else:
            model_dict,eval_results = cluster.trainMultipleModels(xgboost.train,X,y,'XGBoost',params,no_val = False,**kwargs)

        return model_dict,eval_results

    def predict(self,data,models):
        if type(models) != dict:
            if self.explainer_type == 'Linear':
                preds = models.predict(data)
            else:
                preds = models.predict(xgboost.DMatrix(data))
        else:
            preds = {}
            if self.model_type == 'Linear':
                for i in range(len(models)):
                    preds['model{0}'.format(i)] = models['model{0}'.format(i)].predict(data[data['cluster'] == i].iloc[:,0:-1])
            else:
                for i in range(len(models)):
                    preds['model{0}'.format(i)] = models['model{0}'.format(i)].predict(xgboost.DMatrix(data[data['cluster'] == i].iloc[:,0:-1])).reshape(-1,1)
        return preds

    def predictShapleyValues(self,data):
        if self.explainer_type == 'Linear':
            shap_values = self.explainer.shap_values(data)
        else:
            shap_values = self.explainer.shap_values(xgboost.DMatrix(data))
        return shap_values

    def evaluate(self,predictions,target):
        sizes = []
        rmse_array = []
        if not (type(predictions) == dict):
            return np.sqrt(np.mean((predictions-target)**2))
        else:
            for i in range(len(predictions)):
                sizes.append(len(predictions['model{0}'.format(i)]))
                rmse_array.append(np.sqrt(np.mean((predictions['model{0}'.format(i)]-target[target['cluster'] == i].iloc[:,0:-1])**2)))
            total_rmse = metrics.ensembleRMSE(sizes,rmse_array)
            return total_rmse

    def trainExtraModel(self,X_train,y_train,X_val,y_val):
        params = {
            "eta": 0.01,
            "max_depth": 1,
            "objective": "reg:squarederror",
            "subsample": 0.5,
            "eval_metric": "rmse"
        }

        eval_results = {}
        kwargs = {
            'num_boost_round':20000,
            'verbose_eval': 5000,
            'evals_result' : {},
            'early_stopping_rounds' : 100
        }
        if self.explainer_type == 'Linear':
            model = LinearRegression().fit(X_train, y_train)
        else:
            eval = [(xgboost.DMatrix(X_train, label=y_train),"train"),(xgboost.DMatrix(X_val, label=y_val),"test")]
            model = xgboost.train(params, xgboost.DMatrix(X_train, label=y_train),evals = eval,**kwargs)
        return model
