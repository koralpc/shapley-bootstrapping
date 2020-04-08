import sys
import os
sys.path.append("..")
import shap
import xgboost
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import colorsys
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from Framework.models import ShapleyModel
from Framework import cluster
from sklearn.cluster import KMeans
from Framework import datasets
from Framework import utils
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from Framework.building_blocks import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

class CustomPipelineModel(BaseEstimator,RegressorMixin):

    """
    :description : Pipeline model for thesis
    """

    def __init__(self,notebook_mode,explainer_type,ensemble_type,nClusters):
        self.description = 'Pipeline model for thesis'
        self.notebook_mode = notebook_mode
        self.explainer_type = explainer_type
        self.ensemble_type = ensemble_type
        self.nClusters = nClusters
        self.shap_model = ShapleyModel(explainer_type,ensemble_type,nClusters,notebook_mode)
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.pipeline = Pipeline(steps=[('Shapley-Blackbox',self)])

    def fit(self,X,y):
        X_train,shap_dataframe,original_split,y_org,original_split_shapley,y_shap,kmeans_original,kmeans_shapley = self.prepareData(X,y)

        self.original_labels = kmeans_original.labels_
        self.shapley_labels = kmeans_shapley.labels_
        self.X = X_train
        self.shap_values = shap_dataframe
        self.model_dict_shapley,self.eval_results_shapley = self.shap_model.trainPredictor(original_split_shapley,y_shap)
        #Use if task is classification
        self.classes_ = np.unique(y)
        return self

    def predict(self,X):

        #X,_ = processing.clear_nan(X,None)
        X = self.imputer.fit_transform(X)
        X = self.X_scaler.transform(X)
        shapley_test = self.shap_model.predictShapleyValues(X)
        shapley_test_df = pd.DataFrame(shapley_test)

        data_dict_shapley_test= self.shap_model.clusterDataTest(self.shap_values,self.shapley_labels,shapley_test_df)
        original_split_shapley_test = self.shap_model.prepareTestData(data_dict_shapley_test,X,shapley_test_df)

        predictions = original_split_shapley_test.apply(lambda x : self.shap_model.predictRow(x,self.model_dict_shapley),axis = 1)
        return predictions

    def prepareData(self,X,y):

        X,y = utils.clear_nan(X,y)
        X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test,self.X_scaler = utils.prepare_pipeline_data(X,y)

        shap_values = self.shap_model.explainShapley(X_train,y_train,X_train_tr,y_train_tr,X_train_val,y_train_val)
        shap_dataframe = pd.DataFrame(data = shap_values)

        X_instanced,y_instanced,shap_instanced = utils.assign_instances(X_train,X_train_tr,X_train_val,y_train,shap_dataframe)

        #Cluster the data
        data_dict,data_dict_original,kmeans_shapley,kmeans_original = self.shap_model.clusterData(X_train,shap_dataframe)
        y_train_df = pd.DataFrame(y)

        original_split,y_org,original_split_shapley,y_shap = self.shap_model.prepareTrainData(data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,True)

        return X_train,shap_dataframe,original_split,y_org,original_split_shapley,y_shap,kmeans_original,kmeans_shapley

#Will be added in case of classification task addition
"""    def predict_proba(self,X):
        proba = self.predict(X)

        proba = proba[:, :len(self.classes_)]
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
"""

class BuildingBlockPipeline(BaseEstimator,RegressorMixin):

    def __init__(self,processing_block,explainer_block,cluster_block,ensemble_block):
        #self.description = 'Pipeline model for thesis'
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.n_estimators = 0
        self.n_avg_nodes = 0

    def fit(self,X,y):
        pass

    def predict(self,X,y):
        pass

    def calculate_complexity(self):
        models = self.ensemble_block.model_dict
        if models :
            self.param_sum =  sum([models['model{}'.format(i)].trees_to_dataframe().shape[0] for i in range(len(models))])
            self.n_estimators = sum([np.unique(models['model{}'.format(i)].trees_to_dataframe()['Tree']).size for i in range(len(models))])
            self.n_avg_nodes = self.param_sum/self.n_estimators
        elif self.explainer_block.base_model is not None:
            self.param_sum = self.explainer_block.base_model.trees_to_dataframe().shape[0]
            self.n_estimators = np.unique(self.explainer_block.base_model.trees_to_dataframe()['Tree']).size
            self.n_avg_nodes = self.param_sum/self.n_estimators
        else:
            self.param_sum = 0
            self.n_estimators = 0
            self.n_avg_nodes = 0
        return self.param_sum,self.n_estimators,self.n_avg_nodes

class S1_Branch_Pipeline(BuildingBlockPipeline):

    def __init__(self,processing_block,explainer_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Explainer_model -> Prediction'
        self.tag = 'S1'
        #self.pipeline = Pipeline(steps=[('Building-Box-Model',self)])

    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        X = pd.DataFrame(X)
        self.explainer_block.fit(X,y,X_train,y_train,X_val,y_val)

    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        #X = pd.DataFrame(X)
        y_pred = self.explainer_block.predict(X)
        return y_pred

class S2_Branch_Pipeline(BuildingBlockPipeline):

    def __init__(self,processing_block,explainer_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Cluster -> Ensemble -> Prediction'
        self.tag = 'S2'

    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        self.X_train = X
        self.y_train = y
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        cluster_labels = self.cluster_block.cluster_training_instances(X)
        X_train.columns = X.columns
        X_val.columns = X.columns
        self.ensemble_block.train(X_train,X_val,y_train,y_val,cluster_labels)


    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        cluster_labels_test = self.cluster_block.cluster_test_instances(self.X_train,X)
        y_pred = self.ensemble_block.predict(X,cluster_labels_test)
        return y_pred

class S6_Branch_Pipeline(BuildingBlockPipeline):

    def __init__(self,processing_block,explainer_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        params = {
                    "eta": 0.1,
                    "max_depth": 1,
                    "objective": "reg:squarederror",
                    "subsample": 0.75,
                    "eval_metric": "rmse",
                    "lambda" : 0.1
        }
        self.explainer_block.keyword_args =  {
            'num_boost_round':200,
            'verbose_eval': 0,
            'evals_result' : {},
            'early_stopping_rounds' : 20
        }
        self.explainer_block.explainer_params = params
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Explainer -> Shapley -> Cluster -> Ensemble -> Prediction'
        self.tag = 'S6'

    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        #X = pd.DataFrame(X)
        self.X_train = X
        self.y_train = y
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        self.shapley_values = self.explainer_block.fit_transform(X,y,X_train,y_train,X_val,y_val)
        cluster_labels = self.cluster_block.cluster_training_instances(self.shapley_values)
        X_train.columns = X.columns
        X_val.columns = X.columns
        shapley_train = pd.DataFrame(self.shapley_values[X_train.index])
        shapley_val = pd.DataFrame(self.shapley_values[X_val.index])
        self.ensemble_block.train(shapley_train,shapley_val,y_train,y_val,cluster_labels)


    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        #X = pd.DataFrame(X)
        shapley_values_test = self.explainer_block.transform(X)
        cluster_labels_test = self.cluster_block.cluster_test_instances(self.shapley_values,shapley_values_test)
        shapley_values_test = pd.DataFrame(shapley_values_test)
        y_pred = self.ensemble_block.predict(shapley_values_test,cluster_labels_test)
        return y_pred

class S7_Branch_Pipeline(BuildingBlockPipeline):
    def __init__(self,processing_block,explainer_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        params = {
                    "eta": 0.1,
                    "max_depth": 1,
                    "objective": "reg:squarederror",
                    "subsample": 0.75,
                    "eval_metric": "rmse",
                    "lambda" : 0.1
        }
        self.explainer_block.keyword_args =  {
            'num_boost_round':200,
            'verbose_eval': 0,
            'evals_result' : {},
            'early_stopping_rounds' : 20
        }
        self.explainer_block.explainer_params = params
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Explainer -> Shapley -> Cluster ->Map Original Data -> Ensemble -> Prediction'
        self.tag = 'S7'

    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        #X = pd.DataFrame(X)
        self.X_train = X
        self.y_train = y
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        self.shapley_values = self.explainer_block.fit_transform(X,y,X_train,y_train,X_val,y_val)
        cluster_labels = self.cluster_block.cluster_training_instances(self.shapley_values)
        X_train.columns = X.columns
        X_val.columns = X.columns
        self.ensemble_block.train(X_train,X_val,y_train,y_val,cluster_labels)


    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        #X = pd.DataFrame(X)
        shapley_values_test = self.explainer_block.transform(X)
        cluster_labels_test = self.cluster_block.cluster_test_instances(self.shapley_values,shapley_values_test)
        y_pred = self.ensemble_block.predict(X,cluster_labels_test)
        return y_pred

class S10_Branch_Pipeline(BuildingBlockPipeline):

    def __init__(self,processing_block,explainer_block,reduce_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        params = {
                    "eta": 0.1,
                    "max_depth": 1,
                    "objective": "reg:squarederror",
                    "subsample": 0.75,
                    "eval_metric": "rmse",
                    "lambda" : 0.1
        }
        self.explainer_block.keyword_args =  {
            'num_boost_round':200,
            'verbose_eval': 0,
            'evals_result' : {},
            'early_stopping_rounds' : 20
        }
        self.explainer_block.explainer_params = params
        self.reduce_block = reduce_block
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Explainer -> Shapley -> Reduced-Shapley -> Cluster -> Ensemble -> Prediction'
        self.tag = 'S10'


    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        #X = pd.DataFrame(X)
        self.X_train = X
        self.y_train = y
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        self.shapley_values = self.explainer_block.fit_transform(X,y,X_train,y_train,X_val,y_val)
        self.shapley_values_reduced = self.reduce_block.fit_transform(self.shapley_values)
        cluster_labels = self.cluster_block.cluster_training_instances(self.shapley_values_reduced)
        X_train.columns = X.columns
        X_val.columns = X.columns
        shapley_train = pd.DataFrame(self.shapley_values_reduced[X_train.index])
        shapley_val = pd.DataFrame(self.shapley_values_reduced[X_val.index])
        self.ensemble_block.train(shapley_train,shapley_val,y_train,y_val,cluster_labels)


    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        #X = pd.DataFrame(X)
        shapley_values_test = self.explainer_block.transform(X)
        shapley_values_test_reduced = self.reduce_block.fit_transform(shapley_values_test)
        cluster_labels_test = self.cluster_block.cluster_test_instances(self.shapley_values_reduced,shapley_values_test_reduced)
        shapley_values_test_reduced = pd.DataFrame(shapley_values_test_reduced)
        y_pred = self.ensemble_block.predict(shapley_values_test_reduced,cluster_labels_test)
        return y_pred

class S11_Branch_Pipeline(BuildingBlockPipeline):

    def __init__(self,processing_block,explainer_block,reduce_block,cluster_block,ensemble_block):
        self.processing_block = processing_block
        self.explainer_block = explainer_block
        params = {
                    "eta": 0.1,
                    "max_depth": 1,
                    "objective": "reg:squarederror",
                    "subsample": 0.75,
                    "eval_metric": "rmse",
                    "lambda" : 0.1
        }
        self.explainer_block.keyword_args =  {
            'num_boost_round':200,
            'verbose_eval': 0,
            'evals_result' : {},
            'early_stopping_rounds' : 20
        }
        self.explainer_block.explainer_params = params
        self.reduce_block = reduce_block
        self.cluster_block = cluster_block
        self.ensemble_block = ensemble_block
        self.param_sum = 0
        self.description = 'Data -> Explainer -> Shapley -> Reduced-Shapley -> Cluster ->Map Original Data -> Ensemble -> Prediction'
        self.tag = 'S11'


    def fit(self,X,y):
        X,y = self.processing_block.impute_data(X,y)
        #X = pd.DataFrame(X)
        self.X_train = X
        self.y_train = y
        X_train,X_val,y_train,y_val = self.processing_block.split_data(X,y,test_split=0.15)
        self.shapley_values = self.explainer_block.fit_transform(X,y,X_train,y_train,X_val,y_val)
        self.shapley_values_reduced = self.reduce_block.fit_transform(self.shapley_values)
        cluster_labels = self.cluster_block.cluster_training_instances(self.shapley_values_reduced)
        X_train.columns = X.columns
        X_val.columns = X.columns
        self.ensemble_block.train(X_train,X_val,y_train,y_val,cluster_labels)


    def predict(self,X):
        X = self.processing_block.impute_test_data(X)
        #X = pd.DataFrame(X)
        shapley_values_test = self.explainer_block.transform(X)
        shapley_values_test_reduced = self.reduce_block.fit_transform(shapley_values_test)
        cluster_labels_test = self.cluster_block.cluster_test_instances(self.shapley_values_reduced,shapley_values_test_reduced)
        y_pred = self.ensemble_block.predict(X,cluster_labels_test)
        return y_pred
