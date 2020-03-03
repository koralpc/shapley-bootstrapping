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
from Framework import processing
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer


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


#Will be added in case of classification task addition
"""    def predict_proba(self,X):
        proba = self.predict(X)

        proba = proba[:, :len(self.classes_)]
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
"""


    def prepareData(self,X,y):

        X,y = processing.clear_nan(X,y)
        X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test,self.X_scaler = processing.prepare_pipeline_data(X,y)

        shap_values = self.shap_model.explainShapley(X_train,y_train,X_train_tr,y_train_tr,X_train_val,y_train_val)
        shap_dataframe = pd.DataFrame(data = shap_values)

        X_instanced,y_instanced,shap_instanced = processing.assign_instances(X_train,X_train_tr,X_train_val,y_train,shap_dataframe)

        #Cluster the data
        data_dict,data_dict_original,kmeans_shapley,kmeans_original = self.shap_model.clusterData(X_train,shap_dataframe)
        y_train_df = pd.DataFrame(y)

        original_split,y_org,original_split_shapley,y_shap = self.shap_model.prepareTrainData(data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,True)

        return X_train,shap_dataframe,original_split,y_org,original_split_shapley,y_shap,kmeans_original,kmeans_shapley
