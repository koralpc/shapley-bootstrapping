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
        #model = CustomPipelineModel(notebook_mode,explainer_type,ensemble_type,nClusters)
        self.pipeline = Pipeline(steps=[('Shapley-Blackbox',self)])

    def fit(self,X,y):
        X_train,shap_dataframe,original_split,y_org,original_split_shapley,y_shap,kmeans_original,kmeans_shapley = self.prepareData(X,y)

        #model_dict,eval_results = blackbox_model.trainPredictor(original_split,y_org)
        self.original_labels = kmeans_original.labels_
        self.shapley_labels = kmeans_shapley.labels_
        self.X = X_train
        self.shap_values = shap_dataframe
        self.model_dict_shapley,self.eval_results_shapley = self.shap_model.trainPredictor(original_split_shapley,y_shap)

        return self

    def predict(self,X):

        X = self.X_scaler.transform(X)
        shapley_test = self.shap_model.predictShapleyValues(X)
        shapley_test_df = pd.DataFrame(shapley_test)
        print('Labels:' , self.shapley_labels)
        data_dict_shapley_test= self.shap_model.clusterDataTest(self.shap_values,self.shapley_labels,shapley_test_df)
        #data_dict_original_test= self.shap_model.clusterDataTest(self.X,self.original_labels,X)

        print(data_dict_shapley_test)
        original_split_shapley_test = self.shap_model.prepareTestData(data_dict_shapley_test,X,shapley_test_df)
        y_pred = self.shap_model.predict(original_split_shapley_test,self.model_dict_shapley)
        predictions = original_split_shapley_test.apply(lambda x : self.shap_model.predictRow(x,self.model_dict_shapley),axis = 1)
        return predictions


    def prepareData(self,X,y):

        X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test,self.X_scaler = processing.prepare_pipeline_data(X,y)

        # In[5]:

        shap_values = self.shap_model.explainShapley(X_train,y_train,X_train_tr,y_train_tr,X_train_val,y_train_val)

        # In[6]:
        X_instanced = pd.concat((X_train,pd.DataFrame(columns = ['instance'])),axis = 1)
        X_instanced['instance'].iloc[X_train_tr.index] = 'train'
        X_instanced['instance'].iloc[X_train_val.index] = 'val'
        y_instanced = pd.concat((pd.DataFrame(y_train),pd.DataFrame(columns = ['instance'])),axis = 1)
        y_instanced['instance'].iloc[X_train_tr.index] = 'train'
        y_instanced['instance'].iloc[X_train_val.index] = 'val'


        #Gather shapley values and output values in one dataframe
        shap_dataframe = pd.DataFrame(data = shap_values)
        shap_instanced = pd.concat((shap_dataframe,pd.DataFrame(columns = ['instance'])),axis = 1)
        shap_instanced['instance'].loc[X_train_tr.index] = 'train'
        shap_instanced['instance'].loc[X_train_val.index] = 'val'

        #Split the clusters into a dictionary
        data_dict,data_dict_original,kmeans_shapley,kmeans_original = self.shap_model.clusterData(X_train,shap_dataframe)
        y_train_df = pd.DataFrame(y_train)

        original_split,y_org,original_split_shapley,y_shap = self.shap_model.prepareTrainData(data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,True)

        return X_train,shap_dataframe,original_split,y_org,original_split_shapley,y_shap,kmeans_original,kmeans_shapley
