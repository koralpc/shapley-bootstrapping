#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.manifold import TSNE
from sklearn import preprocessing
from Framework.models import ShapleyModel
from Framework import cluster
from sklearn.cluster import KMeans
from Framework import datasets
from Framework import processing


# In[2]:


print(os.getcwd())

shap.initjs()

notebook_mode = sys.argv[1]
explainer_type = sys.argv[2]
model_type = sys.argv[3]
nClusters = int(sys.argv[4])
dataset_count = int(sys.argv[5])

X,y,name = datasets.returnDataset(dataset_count)

#X_train_pca,explained_var = processing.dimensional_reduce(PCA(n_components = 3),X_train)
#X_test_pca,_ = processing.dimensional_reduce(PCA(n_components = 3),X_test)

blackbox_model = ShapleyModel(explainer_type,model_type,nClusters,notebook_mode)

X_train_pca,X_train_tr_pca,X_train_val_pca,X_test_pca,y_train_pca,y_train_tr_pca,y_train_val_pca,y_test_pca = processing.prepare_pipeline_reduced_data(X,y,PCA(2))
X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test = processing.prepare_pipeline_data(X,y)

shap_values = blackbox_model.explainShapley(X_train,y_train,X_train_tr,y_train_tr,X_train_val,y_train_val)
shap_dataframe = pd.DataFrame(data = shap_values,columns = X_train.columns)

shap_dataframe_pca,explained_var_shap = processing.dimensional_reduce(PCA(n_components = 2),shap_dataframe)
shap_dataframe_tsne,explained_var_tsne = processing.dimensional_reduce(TSNE(n_components=2),shap_dataframe)

X_instanced = pd.concat((X_train_pca,pd.DataFrame(columns = ['instance'])),axis = 1)
X_instanced['instance'].iloc[X_train_tr.index] = 'train'
X_instanced['instance'].iloc[X_train_val.index] = 'val'
y_instanced = pd.concat((pd.DataFrame(y_train_pca,columns=['label']),pd.DataFrame(columns = ['instance'])),axis = 1)
y_instanced['instance'].iloc[X_train_tr.index] = 'train'
y_instanced['instance'].iloc[X_train_val.index] = 'val'

#Gather shapley values and output values in one dataframe

shap_instanced = pd.concat((pd.DataFrame(shap_dataframe_pca),pd.DataFrame(columns = ['instance'])),axis = 1)
shap_instanced['instance'].loc[X_train_tr.index] = 'train'
shap_instanced['instance'].loc[X_train_val.index] = 'val'


#Split the clusters into a dictionary
data_dict,data_dict_original,kmeans_shapley,kmeans_original = blackbox_model.clusterData(X_train_pca,shap_dataframe_pca)

y_train_df = pd.DataFrame(y_train)
original_split,y_org,original_split_shapley,y_shap = blackbox_model.prepareTrainData(data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,False)



model_dict,eval_results = blackbox_model.trainPredictor(original_split,y_org)
model_dict_shapley,eval_results_shapley = blackbox_model.trainPredictor(original_split_shapley,y_shap)


# In[10]:


shapley_test = blackbox_model.predictShapleyValues(X_test)
shap_test_pca,_ = processing.dimensional_reduce(PCA(n_components = 2),shapley_test)
shapley_test_df = pd.DataFrame(shap_test_pca,columns = X_train_pca.columns)
data_dict_shapley_test= blackbox_model.clusterDataTest(shap_dataframe_pca,kmeans_shapley.labels_,shapley_test_df)
data_dict_original_test= blackbox_model.clusterDataTest(X_train_pca,kmeans_original.labels_,X_test_pca)


# In[11]:


y_test_df = pd.DataFrame(y_test_pca)
original_split_test,y_test_org,original_split_shapley_test,y_test_shap = blackbox_model.prepareTrainData(data_dict_shapley_test,data_dict_original_test,X_test_pca,y_test_df,shapley_test_df,True)


preds_org = blackbox_model.predict(original_split_test,model_dict)
tot_rmse_org = blackbox_model.evaluate(preds_org,y_test_org)


# In[14]:


preds_shap = blackbox_model.predict(original_split_shapley_test,model_dict_shapley)
tot_rmse_shap = blackbox_model.evaluate(preds_shap,y_test_shap)


# In[15]:


y_train_tr = pd.DataFrame(y_train_tr_pca,index = X_train_tr_pca.index)
y_train_val = pd.DataFrame(y_train_val_pca,index = X_train_val_pca.index)
X_train_tr_pca.sort_index(inplace = True)
y_train_tr.sort_index(inplace = True)
X_train_val_pca.sort_index(inplace = True)
y_train_val.sort_index(inplace = True)

exp_model = blackbox_model.trainExtraModel(X_train_tr_pca,y_train_tr.iloc[:,0],X_train_val_pca,y_train_val.iloc[:,0])
preds_big = blackbox_model.predict(X_test_pca,exp_model)
tot_rmse_big = blackbox_model.evaluate(preds_big,y_test_pca)


# In[17]:


names = ['Whole model','Original_ensemble','Shapley_ensemble']
values = [tot_rmse_big,tot_rmse_org[0],tot_rmse_shap[0]]
f=open("../Data/test_pca.txt", "a+")
f.write(name + ',')
[f.write('{0:.3f},'.format(values[v])) for v in range(len(values))]
f.write('{0},'.format(nClusters))
f.write('{0},'.format(notebook_mode))
f.write('{0},'.format(explainer_type))
f.write('{0}'.format(model_type))
f.write('\n')
f.close()
