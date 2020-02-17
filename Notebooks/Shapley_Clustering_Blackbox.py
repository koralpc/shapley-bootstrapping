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
from sklearn import preprocessing
from Framework.models import ShapleyModel
from Framework import cluster
from sklearn.cluster import KMeans
from Framework import datasets
from Framework import processing


print(os.getcwd())

# In[2]:


shap.initjs()


# In[3]:


notebook_mode = sys.argv[1]
explainer_type = sys.argv[2]
model_type = sys.argv[3]
nClusters = int(sys.argv[4])
dataset_count = int(sys.argv[5])

dataset_array = datasets.returnAlldatasets()
X,y,name = dataset_array[dataset_count]
dataset_name = name
X.reset_index(inplace = True)

X.drop(['index'],axis = 1,inplace = True)
if dataset_name == 'Nhanes':
    X.drop(['Unnamed: 0'],axis = 1,inplace = True)

## Dropping NA values
drop_indexes = np.unique(np.where(X.isna())[0])
X.dropna(inplace = True)
mask = np.ones(len(y),dtype = bool)
mask[drop_indexes] = False
y = y[mask]


blackbox_model = ShapleyModel(explainer_type,model_type,nClusters,notebook_mode)


X_train,X_train_tr,X_train_val,X_test,y_train,y_train_tr,y_train_val,y_test = processing.prepare_pipeline_data(X,y)

# In[5]:


shap_values = blackbox_model.explainShapley(X_train,y_train,X_train_tr,y_train_tr,X_train_val,y_train_val)


# In[6]:


X_instanced = pd.concat((X_train,pd.DataFrame(columns = ['instance'])),axis = 1)
X_instanced['instance'].iloc[X_train_tr.index] = 'train'
X_instanced['instance'].iloc[X_train_val.index] = 'test'
y_instanced = pd.concat((pd.DataFrame(y_train),pd.DataFrame(columns = ['instance'])),axis = 1)
y_instanced['instance'].iloc[X_train_tr.index] = 'train'
y_instanced['instance'].iloc[X_train_val.index] = 'test'


# In[7]:


#Gather shapley values and output values in one dataframe
shap_dataframe = pd.DataFrame(data = shap_values,columns = X.columns)


# In[8]:


shap_instanced = pd.concat((shap_dataframe,pd.DataFrame(columns = ['instance'])),axis = 1)
shap_instanced['instance'].loc[X_train_tr.index] = 'train'
shap_instanced['instance'].loc[X_train_val.index] = 'test'


# In[9]:


#Split the clusters into a dictionary
data_dict,data_dict_original,kmeans_shapley,kmeans_original = blackbox_model.clusterData(X_train,shap_dataframe)


# In[10]:


y_train_df = pd.DataFrame(y_train)
original_split,original_split_shapley = blackbox_model.prepareTrainData(data_dict,data_dict_original,X_instanced,y_instanced,shap_instanced,False)


# In[11]:


model_dict,eval_results = blackbox_model.trainPredictor(original_split)


# In[12]:


model_dict_shapley,eval_results_shapley = blackbox_model.trainPredictor(original_split_shapley)


# In[13]:


# In[14]:


shapley_test = blackbox_model.predictShapleyValues(X_test)
shapley_test_df = pd.DataFrame(shapley_test,columns = X.columns)
data_dict_shapley_test= blackbox_model.clusterDataTest(shap_values,kmeans_shapley.labels_,shapley_test_df)
data_dict_original_test= blackbox_model.clusterDataTest(X_train,kmeans_original.labels_,X_test)



# In[16]:


y_test_df = pd.DataFrame(y_test)
original_split_test,original_split_shapley_test = blackbox_model.prepareTrainData(data_dict_shapley_test,data_dict_original_test,X_test,y_test_df,shapley_test_df)


# In[17]:


preds_org = blackbox_model.predict(original_split_test,model_dict)
tot_rmse_org = blackbox_model.evaluate(preds_org,original_split_test)


# In[18]:


preds_shap = blackbox_model.predict(original_split_shapley_test,model_dict_shapley)
tot_rmse_shap = blackbox_model.evaluate(preds_shap,original_split_shapley_test)


# In[19]:

y_train_tr = pd.DataFrame(y_train_tr,index = X_train_tr.index)
y_train_val = pd.DataFrame(y_train_val,index = X_train_val.index)
X_train_tr.sort_index(inplace = True)
y_train_tr.sort_index(inplace = True)
X_train_val.sort_index(inplace = True)
y_train_val.sort_index(inplace = True)

exp_model = blackbox_model.trainExtraModel(X_train_tr,y_train_tr.iloc[:,0],X_train_val,y_train_val.iloc[:,0])
preds_big = blackbox_model.predict(X_test,exp_model)
tot_rmse_big = blackbox_model.evaluate(preds_big,y_test)


# In[21]:


names = ['Whole model','Original_ensemble','Shapley_ensemble']
values = [tot_rmse_big,tot_rmse_org[0],tot_rmse_shap[0]]
f=open("../Data/test_scaled.txt", "a+")
f.write(dataset_name + ',')
[f.write('{0:.3f},'.format(values[v])) for v in range(len(values))]
f.write('{0},'.format(nClusters))
f.write('{0},'.format(notebook_mode))
f.write('{0},'.format(explainer_type))
f.write('{0}'.format(model_type))
f.write('\n')
f.close()
