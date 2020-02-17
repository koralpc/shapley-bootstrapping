#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
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


shap.initjs()


notebook_mode = sys.argv[1]
explainer_type = sys.argv[2]
model_type = sys.argv[3]
dataset_name = sys.argv[4]
nClusters = int(sys.argv[5])
X,y = shap.datasets.diabetes()
X = X.reset_index()


# train XGBoost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LinearRegression
if explainer_type == 'Linear':
    model = LinearRegression().fit(X, y)
else:
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)


# In[5]:


X_instanced = pd.concat((X,pd.DataFrame(columns = ['instance'])),axis = 1)
X_instanced['instance'].loc[X_train.index] = 'train'
X_instanced['instance'].loc[X_test.index] = 'test'
y_instanced = pd.concat((pd.DataFrame(y),pd.DataFrame(columns = ['instance'])),axis = 1)
y_instanced['instance'].loc[X_train.index] = 'train'
y_instanced['instance'].loc[X_test.index] = 'test'


# In[6]:


# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
if explainer_type == 'Linear':
    explainer = shap.LinearExplainer(model,X)
else:
    explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_instanced = pd.concat((pd.DataFrame(shap_values),pd.DataFrame(columns = ['instance'])),axis = 1)
shap_instanced['instance'].loc[X_train.index] = 'train'
shap_instanced['instance'].loc[X_test.index] = 'test'


#Gather shapley values and output values in one dataframe
shap_dataframe = pd.DataFrame(data = shap_values,columns = X.columns)
output_dataframe = pd.DataFrame(data = y,columns = ['targets'])
shap_dataframe = pd.concat([shap_dataframe,output_dataframe],axis = 1)


# In[12]:


#Make interaction plot for all features
#shap_interaction_values = explainer.shap_interaction_values(X)
#shap.summary_plot(shap_interaction_values, X, max_display = 10)
shap_dataframe.head(20)
#sum(shap_dataframe.iloc[1,:-1])



#Start clustering
from Framework import cluster
from sklearn.cluster import KMeans
#nClusters = 1
#Train KMeans, because the data is (Regression data)
#kmeans = KMeans(n_clusters=3, random_state=0).fit(shap_values)
kmeans = cluster.clusterData(KMeans(n_clusters=nClusters, random_state=0).fit,shap_values)


# In[16]:


#Get the labels, concat into original data, and sor the labels for into cluster groups
shap_dataframe_labeled = pd.concat([shap_dataframe,pd.DataFrame(kmeans.labels_,columns =[ 'labels'])],axis = 1)
#shap_grouped = shap_dataframe_labeled.sort_values(['labels'])
X_labeled = pd.concat([X,shap_dataframe_labeled['labels']], axis = 1)
plt.scatter(np.linspace(0,len(X),len(X)),y,c = X_labeled['labels'])


# In[17]:


kmeans_original = cluster.clusterData(KMeans(n_clusters=nClusters, random_state=0).fit,X)
plt.scatter(np.arange(len(X)),y,c = kmeans_original.labels_)


# In[18]:


#Split the clusters into a dictionary
data_dict = cluster.splitDataLabeled(nClusters,shap_dataframe,shap_dataframe_labeled['labels'])
data_dict_original = cluster.splitDataLabeled(nClusters,X,kmeans_original.labels_)


# In[19]:


shap_instanced = pd.concat((pd.DataFrame(shap_values),pd.DataFrame(columns = ['instance'])),axis = 1)
shap_instanced['instance'].loc[X_train.index] = 'train'
shap_instanced['instance'].loc[X_test.index] = 'test'


if notebook_mode == 'Original' or notebook_mode == 'Original-NC':
    original_split = cluster.convertOriginalData(data_dict_original,X_instanced,y_instanced)
    original_split_shapley = cluster.convertOriginalData(data_dict,X_instanced,y_instanced)
elif notebook_mode =='Shapley' or notebook_mode == 'Shapley-NC':
    original_split = cluster.convertOriginalData(data_dict_original,X_instanced,y_instanced)
    original_split_shapley = cluster.convertOriginalData(data_dict,shap_instanced,y_instanced)
else:
    print('Error!')



#Train split XGBoost models over original data
params = {
    "eta": 0.05,
    "max_depth": 1,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "rmse"
}

eval_results = {}
kwargs = {
    'num_boost_round':10000,
    'verbose_eval': 1000,
    'evals_result' : {},
    'early_stopping_rounds' : 200
}
if model_type == 'Linear':
    model_dict,eval_results = cluster.trainMultipleModels(LinearRegression().fit,original_split,'LinearRegressor',params)
    for i in range(nClusters):
        eval_results['eval{0}'.format(i)]['test']['rmse'] = cluster._calculate_accuracy(model_dict['model{0}'.format(i)].predict,original_split['original_test_cluster{0}'.format(i)],original_split['original_test_label_cluster{0}'.format(i)])
else:
    model_dict,eval_results = cluster.trainMultipleModels(xgboost.train,original_split,'XGBoost',params,**kwargs)
#small_model_1 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster0'], label=original_split['original_label_cluster0']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster0'], label=original_split['original_label_cluster0']), "train")] ,verbose_eval = 1000)
#small_model_2 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster1'], label=original_split['original_label_cluster1']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster1'], label=original_split['original_label_cluster1']), "train")] ,verbose_eval = 1000)
#small_model_3 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster2'], label=original_split['original_label_cluster2']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster2'], label=original_split['original_label_cluster2']), "train")] ,verbose_eval = 1000)


# In[23]:


#Train split XGBoost models over original data
params = {
    "eta": 0.05,
    "max_depth": 1,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "rmse"
}
eval_results_shapley = {}
kwargs = {
    'num_boost_round':10000,
    'verbose_eval': 1000,
    'evals_result' : {},
    'early_stopping_rounds' : 200
}
if model_type == 'Linear':
    model_dict_shapley,eval_results_shapley = cluster.trainMultipleModels(LinearRegression().fit,original_split_shapley,'LinearRegressor',params)
    for i in range(nClusters):
        eval_results_shapley['eval{0}'.format(i)]['test']['rmse'] = cluster._calculate_accuracy(model_dict_shapley['model{0}'.format(i)].predict,original_split_shapley['original_test_cluster{0}'.format(i)],original_split_shapley['original_test_label_cluster{0}'.format(i)])
else:
    model_dict_shapley,eval_results_shapley = cluster.trainMultipleModels(xgboost.train,original_split_shapley,'XGBoost',params,**kwargs)
#small_model_shapley1 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster0'], label=original_split_shapley['original_label_cluster0']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster0'], label=original_split_shapley['original_label_cluster0']), "train")] ,verbose_eval = 1000)
#small_model_shapley2 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster1'], label=original_split_shapley['original_label_cluster1']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster1'], label=original_split_shapley['original_label_cluster1']), "train")] ,verbose_eval = 1000)
#small_model_shapley3 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster2'], label=original_split_shapley['original_label_cluster2']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster2'], label=original_split_shapley['original_label_cluster2']), "train")] ,verbose_eval = 1000)


#Train overall model
ev_result = {}
params = {
    "eta": 0.05,
    "max_depth": 1,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "rmse",
}
if model_type == 'Linear':
    model = LinearRegression().fit(X_train,y_train)
    ev_result ={'test': {'rmse': cluster._calculate_accuracy(model.predict,X_test,y_test)}}
    tot_result_big_model = ev_result['test']['rmse']
else:
    model = xgboost.train(params,xgboost.DMatrix(X_train, label=y_train),10000,evals = [(xgboost.DMatrix(X_train, label=y_train), "train"),(xgboost.DMatrix(X_test, label=y_test), "test")] ,verbose_eval = 1000,evals_result = ev_result,early_stopping_rounds = 200)
    tot_result_big_model = min(ev_result['test']['rmse'])



#Evaluation of results
from Framework import metrics
sizes = []
rmse_array = []
for i in range(nClusters):
    sizes.append(len(original_split['original_test_label_cluster{0}'.format(i)]))
    rmse_array.append(min(eval_results['eval{0}'.format(i)]['test']['rmse']))
tot_rmse_org = metrics.ensembleRMSE(sizes,rmse_array)

sizes = []
rmse_array = []
for i in range(nClusters):
    sizes.append(len(original_split_shapley['original_test_label_cluster{0}'.format(i)]))
    rmse_array.append(min(eval_results_shapley['eval{0}'.format(i)]['test']['rmse']))
tot_rmse_shap = metrics.ensembleRMSE(sizes,rmse_array)


# In[34]:


#Do PCA for dimensionality reduction
nPcaComponents = 2
pca = PCA(n_components=nPcaComponents)
shap_values_pca = pca.fit_transform(shap_values)
print(pca.explained_variance_ratio_.sum())


# In[35]:


#Do PCA for dimensionality reduction
nPcaComponents = 2
pca_org = PCA(n_components=nPcaComponents)
org_values_pca = pca_org.fit_transform(X)
print(pca_org.explained_variance_ratio_.sum())



# In[37]:


kmeans_pca = cluster.clusterData(KMeans(n_clusters=nPcaComponents, random_state=0).fit,shap_values_pca)
kmeans_org_pca = cluster.clusterData(KMeans(n_clusters=nPcaComponents, random_state=0).fit,org_values_pca)


# In[39]:


data_dict_shap_pca = cluster.splitDataLabeled(2,shap_values_pca,kmeans_pca.labels_)
data_dict_org_pca = cluster.splitDataLabeled(2,org_values_pca,kmeans_org_pca.labels_)


# In[40]:


pca_instanced = pd.concat((pd.DataFrame(org_values_pca),pd.DataFrame(columns = ['instance'])),axis = 1)
pca_instanced['instance'].loc[X_train.index] = 'train'
pca_instanced['instance'].loc[X_test.index] = 'test'
shap_pca_instanced = pd.concat((pd.DataFrame(shap_values_pca),pd.DataFrame(columns = ['instance'])),axis = 1)
shap_pca_instanced['instance'].loc[X_train.index] = 'train'
shap_pca_instanced['instance'].loc[X_test.index] = 'test'


# In[41]:


original_shap_pca = cluster.convertOriginalData(data_dict_shap_pca,shap_pca_instanced,y_instanced)
original_pca = cluster.convertOriginalData(data_dict_org_pca,pca_instanced,y_instanced)
original_pca


# In[42]:


#Train split XGBoost models over original data
params = {
    "eta": 0.05,
    "max_depth": 1,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "rmse"
}
eval_results_org_pca = {}
kwargs = {
    'num_boost_round':10000,
    'verbose_eval': 1000,
    'evals_result' : {},
    'early_stopping_rounds' : 200
}
if model_type == 'Linear':
    model_dic_org_pca,eval_results_org_pca = cluster.trainMultipleModels(LinearRegression().fit,original_pca,'LinearRegressor',params)
else:
    model_dict_org_pca,eval_results_org_pca = cluster.trainMultipleModels(xgboost.train,original_pca,'XGBoost',params,**kwargs)


# In[43]:


#Train split XGBoost models over original data
params = {
    "eta": 0.05,
    "max_depth": 1,
    "objective": "reg:squarederror",
    "subsample": 0.5,
    "base_score": np.mean(y),
    "eval_metric": "rmse"
}
eval_results_shap_pca = {}
kwargs = {
    'num_boost_round':10000,
    'verbose_eval': 1000,
    'evals_result' : {},
    'early_stopping_rounds' : 200
}
if model_type == 'Linear':
    model_dic_shap_pca,eval_results_shap_pca = cluster.trainMultipleModels(LinearRegression().fit,original_shap_pca,'LinearRegressor',params)
else:
    model_dict_shap_pca,eval_results_shap_pca = cluster.trainMultipleModels(xgboost.train,original_shap_pca,'XGBoost',params,**kwargs)


# In[44]:


sizes = []
rmse_array = []
for i in range(nPcaComponents):
    sizes.append(len(original_shap_pca['original_test_label_cluster{0}'.format(i)]))
    rmse_array.append(min(eval_results_shap_pca['eval{0}'.format(i)]['test']['rmse']))
tot_rmse_shap_pca = metrics.ensembleRMSE(sizes,rmse_array)

sizes = []
rmse_array = []
for i in range(2):
    sizes.append(len(original_pca['original_test_label_cluster{0}'.format(i)]))
    rmse_array.append(min(eval_results_org_pca['eval{0}'.format(i)]['test']['rmse']))
tot_rmse_org_pca = metrics.ensembleRMSE(sizes,rmse_array)

# In[46]:


ax = plt.figure()
plt.barh(['Whole model','Original_ensemble','Shapley_ensemble','Shapley_ensemble + PCA','Original_ensemble + PCA'],[tot_result_big_model,tot_rmse_org,tot_rmse_shap,tot_rmse_shap_pca,tot_rmse_org_pca])
plt.title('Test-RMSE value vs. Method used')
for i,v in enumerate([tot_result_big_model,tot_rmse_org,tot_rmse_shap,tot_rmse_shap_pca,tot_rmse_org_pca]):
    plt.text(v , i + .05, '{0:.3f}'.format(v), color='blue', fontweight='bold')


# In[47]:


names = ['Whole model','Original_ensemble','Shapley_ensemble','Shapley_ensemble + PCA','Original_ensemble + PCA']
values = [tot_result_big_model,tot_rmse_org,tot_rmse_shap,tot_rmse_shap_pca,tot_rmse_org_pca]
f=open("../Data/test_results.txt", "a+")
f.write(dataset_name + ',')
[f.write('{0:.3f},'.format(values[v])) for v in range(len(values))]
f.write('{0},'.format(notebook_mode))
f.write('{0},'.format(explainer_type))
f.write('{0}'.format(model_type))
f.write('\n')
f.close()


# In[ ]:





# In[ ]:




