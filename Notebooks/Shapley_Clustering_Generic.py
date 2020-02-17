#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


shap.initjs()


# In[ ]:


notebook_mode = sys.argv[1]
explainer_type = sys.argv[2]
model_type = sys.argv[3]
dataset_name = sys.argv[4]
nClusters = int(sys.argv[5])
print(notebook_mode)
#explainer_type = 'XGBoost'
#model_type = 'Linear'
#dataset_name = 'Boston'
#nClusters = 3
X,y = shap.datasets.communitiesandcrime()
X = X.reset_index()

# In[ ]:


# train XGBoost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.linear_model import LinearRegression
if explainer_type == 'Linear':
    model = LinearRegression().fit(X, y)
else:
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)


# In[ ]:


X_instanced = pd.concat((X,pd.DataFrame(columns = ['instance'])),axis = 1)
X_instanced['instance'].loc[X_train.index] = 'train'
X_instanced['instance'].loc[X_test.index] = 'test'
y_instanced = pd.concat((pd.DataFrame(y),pd.DataFrame(columns = ['instance'])),axis = 1)
y_instanced['instance'].loc[X_train.index] = 'train'
y_instanced['instance'].loc[X_test.index] = 'test'


# In[ ]:


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


# In[ ]:


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#shap.force_plot(explainer.expected_value, shap_values[1,:], X.iloc[1,:])


# In[ ]:


# visualize the training set predictions
#shap.force_plot(explainer.expected_value, shap_values, X)


# In[ ]:


# summarize the effects of all the features
#shap.summary_plot(shap_values, X,plot_type = 'bar')


# In[ ]:


# summarize the effects of all the features
#shap.summary_plot(shap_values, X)


# In[ ]:


#Gather shapley values and output values in one dataframe
shap_dataframe = pd.DataFrame(data = shap_values,columns = X.columns)
output_dataframe = pd.DataFrame(data = y,columns = ['targets'])
shap_dataframe = pd.concat([shap_dataframe,output_dataframe],axis = 1)


# In[ ]:


#Make interaction plot for all features
#shap_interaction_values = explainer.shap_interaction_values(X)
#shap.summary_plot(shap_interaction_values, X, max_display = 10)
shap_dataframe.head(20)
#sum(shap_dataframe.iloc[1,:-1])


# In[ ]:


X.head(20)


# In[ ]:


from Framework import plotHelper
#plotHelper.plotAllFeatures(shap_dataframe)


# In[ ]:


#Start clustering
from Framework import cluster
from sklearn.cluster import KMeans
#nClusters = 1
#Train KMeans, because the data is unsupervised(Regression data)
#kmeans = KMeans(n_clusters=3, random_state=0).fit(shap_values)
kmeans = cluster.clusterData(KMeans(n_clusters=nClusters, random_state=0).fit,shap_values)


# In[ ]:


#Get the labels, concat into original data, and sor the labels for into cluster groups
shap_dataframe_labeled = pd.concat([shap_dataframe,pd.DataFrame(kmeans.labels_,columns =[ 'labels'])],axis = 1)
#shap_grouped = shap_dataframe_labeled.sort_values(['labels'])
X_labeled = pd.concat([X,shap_dataframe_labeled['labels']], axis = 1)
#plt.scatter(np.linspace(0,len(X),len(X)),y,c = X_labeled['labels'])


# In[ ]:


kmeans_original = cluster.clusterData(KMeans(n_clusters=nClusters, random_state=0).fit,X)
#plt.scatter(np.arange(len(X)),y,c = kmeans_original.labels_)


# In[ ]:


#Split the clusters into a dictionary
data_dict = cluster.splitDataLabeled(nClusters,shap_dataframe,shap_dataframe_labeled['labels'])
data_dict_original = cluster.splitDataLabeled(nClusters,X,kmeans_original.labels_)


# In[ ]:


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
    print('Error')


# In[ ]:


#Scatter the groups again...
#plt.scatter(np.linspace(-1,1,len(original_label_cluster2)),cluster2_predictions)
#plt.scatter(np.arange(len(original_split['original_label_cluster0'])),original_split['original_label_cluster0'].iloc[:,0])
#plt.scatter(np.arange(len(original_split['original_label_cluster1'])),original_split['original_label_cluster1'].iloc[:,0])
#plt.scatter(np.arange(len(original_split['original_label_cluster2'])),original_split['original_label_cluster2'].iloc[:,0])


# In[ ]:


#plt.scatter(np.arange(len(original_split_shapley['original_test_label_cluster0'])),original_split_shapley['original_test_label_cluster0'].iloc[:,0])
#plt.scatter(np.arange(len(original_split_shapley['original_test_label_cluster1'])),original_split_shapley['original_test_label_cluster1'].iloc[:,0])
#plt.scatter(np.arange(len(original_split_shapley['original_test_label_cluster2'])),original_split_shapley['original_test_label_cluster2'].iloc[:,0])


# In[ ]:



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
else:
    model_dict,eval_results = cluster.trainMultipleModels(xgboost.train,original_split,'XGBoost',params,**kwargs)
#small_model_1 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster0'], label=original_split['original_label_cluster0']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster0'], label=original_split['original_label_cluster0']), "train")] ,verbose_eval = 1000)
#small_model_2 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster1'], label=original_split['original_label_cluster1']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster1'], label=original_split['original_label_cluster1']), "train")] ,verbose_eval = 1000)
#small_model_3 = xgboost.train(params,xgboost.DMatrix(original_split['original_data_cluster2'], label=original_split['original_label_cluster2']),20000,evals = [(xgboost.DMatrix(original_split['original_data_cluster2'], label=original_split['original_label_cluster2']), "train")] ,verbose_eval = 1000)


# In[ ]:


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
else:
    model_dict_shapley,eval_results_shapley = cluster.trainMultipleModels(xgboost.train,original_split_shapley,'XGBoost',params,**kwargs)
#small_model_shapley1 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster0'], label=original_split_shapley['original_label_cluster0']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster0'], label=original_split_shapley['original_label_cluster0']), "train")] ,verbose_eval = 1000)
#small_model_shapley2 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster1'], label=original_split_shapley['original_label_cluster1']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster1'], label=original_split_shapley['original_label_cluster1']), "train")] ,verbose_eval = 1000)
#small_model_shapley3 = xgboost.train(params,xgboost.DMatrix(original_split_shapley['original_data_cluster2'], label=original_split_shapley['original_label_cluster2']),20000,evals = [(xgboost.DMatrix(original_split_shapley['original_data_cluster2'], label=original_split_shapley['original_label_cluster2']), "train")] ,verbose_eval = 1000)


# In[ ]:


"""plt.plot(np.arange(len(eval_results['eval0']['test']['rmse'])),eval_results['eval0']['test']['rmse'])
plt.plot(np.arange(len(eval_results['eval1']['test']['rmse'])),eval_results['eval1']['test']['rmse'])
plt.plot(np.arange(len(eval_results['eval2']['test']['rmse'])),eval_results['eval2']['test']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval0']['test']['rmse'])),eval_results_shapley['eval0']['test']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval1']['test']['rmse'])),eval_results_shapley['eval1']['test']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval2']['test']['rmse'])),eval_results_shapley['eval2']['test']['rmse'])
plt.xlabel('Epoch count')
plt.ylabel('Test-RMSE')
plt.legend(['Model0','Model1','Model2','Model0_Shapley','Model1_Shapley','Model2_Shapley'])"""


# In[ ]:


"""plt.plot(np.arange(len(eval_results['eval0']['train']['rmse'])),eval_results['eval0']['train']['rmse'])
plt.plot(np.arange(len(eval_results['eval1']['train']['rmse'])),eval_results['eval1']['train']['rmse'])
plt.plot(np.arange(len(eval_results['eval2']['train']['rmse'])),eval_results['eval2']['train']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval0']['train']['rmse'])),eval_results_shapley['eval0']['train']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval1']['train']['rmse'])),eval_results_shapley['eval1']['train']['rmse'])
plt.plot(np.arange(len(eval_results_shapley['eval2']['train']['rmse'])),eval_results_shapley['eval2']['train']['rmse'])
plt.xlabel('Epoch count')
plt.ylabel('Train-RMSE')
plt.legend(['Model0','Model1','Model2','Model0_Shapley','Model1_Shapley','Model2_Shapley'])"""


# In[ ]:


#preds =model.predict(xgboost.DMatrix(original_split['original_data_cluster0']))
#plt.scatter(np.arange(len(original_split['original_test_label_cluster0'])),original_split['original_test_label_cluster0'])
#plt.scatter(np.arange(len(original_split['original_test_label_cluster0'])),model_dict['model0'].predict(xgboost.DMatrix(original_split['original_test_cluster0'])))


# In[ ]:


#preds =model.predict(xgboost.DMatrix(original_split['original_data_cluster0']))
#plt.scatter(np.arange(len(original_split_shapley['original_test_label_cluster0'])),original_split_shapley['original_test_label_cluster0'])
#plt.scatter(np.arange(len(original_split_shapley['original_test_label_cluster0'])),model_dict_shapley['model0'].predict(xgboost.DMatrix(original_split_shapley['original_test_cluster0'])))


# In[ ]:


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


# In[ ]:


#plt.plot(np.arange(len(ev_result['test']['rmse'])),ev_result['test']['rmse'])
#plt.xlabel('Epoch count')
#plt.ylabel('Test-RMSE')
#plt.legend(['General Model'])


# In[ ]:


#plt.plot(np.arange(len(ev_result['train']['rmse'])),ev_result['train']['rmse'])
#plt.xlabel('Epoch count')
#plt.ylabel('Train-RMSE')
#plt.legend(['General Model'])


# In[ ]:


#plt.scatter(np.arange(len(X_test)),y_test)
#plt.scatter(np.arange(len(X_test)),model.predict(xgboost.DMatrix(X_test)))


# In[ ]:


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


# In[ ]:


#Do PCA for dimensionality reduction
nPcaComponents = 2
pca = PCA(n_components=nPcaComponents)
shap_values_pca = pca.fit_transform(shap_values)
print(pca.explained_variance_ratio_.sum())


# In[ ]:


#Do PCA for dimensionality reduction
nPcaComponents = 2
pca_org = PCA(n_components=nPcaComponents)
org_values_pca = pca_org.fit_transform(X)
print(pca_org.explained_variance_ratio_.sum())


# In[ ]:


#PCA Clusters are quite seperate,could be used in clustering
label_colors = [ 0 if a == 'train' else 1 for a in X_instanced['instance']]
plt.scatter(shap_values_pca[:,0],shap_values_pca[:,1], c = label_colors)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
#Add to labels to the scatter plot (Colors)
#How do we interpret this ?


# In[ ]:


kmeans_pca = cluster.clusterData(KMeans(n_clusters=nPcaComponents, random_state=0).fit,shap_values_pca)
kmeans_org_pca = cluster.clusterData(KMeans(n_clusters=nPcaComponents, random_state=0).fit,org_values_pca)


# In[ ]:


plt.scatter(shap_values_pca[:,0],shap_values_pca[:,1], c = kmeans_pca.labels_)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')


# In[ ]:


data_dict_shap_pca = cluster.splitDataLabeled(2,shap_values_pca,kmeans_pca.labels_)
data_dict_org_pca = cluster.splitDataLabeled(2,org_values_pca,kmeans_org_pca.labels_)


# In[ ]:


pca_instanced = pd.concat((pd.DataFrame(org_values_pca),pd.DataFrame(columns = ['instance'])),axis = 1)
pca_instanced['instance'].loc[X_train.index] = 'train'
pca_instanced['instance'].loc[X_test.index] = 'test'
shap_pca_instanced = pd.concat((pd.DataFrame(shap_values_pca),pd.DataFrame(columns = ['instance'])),axis = 1)
shap_pca_instanced['instance'].loc[X_train.index] = 'train'
shap_pca_instanced['instance'].loc[X_test.index] = 'test'


# In[ ]:


original_shap_pca = cluster.convertOriginalData(data_dict_shap_pca,shap_pca_instanced,y_instanced)
original_pca = cluster.convertOriginalData(data_dict_org_pca,pca_instanced,y_instanced)
original_pca


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


plt.plot(np.arange(len(eval_results_shap_pca['eval0']['test']['rmse'])),eval_results_shap_pca['eval0']['test']['rmse'])
plt.plot(np.arange(len(eval_results_shap_pca['eval1']['test']['rmse'])),eval_results_shap_pca['eval1']['test']['rmse'])
plt.plot(np.arange(len(eval_results_shap_pca['eval0']['train']['rmse'])),eval_results_shap_pca['eval0']['train']['rmse'])
plt.plot(np.arange(len(eval_results_shap_pca['eval1']['train']['rmse'])),eval_results_shap_pca['eval1']['train']['rmse'])
plt.xlabel('Epoch count')
plt.ylabel('Test-RMSE')
plt.legend(['Model0-Test','Model1-Test','Model0-Train','Model1-Train'])


# In[ ]:


ax = plt.figure()
plt.barh(['Whole model','Original_ensemble','Shapley_ensemble','Shapley_ensemble + PCA','Original_ensemble + PCA'],[tot_result_big_model,tot_rmse_org,tot_rmse_shap,tot_rmse_shap_pca,tot_rmse_org_pca])
plt.title('Test-RMSE value vs. Method used')
for i,v in enumerate([tot_result_big_model,tot_rmse_org,tot_rmse_shap,tot_rmse_shap_pca,tot_rmse_org_pca]):
    plt.text(v , i + .05, '{0:.3f}'.format(v), color='blue', fontweight='bold')


# In[ ]:


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




