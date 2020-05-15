"""
Author : Koralp Catalsakal
Date : 20/03/2020
"""

import Framework.utils as utils
import Framework.datasets as datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_pairvs(benchmark_df,branch1,branch2,label_opt = None,epsilon = 1e-03):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
        label_opt ([type], optional): [description]. Defaults to None.
        epsilon ([type], optional): [description]. Defaults to 1e-03.
    """    

    #bench_b1_vs_b2 = benchmark_df.copy()
    #bsl1 = bench_b1_vs_b2.groupby('Dataset_name').agg({branch1 : min})
    #bsl2 = bench_b1_vs_b2.groupby('Dataset_name').agg({branch2 : min})
    #for j in range(len(bench_b1_vs_b2['Dataset_name'])):
    #    bench_b1_vs_b2[branch1].iloc[j] = bsl1.loc[bench_b1_vs_b2['Dataset_name'].iloc[j]][0]
    #    bench_b1_vs_b2[branch2].iloc[j] = bsl2.loc[bench_b1_vs_b2['Dataset_name'].iloc[j]][0]
    bench_b1_vs_b2 = utils.set_branch_best(benchmark_df,branch1,branch2)
    bench_b1_vs_b2['Vs_results'] = bench_b1_vs_b2.apply(lambda x : _label_vs(x,branch1,branch2,epsilon) ,axis = 1)
    bench_df_grouped = bench_b1_vs_b2.groupby('Vs_results').agg({'Dataset_name':'nunique'})
    bench_df_grouped.reset_index(inplace = True)
    bench_df_grouped.columns = ['Comparison Winner','Number of Wins']
    bench_df_grouped_cat = bench_b1_vs_b2.groupby(['Dataset_name','Vs_results']).agg({'Dataset_name':'nunique'})
    bench_df_grouped_cat.columns = ['Dataset Index']
    bench_df_grouped_cat['Dataset Index'] = np.arange(bench_df_grouped_cat.shape[0])
    bench_df_grouped_cat.reset_index(inplace = True)

    ax1 = sns.barplot(y = 'Number of Wins',x = 'Comparison Winner',data = bench_df_grouped)
    ax1.set_title('Comparison results with significance level σ: {}'.format(epsilon))
    ax2 = sns.catplot(x = 'Vs_results',y ='Dataset Index',hue = 'Dataset_name',data = bench_df_grouped_cat)
   
    if label_opt is None:
        ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
        ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
        plt.show()
    else:
        label_opt_list = [label_opt[i] for i in bench_df_grouped['Comparison Winner']]
        print(label_opt_list)
        ax1.set_xticklabels(label_opt_list)
        ax2.set_xticklabels(label_opt_list)
        ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
        ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
        plt.show()


def visualize_pairvs_2(benchmark_df,branch1,branch2,label_opt = None,epsilon = 1e-03):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
        label_opt ([type], optional): [description]. Defaults to None.
        epsilon ([type], optional): [description]. Defaults to 1e-03.
    """    
    benchmark_mins,benchmark_min_idx = get_branch_mins(benchmark_df)
    benchmark_mins['Vs_results'] = benchmark_mins.apply(lambda x : _label_vs(x,branch1,branch2,epsilon) ,axis = 1)
    bench_df_grouped = benchmark_mins.groupby('Vs_results').agg({'Dataset_name':'nunique'})
    bench_df_grouped.reset_index(inplace = True)
    bench_df_grouped.columns = ['Comparison Winner','Number of Wins']
    bench_df_grouped_cat = benchmark_mins.groupby(['Dataset_name','Vs_results']).agg({'Dataset_name':'nunique'})
    bench_df_grouped_cat.columns = ['Dataset Index']
    bench_df_grouped_cat['Dataset Index'] = np.arange(bench_df_grouped_cat.shape[0])
    bench_df_grouped_cat.reset_index(inplace = True)
    ax1 = sns.barplot(y = 'Number of Wins',x = 'Comparison Winner',data = bench_df_grouped)
    ax1.set_title('Comparison results with significance level σ: {}'.format(epsilon))
    ax2 = sns.catplot(x = 'Vs_results',y ='Dataset Index',hue = 'Dataset_name',data = bench_df_grouped_cat)
   
    if label_opt is None:
        ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
        ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
        ax1.set(yticks = np.arange(0,bench_df_grouped_cat.shape[0],2))
        plt.show()
    else:
        label_opt_list = [label_opt[i] for i in bench_df_grouped['Comparison Winner']]

        ax1.set_xticklabels(label_opt_list)
        ax2.set_xticklabels(label_opt_list)
        ax1.set(yticks = np.arange(0,bench_df_grouped_cat.shape[0],2))
        ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
        ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
        plt.show()

    return ax1
    

def visualize_multiple_vs(benchmark_df,branch_list,label_opt = None,epsilon = 1e-03):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
        label_opt ([type], optional): [description]. Defaults to None.
        epsilon ([type], optional): [description]. Defaults to 1e-03.
    """    
    i = 0
    plt.figure(figsize = (30,10))
    for eps in [0.01,0.001,0.0001]:
        i += 1
        plt.subplot(1,3,i)
        benchmark_mins,benchmark_min_idx = get_branch_mins(benchmark_df)
        benchmark_mins['Vs_results'] = benchmark_mins.apply(lambda x : label_vs_triple(x,branch_list,eps) ,axis = 1)
        bench_df_grouped = benchmark_mins.groupby('Vs_results').agg({'Dataset_name':'nunique'})
        bench_df_grouped.reset_index(inplace = True)
        bench_df_grouped.sort_index(inplace = True)
        bench_df_grouped.columns = ['Comparison Winner','Number of Wins']
        bench_df_grouped_cat = benchmark_mins.groupby(['Dataset_name','Vs_results']).agg({'Dataset_name':'nunique'})
        bench_df_grouped_cat.columns = ['Dataset Index']
        bench_df_grouped_cat['Dataset Index'] = np.arange(bench_df_grouped_cat.shape[0])
        bench_df_grouped_cat.reset_index(inplace = True)
        ax1 = sns.barplot(y = 'Number of Wins',x = 'Comparison Winner',data = bench_df_grouped)
        ax1.set_title('Comparison results with significance level σ: {}'.format(eps),fontsize = 24)
        #ax2 = sns.catplot(x = 'Vs_results',y ='Dataset Index',hue = 'Dataset_name',data = bench_df_grouped_cat)
    
        if label_opt is None:
           # ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
           # ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
            ax1.set(yticks = np.arange(0,bench_df_grouped_cat.shape[0],2))
            ax1.set_ylabel("Comparison Winner",fontsize=22)
            ax1.set_xlabel("Number of Wins",fontsize=22)
            #plt.show()
        else:
            label_opt_list = [label_opt[i] for i in bench_df_grouped['Comparison Winner']]

            ax1.set_xticklabels(label_opt_list)
            ax1.set_ylabel("Comparison Winner",fontsize=22)
            ax1.set_xlabel("Number of Wins",fontsize=22)
            ax1.tick_params(labelsize=22)
            #ax2.set_xticklabels(label_opt_list)
            ax1.set(yticks = np.arange(0,bench_df_grouped_cat.shape[0],2))
            #ax2.set(xlabel='Better Performance',ylabel = 'Datasets',yticks = np.arange(bench_df_grouped_cat.shape[0]))
            #ax2.set_yticklabels(list(bench_df_grouped_cat['Dataset_name']))
            #plt.show()
    plt.tight_layout()
    return ax1

def _label_vs(row,branch1,branch2,epsilon = 1e-03):
    if abs(row[branch1] - row[branch2]) <= epsilon:
        return 'Draw'
    else:
        if row[branch1] < row[branch2] :
            return branch1
        elif row[branch1] > row[branch2] :
            return branch2

def label_vs_triple(row,branch_list,epsilon = 1e-03):
    branch_array = np.asarray(branch_list)
    benchmark_min_idx = row[branch_list].astype(float).idxmin()
    for el in branch_array[branch_array != benchmark_min_idx]:
        if abs(row[el] - row[benchmark_min_idx]) <= epsilon:
            return 'Draw'
    return benchmark_min_idx

def dataset_stats(index_range):
    """[summary]
    
    Args:
        index_range ([type]): [description]
    
    Returns:
        [type]: [description]
    """    
    dataset_features = {}
    for i in range(index_range):
        X,_,name = datasets.returnDataset(i)
        dataset_features[name] = {'Instances': X.shape[0], 'Features' : X.shape[1]}

    return dataset_features

def scatter_datasets(benchmark_df,branch1,branch2,index_range = 20):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
        index_range (int, optional): [description]. Defaults to 20.
    """    
    dataset_dictionary = dataset_stats(index_range)
    bench_b1_vs_b2 = utils.set_branch_best(benchmark_df,branch1,branch2)
    bench_b1_vs_b2['Vs_results'] = bench_b1_vs_b2.apply(lambda x : _label_vs(x,branch1,branch2) ,axis = 1)
    bench_b1_vs_b2['Instances'] = bench_b1_vs_b2.apply(lambda x : dataset_dictionary[x['Dataset_name']]['Instances'] , axis = 1)
    bench_b1_vs_b2['Features'] = bench_b1_vs_b2.apply(lambda x : dataset_dictionary[x['Dataset_name']]['Features'] , axis = 1)

    sns.scatterplot(x = 'Instances',y = 'Features' , hue = 'Vs_results', data = bench_b1_vs_b2)
    return dataset_dictionary

def get_branch_mins(dataset):
    """[summary]
    
    Args:
        dataset ([type]): [description]
        sigma (float, optional): [description]. Defaults to 0.001.
    
    Returns:
        [type]: [description]
    """    
    branch_mins_df = dataset.groupby('Dataset_name',as_index = False)['B1','B2','B4','B5','B7','B8'].min()
    best_performance_idx = dataset.groupby('Dataset_name',as_index = False).agg(dict(zip(['B1','B2','B4','B5','B7','B8'],['idxmin']*6)))
    return branch_mins_df,best_performance_idx
    
def get_branch_minidx(grouped_df,best_performance_idx,branch1,branch2,sigma = 0.001):
    """[summary]
    
    Args:
        grouped_df ([type]): [description]
        best_performance_idx ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
    
    Returns:
        [type]: [description]
    """    
    datasets_better = grouped_df[(grouped_df[branch1] > grouped_df[branch2]) & (abs(grouped_df[branch1] - grouped_df[branch2]) > sigma)]
    dataset_better_idx = best_performance_idx[best_performance_idx['Dataset_name'].isin(datasets_better['Dataset_name'])]
    return dataset_better_idx.loc[:,['Dataset_name',branch1,branch2]]

def compare_branch_best_runtime(benchmark_df,runtime_df,branch1,branch2,label_opt = None,sigma = 0.001):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        runtime_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
    """    
    bcn_n0_c1_mins,bcn_n0_c1_minidx = get_branch_mins(benchmark_df)
    dataset_btt = get_branch_minidx(bcn_n0_c1_mins,bcn_n0_c1_minidx,branch1,branch2,sigma)
    b1_best_runtime = runtime_df[['Dataset_name',branch1]].loc[dataset_btt[branch1].tolist()]
    b2_best_runtime = runtime_df[['Dataset_name',branch2]].loc[dataset_btt[branch2].tolist()]
    b1_melt = pd.melt(b1_best_runtime,id_vars = 'Dataset_name')
    b2_melt = pd.melt(b2_best_runtime,id_vars = 'Dataset_name')
    branch_ct = pd.concat([b1_melt,b2_melt])
    if label_opt is None:   
        ax = sns.catplot(y = 'Dataset_name',x= 'value',hue = 'variable',kind = 'bar',data = branch_ct)
        ax.set(xlabel ='Training completion time(secs)',ylabel ='Dataset name',title='Runtime comparison of {} and {}, for datasets where {} performed better with significance level σ: {}'.format(branch1,branch2,branch2,sigma))
    else:
        branch_ct['variable'].loc[branch_ct['variable'] == branch1 ] = label_opt[branch1]
        branch_ct['variable'].loc[branch_ct['variable'] == branch2 ] = label_opt[branch2]
        ax = sns.catplot(y = 'Dataset_name',x= 'value',hue = 'variable',kind = 'bar',data = branch_ct)
        ax.set(xlabel ='Training completion time(secs)',ylabel ='Dataset name',title='Runtime comparison of {} and {}, for datasets where {} performed better with significance level σ: {}'.format(label_opt[branch1],label_opt[branch2],label_opt[branch2],sigma))

def compare_branch_best_complexity(benchmark_df,complexity_df,branch1,branch2,label_opt = None,sigma = 0.001):
    """[summary]
    
    Args:
        benchmark_df ([type]): [description]
        runtime_df ([type]): [description]
        branch1 ([type]): [description]
        branch2 ([type]): [description]
    """    
    bcn_n0_c1_mins,bcn_n0_c1_minidx = get_branch_mins(benchmark_df)
    dataset_btt = get_branch_minidx(bcn_n0_c1_mins,bcn_n0_c1_minidx,branch1,branch2,sigma)
    b1_best_complexity = complexity_df[['Dataset_name',(branch1+'_avg_node')]].loc[dataset_btt[(branch1)].tolist()]
    b2_best_complexity = complexity_df[['Dataset_name',(branch2+'_avg_node')]].loc[dataset_btt[(branch2)].tolist()]
    b1_melt = pd.melt(b1_best_complexity,id_vars = 'Dataset_name')
    b2_melt = pd.melt(b2_best_complexity,id_vars = 'Dataset_name')
    branch_ct = pd.concat([b1_melt,b2_melt])
    if label_opt is None:   
        ax = sns.catplot(y = 'Dataset_name',x= 'value',hue = 'variable',kind = 'bar',data = branch_ct)
        ax.set(xlabel ='Average number of nodes per tree',ylabel ='Dataset name',title='Average node count per tree comparison of {} and {}, for datasets where {} performed better with significance level σ: {}'.format(branch1,branch2,branch2,sigma))
    else:
        branch_ct['variable'].loc[branch_ct['variable'] == (branch1 + '_avg_node')] = (label_opt[branch1] + '_avg_node')
        branch_ct['variable'].loc[branch_ct['variable'] == (branch2 + '_avg_node')] = (label_opt[branch2] + '_avg_node')
        ax = sns.catplot(y = 'Dataset_name',x= 'value',hue = 'variable',kind = 'bar',data = branch_ct)
        ax.set(xlabel ='Average number of nodes per tree',ylabel ='Dataset name',title='Average node count per tree comparison of {} and {}, for datasets where {} performed better with significance level σ: {}'.format(label_opt[branch1],label_opt[branch2],label_opt[branch2],sigma))