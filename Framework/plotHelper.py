"""
Author : Koralp Catalsakal
Date : 15/10/2019
"""
import seaborn as sns
import matplotlib.pyplot as plt

def plotAllFeatures(data):

    """
        Plotting function that shows the scatter plot of each individual feature in
        its respective figure

        Args:
            data(pd.DataFrame): Data should carry both features and output labels.
                Output label should be the last column

        Returns:
            None
    """

    palette = sns.color_palette("Blues", len(data))
    for i in range(len(data.columns) - 1):
            plt.figure()
            plt.scatter(data.iloc[:,i],data.iloc[:,-1],color = palette)
            plt.xlabel('Feature({0}) shapley contribution'.format(data.columns[i]))
            plt.ylabel('{0} progression'.format(data.columns[-1]))
            plt.title('Feature({0}) contribution vs. {1} progression'.format(data.columns[i],data.columns[-1]))
