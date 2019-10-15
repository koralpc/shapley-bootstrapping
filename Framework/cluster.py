"""
Author : Koralp Catalsakal
Date : 15/10/2019
"""


def clusterData(model_func = None, data = None , label = None):

    """
    Model agnostic method to be used in clustering. Can be used to try different models over datasets.
    Default setup accepts unsupervised dataset.

    Args:
        model_func(Model.function) : The specific model function to be used. The result of the function is
            returned in the model

        data(numpy.ndarray or pandas.DataFrame) : Data to be used in training(Without the ground truth data)

        label(numpy.ndarray or pandas.DataFrame) : Labels(Ground truth) of the data argument.

    Returns:
        result(Model) : Returns the fitted/trained model as output


    """
    if label == None:
        result = model_func(data)
    else:
        result = model_func(data,label)
    return result
