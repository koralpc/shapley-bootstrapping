from sklearn.linear_model import LinearRegression
import xgboost
import shap
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing, impute
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA


class ProcessingBlock:
    def __init__(self):
        """[Initaties the pre-processing block. Uses `sklearn.StandardScaler`
            for standardization of inputs and `sklearn.SimpleImputer` 
            for imputing missing values]
        """
        print("Processing Block Constructed")
        self.X_scaler = preprocessing.StandardScaler()
        self.y_scaler = preprocessing.StandardScaler()
        self.imputer = impute.SimpleImputer(
            missing_values=np.nan, strategy="most_frequent"
        )

    def fit(self, X, y):
        """[Stores the given X,y data in the object fields X and y]

        Args:
            X ([np.array or pd.DataFrame]): [Input data]
            y ([np.array or pd.DataFrame]): [Input labels]

        Returns:
            [self]: [returns the object itselt]
        """
        self.X = X
        self.y = y
        return self

    def split_data(self, X, y=None, test_split=0.2, scale=False):
        """[Splits the data into training and test set]

        Args:
            X ([np.array or pd.DataFrame]): [Input data]
            y ([np.array or pd.DataFrame], optional): [Input labels.]. Defaults to None.
            test_split (float, optional): [Test data split size]. Defaults to 0.2.
            scale (bool, optional): [Keyword to enable standardization of data]. Defaults to False.

        Returns:
            [np.array or pd.DataFrame]: [If `y` is given, returns X_train,X_test,y_train,y_test
            Otherwise returns X_train,X_test]
        """
        if scale:
            X = self.X_scaler.fit_transform(X)

        if y is not None:
            X_df = pd.DataFrame(X)
            X_train, X_val, y_train, y_val = train_test_split(
                X_df, y, test_size=test_split, random_state=0
            )

            self.train_idx = X_train.index
            self.val_idx = X_val.index

            # X_train.reset_index(inplace = True)
            # X_train = X_train.drop(['index'],axis = 1)
            # X_val.reset_index(inplace = True)
            # X_val = X_val.drop(['index'],axis = 1)

            return X_train, X_val, y_train, y_val
        else:
            X_df = pd.DataFrame(X)
            X_train, X_val = train_test_split(
                X_df, test_size=test_split, random_state=0
            )

            self.train_idx = X_train.index
            self.val_idx = X_val.index

            # X_train.reset_index(inplace = True)
            # X_train = X_train.drop(['index'],axis = 1)
            # X_val.reset_index(inplace = True)
            # X_val = X_val.drop(['index'],axis = 1)

            return X_train, X_val

    def impute_data(self, X, y=None,scale = False):
        """[Imputes missing instances in given data]

        Args:
            X ([np.array or pd.DataFrame]): [Input data]
            y ([np.array or pd.DataFrame], optional): [Input labels. If not given in arguments,
            can be used to impute test data]. Defaults to None.
            scale ([bool],optional) : [Standardizes the dava if True]. Defaults to False.

        Returns:
            [np.array or pd.DataFrame]: [If y is given, returns X and y with imputed values. Else, returns only X with imputed values]
        """
        if scale:
            X = self.X_scaler.transform(X)

        if y is None:
            X = self.imputer.fit_transform(X)
            X = pd.DataFrame(X)
            return X
        else:
            X = self.imputer.fit_transform(X)
            y = self.imputer.fit_transform(y.reshape(-1, 1))
            X = pd.DataFrame(X)
            return X, y


class ExplainerBlock:
    def __init__(self, explainer_type, params=None, kwargs=None):
        """[Instantiates explainer block object]

        Args:
            explainer_type ([str]): [Explainer model type. Currently `Linear` and `XGBoost` is supported]
            params ([dict], optional): [XGBoost parameters]. Defaults to None.
            kwargs ([dict], optional): [XGBoost keyword-arguments]. Defaults to None.
        """
        print("Shapley Explainer Constructed")
        self.explainer_type = explainer_type
        self.eval_results = {}
        self.base_model = None

        if params is None:
            self.explainer_params = {
                "eta": 0.05,
                "max_depth": 3,
                "objective": "reg:squarederror",
                "subsample": 0.7,
                "eval_metric": "rmse",
                "lambda": 0.1,
            }
        else:
            self.explainer_params = params

        if kwargs is None:
            self.keyword_args = {
                "num_boost_round": 5000,
                "verbose_eval": 0,
                "evals_result": {},
                "early_stopping_rounds": 200,
            }
        else:
            self.keyword_args = kwargs

    def fit(self, X_exp, y_exp, X_train, y_train, X_val, y_val):
        """[Trains a model over the input data and constructs the Explainer object using trained model]

        Args:
            X_exp ([np.array or pd.DataFrame]): [Input data of which Shapley values are calculated]
            y_exp ([np.array or pd.DataFrame]): [Input labels]
            X_train ([np.array or pd.DataFrame]): [Train partition of input data]
            y_train ([np.array or pd.DataFrame]): [Train partition of input labels]
            X_val ([np.array or pd.DataFrame]): [Test partition of input data]
            y_val ([np.array or pd.DataFrame]): [Test partition of input labels]

        Returns:
            [self]: [Returns model itself with `explainer_model` and `base_model`]
        """

        if self.explainer_type == "Linear":
            self.base_model = LinearRegression().fit(X_exp, y_exp)
        else:
            eval = [
                (xgboost.DMatrix(X_train, label=y_train), "train"),
                (xgboost.DMatrix(X_val, label=y_val), "val"),
            ]
            self.base_model = xgboost.train(
                self.explainer_params,
                xgboost.DMatrix(X_train, label=y_train),
                evals=eval,
                **self.keyword_args
            )

        if self.explainer_type == "Linear":
            self.explainer = shap.LinearExplainer(
                self.base_model, X_exp, feature_dependence="independent"
            )
        else:
            self.explainer = shap.TreeExplainer(self.base_model)

        return self

    def transform(self, X):
        """[Transforms input features to Shapley values]

        Args:
            X ([np.array or pd.DataFrame]): [Input features]

        Returns:
            [np.array]: [Shapley values of input features]
        """
        shapley_values = self.explainer.shap_values(X)
        return shapley_values

    def fit_transform(self, X, y, X_train, y_train, X_val, y_val):
        """[Fit and transform combined. Shortcut method if one wants to use `fit` and `transform`
           directly after each other]
        """        
        self.fit(X, y, X_train, y_train, X_val, y_val)
        shapley_values = self.transform(X)
        return shapley_values

    def predict(self, X):
        """[Uses the trained model in explainer to make predictions]

        Args:
            X ([np.array or pd.DataFrame]): [Input features]

        Returns:
            [np.array]: [Predictions]
        """        
        if self.explainer_type == "Linear":
            y_pred = self.base_model.predict(X)
        if self.explainer_type == "XGBoost":
            y_pred = self.base_model.predict(xgboost.DMatrix(X))
        return y_pred


class ClusterBlock(BaseEstimator, RegressorMixin):
    def __init__(self, nClusters, training_set_model, test_set_model):
        # print('Clustering Block Constructed')
        self.n_clusters = nClusters
        self.training_set_model = training_set_model
        self.test_set_model = test_set_model

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        print("transform")
        pass

    def cluster_training_instances(self, X):
        self.training_set_model.fit(X)
        return self.training_set_model.labels_

    def cluster_test_instances(self, X, X_test):
        self.test_set_model.fit(X, self.training_set_model.labels_)
        prediction = self.test_set_model.predict(X_test)
        return prediction


class EnsembleBlock(BaseEstimator, RegressorMixin):
    def __init__(self, model_type, params=None, keyword_args=None):
        # print('Ensemble Models Constructed')
        self.eval_dict = {}
        self.model_dict = {}
        self.model_type = model_type

        if params is None:
            self.ensemble_params = {
                "eta": 0.05,
                "max_depth": 3,
                "objective": "reg:squarederror",
                "subsample": 0.7,
                "eval_metric": "rmse",
                "lambda": 0.1,
            }
        else:
            self.ensemble_params = params

        if keyword_args is None:
            self.keyword_args = {
                "num_boost_round": 5000,
                "verbose_eval": 0,
                "evals_result": {},
                "early_stopping_rounds": 200,
            }
        else:
            self.keyword_args = keyword_args

    def fit(self, X, y):
        pass

    def train(self, X_train, X_val, y_train, y_val, cluster_labels):
        if self.model_type == "Linear":
            for i in range(len(np.unique(cluster_labels))):
                c_idx = cluster_labels == i
                X_train_cluster = X_train[c_idx[X_train.index]]
                y_train_cluster = y_train[c_idx[X_train.index]]
                X_val_cluster = X_val[c_idx[X_val.index]]
                y_val_cluster = y_val[c_idx[X_val.index]]
                self.model_dict["model{0}".format(i)] = LinearRegression().fit(
                    X_train_cluster, y_train_cluster
                )
                # self.eval_dict['eval{0}'.format(i)] = {'train': {'rmse': _calculate_accuracy(model_dict['model{0}'.format(i)].predict,X_val_cluster,y_val_cluster)}}

        if self.model_type == "XGBoost":
            self.keyword_args["evals_result"] = {}
            for i in range(len(np.unique(cluster_labels))):
                c_idx = cluster_labels == i
                X_train_cluster = X_train[c_idx[X_train.index]]
                y_train_cluster = y_train[c_idx[X_train.index]]
                X_val_cluster = X_val[c_idx[X_val.index]]
                y_val_cluster = y_val[c_idx[X_val.index]]
                if not y_val_cluster.size == 0:
                    dtrain = xgboost.DMatrix(X_train_cluster, label=y_train_cluster)
                    eval = [
                        (
                            xgboost.DMatrix(X_train_cluster, label=y_train_cluster),
                            "train",
                        ),
                        (xgboost.DMatrix(X_val_cluster, label=y_val_cluster), "val"),
                    ]
                    self.model_dict["model{0}".format(i)] = xgboost.train(
                        self.ensemble_params, dtrain, evals=eval, **self.keyword_args
                    )
                    self.eval_dict["eval{0}".format(i)] = self.keyword_args[
                        "evals_result"
                    ]
                else:
                    dtrain = xgboost.DMatrix(X_train_cluster, label=y_train_cluster)
                    eval = [
                        (
                            xgboost.DMatrix(X_train_cluster, label=y_train_cluster),
                            "train",
                        )
                    ]
                    self.model_dict["model{0}".format(i)] = xgboost.train(
                        self.ensemble_params, dtrain, evals=eval, **self.keyword_args
                    )
                    self.eval_dict["eval{0}".format(i)] = self.keyword_args[
                        "evals_result"
                    ]

    def predict(self, X_test, cluster_labels):
        y_pred = np.zeros(shape=(X_test.shape[0],))
        for i in range(len(np.unique(cluster_labels))):
            if (cluster_labels == i).any():
                if self.model_type == "Linear":
                    y_pred[cluster_labels == i] = (
                        self.model_dict["model{0}".format(i)]
                        .predict(X_test[cluster_labels == i])
                        .reshape(-1,)
                    )
                else:
                    y_pred[cluster_labels == i] = self.model_dict[
                        "model{0}".format(i)
                    ].predict(xgboost.DMatrix(X_test[cluster_labels == i]))
            else:
                continue
        return y_pred


class ReduceBlock(BaseEstimator, RegressorMixin):
    def __init__(self, reduce_model):
        # print('Dimensionality Reduction Block Constructed')
        self.reduce_model = reduce_model

    def fit(self, X):
        # from sklearn.decomposition import PCA

        explained_var_ratio = 0
        k = 1
        n_features = X.shape[1]
        while explained_var_ratio < 0.95:
            k += 1
            pca = PCA(n_components=min(k, n_features))
            pca.fit(X)
            explained_var_ratio = pca.explained_variance_ratio_.sum()

        self.reduce_model = pca

    def transform(self, X):
        X_reduced = self.reduce_model.transform(X)
        return X_reduced

    def fit_transform(self, X):
        self.fit(X)
        X_reduced = self.transform(X)
        return X_reduced
