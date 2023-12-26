"""
Some transformers
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class Selector(BaseEstimator, TransformerMixin):
    """
    Selcects the features (numerical, categorical or all)
    """

    def __init__(self, select):
        """
        select has to be "num features", "cat features" or "all features"
        """

        if select not in ["num features", "cat features", "all features"]:
            raise TypeError("for select only num features, cat features or all features")

        self.select = select
        self.num_attr = None
        self.cat_attr = None

    def fit(self, x: pd.DataFrame, _y=None):
        """fits the parameter"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("Selector needs Pandas Dataframe!")

        self.num_attr = list(x.select_dtypes(include=[np.number]).columns)
        self.cat_attr = list(x.select_dtypes(exclude=[np.number]).columns)

        return self

    def transform(self, x: pd.DataFrame, _y=None):
        """does the transformation"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("Selector needs Pandas Dataframe!")

        if self.select == "num features":
            x_new = x[self.num_attr].copy()
        elif self.select == "cat features":
            x_new = x[self.cat_attr].copy()
        elif self.select == "all features":
            x_new = x[self.num_attr + self.cat_attr].copy()
        else:
            raise TypeError("for select only num features, cat features or all features")

        return x_new

    def get_feature_names_out(self):
        """this method is needed, otherwise we cannot use set_ouput"""
        pass


class MyPowerTransformer(BaseEstimator, TransformerMixin):
    """Performs a power transformation, possible: "yeo-johnson", or None (default) """

    def __init__(self, method=None):
        self.method = method  # "yeo-johnson" or None
        self.powertransformer = PowerTransformer()
        self.powertransformer.set_output(transform="pandas")

    def fit(self, x, _y=None):
        """fits the parameter"""
        if self.method == "yeo-johnson":
            self.powertransformer.fit(x)
            pass
        elif self.method is None:
            pass
        else:
            raise TypeError("for method only yeo-johnson or None possible")

        return self

    def transform(self, x, _y=None):
        """does the transformation"""

        if self.method == "yeo-johnson":
            return self.powertransformer.transform(x)
        elif self.method is None:
            return x
        else:
            raise TypeError("for method only yeo-johnson, box-cox, standardscaler or none possible")

    def get_feature_names_out(self):
        """this method is needed, otherwise we cannot use set_ouput"""
        pass