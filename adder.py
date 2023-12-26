"""
Some adder
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import PowerTransformer

class NumAttributesAdder(BaseEstimator, TransformerMixin):
    """Adds new numeric features"""

    def __init__(self):
        pass

    def fit(self, x: pd.DataFrame, _y=None):
        """fits the parameter"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("NumAttributesAdder needs Pandas Dataframe!")

        return self

    @staticmethod  # static because in transform self is not used
    def transform(_self, x: pd.DataFrame, _y=None):
        """does the transformation"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("NumAttributesAdder needs Pandas Dataframe!")

        x_new = x.copy()
        x_new["rooms_per_household"] = x_new["total_rooms"] / x_new["households"]
        x_new["rooms_per_household"].clip(lower=1, inplace=True)  # hard to find an upper value (consider touristic region with hotels only)
        x_new["population_per_household"] = x_new["population"] / x_new["households"]
        x_new["population_per_household"].clip(0, 10, inplace=True)
        x_new["bedrooms_per_room"] = x_new["total_bedrooms"] / x_new["total_rooms"]
        x_new["bedrooms_per_room"].clip(0, 1, inplace=True)

        return x_new

    def get_feature_names_out(self):
        """this method is needed, otherwise we cannot use set_ouput"""
        pass


class CatAttributesAdder(BaseEstimator, TransformerMixin):
    """Adds new numeric features"""

    def __init__(self):
        pass

    def fit(self, x: pd.DataFrame, _y=None):
        """fits the parameter"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("CatAttributesAdder needs Pandas Dataframe!")

        return self

    @staticmethod  # static because in transform self is not used
    def transform(_self, x: pd.DataFrame, _y=None):
        """does the transformation"""

        if not isinstance(x, pd.DataFrame):
            raise TypeError("CatAttributesAdder needs Pandas Dataframe!")

        x_new = x.copy()
        x_new["median_income_max"] = (x["median_income"] >= 15)
        x_new["housing_median_age_max"] = (x["housing_median_age"] >= 52)

        return x_new

    def get_feature_names_out(self):
        """this method is needed, otherwise we cannot use set_ouput"""
        pass
