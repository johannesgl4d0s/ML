#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Project 1

# In[87]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import joblib
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR, LinearSVR, LinearSVC
from sklearn.base import clone
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
sns.set_theme()


# In[89]:


# Column Selector based on dtype (from lecture)
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


# In[90]:


df = pd.read_csv('project_1_train.csv')


# In[91]:


df.head()


# In[92]:


df.info()


# In[93]:


df.describe()


# In[7]:


# Histogram of the target variable
sns.histplot(df["DirectChol"], bins=20)
plt.savefig("assets/DirectChol.png", dpi=600, bbox_inches='tight')
plt.show()


# In[8]:


# Histogram of the target variable
sns.histplot(df["Diabetes"])
plt.savefig("assets/Diabetes.png", dpi=600, bbox_inches='tight')
plt.show()


# In[9]:


# Top 30 columns with most NA values
prop_na = df.isnull().sum().sort_values(ascending=False).head(30)
sns.barplot(x=prop_na.values, y=prop_na.index)
plt.title("Top 30 columns with most NA values")
plt.savefig('assets/top30na.png', dpi=600, bbox_inches='tight')
plt.show()


# In[99]:


# Drop columns with more than 1500 nas
max_number_of_nas = 1500
newdf = df.loc[:, (df.isnull().sum(axis=0) <= max_number_of_nas)]

# drop rows with nan values in the target variables
newdf = newdf.dropna(subset=["DirectChol", "Diabetes"])

#newdf.dropna(inplace=True)
newdf.info()


# In[11]:


# Cardinality of categorical features
d = df.select_dtypes(include="object").nunique().to_dict()
sns.barplot(x=list(d.values()), y=list(d.keys()))
plt.title("Cardinality of Categorical Features")
plt.savefig('assets/cardinality.png', dpi=600, bbox_inches='tight')
plt.show()


# ## Regression

# In[100]:


# Prep for regression
y_reg = newdf['DirectChol']
x_reg = newdf.drop(columns=['DirectChol'])

X_train, X_test, y_train, y_test = train_test_split(x_reg, y_reg, test_size=0.2, random_state=123)


# In[101]:


scores = {}  # containing scores and pipeline info

# Score Helper Function
def display_scores(model_score):
    """print the list of scores, the mean and the standard deviation"""
    print("SCORES OF CROSS VALIDATION:")
    print(np.round(model_score, decimals=1))
    print("MEAN SCORE: %0.1f" % model_score.mean())
    print("STD SCORE: %0.1f\n" % model_score.std())


# ### Linear Regression

# In[102]:


def lin_reg(x, y):
    """define model 1"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),  # select numeric attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set toDo better to drop with our dataset?
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1 toDo: Maybe min_max_scaler?
    ])

    # categorical feature preparation toDo: is this necessary?
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),  # select categorical attributes
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),  # one-hot-encoding
    ])

    # combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # set output to pandas, so we can use pandas tools again (e.g. for feature_names_in_)
    pipeline_full.set_output(transform="pandas")

    model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", LinearRegression()),
    ])

    model.fit(x, y)

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    with open("output_reg/lin_reg.pkl", "wb") as file:
        pickle.dump(model, file)
    joblib.dump(model_predictions, filename="output_reg/lin_reg_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/lin_reg_scores.pkl")
    joblib.dump(model_time, filename="output_reg/lin_reg_time.pkl")


# In[103]:


#lin_reg(X_train, y_train)


# In[104]:


model_1 = joblib.load("output_reg/lin_reg.pkl")
model_1_predictions = joblib.load("output_reg/lin_reg_predictions.pkl")
model_1_scores = joblib.load("output_reg/lin_reg_scores.pkl")
model_1_time = joblib.load("output_reg/lin_reg_time.pkl")

print("TIME: ", round(model_1_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_1_predictions)))

scores["model_1"] = [np.sqrt(-model_1_scores).mean(), model_1, "lin_reg"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_1"][0], 2), "\n")
display_scores(np.sqrt(-model_1_scores))

print("USED FEATURES:", model_1["regressor"].n_features_in_)

model_1_feature_importances = pd.DataFrame(
    {
        "name": model_1["regressor"].feature_names_in_,
        "value": model_1["regressor"].coef_.flatten(),
        "abs_value": np.abs(model_1["regressor"].coef_).flatten()
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_1_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_1_feature_importances.tail(6)[["name", "value"]], 2), "\n")


# In[17]:


def lin_reg_poly(x, y):
    """define model 2"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),  # select numeric attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set toDo better to drop with our dataset?
        ("poly", PolynomialFeatures(degree=2)),  # add polynomial features
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1 toDo: Maybe min_max_scaler?
    ])

    # categorical feature preparation toDo: is this necessary?
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),  # select categorical attributes
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),  # one-hot-encoding
    ])

    # combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # set output to pandas, so we can use pandas tools again (e.g. for feature_names_in_)
    pipeline_full.set_output(transform="pandas")

    model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", LinearRegression()),
    ])

    model.fit(x, y)

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    with open("output_reg/lin_reg_poly.pkl", "wb") as file:
        pickle.dump(model, file)
    joblib.dump(model_predictions, filename="output_reg/lin_reg_poly_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/lin_reg_poly_scores.pkl")
    joblib.dump(model_time, filename="output_reg/lin_reg_poly_time.pkl")


# In[105]:


#lin_reg_poly(X_train, y_train)


# In[107]:


model_2 = joblib.load("output_reg/lin_reg_poly.pkl")
model_2_predictions = joblib.load("output_reg/lin_reg_poly_predictions.pkl")
model_2_scores = joblib.load("output_reg/lin_reg_poly_scores.pkl")
model_2_time = joblib.load("output_reg/lin_reg_poly_time.pkl")

print("TIME: ", round(model_2_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_2_predictions)))

scores["model_2"] = [np.sqrt(-model_2_scores).mean(), model_2, "lin_reg_poly"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_2"][0], 2), "\n")
display_scores(np.sqrt(-model_2_scores))

print("USED FEATURES:", model_2["regressor"].n_features_in_)

model_2_feature_importances = pd.DataFrame(
    {
        "name": model_2["regressor"].feature_names_in_,
        "value": model_2["regressor"].coef_.flatten(),
        "abs_value": np.abs(model_2["regressor"].coef_).flatten()
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_2_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_2_feature_importances.tail(6)[["name", "value"]], 2), "\n")


# ### Decision Tree

# In[20]:


def decision_tree_reg(x, y):
    """define model 3"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
    ])

    # categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),  # select categorical attributes
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),  # one-hot-encoding
    ])

    # combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # set output to pandas, so we can use pandas tools again (e.g. for feature_names_in_)
    pipeline_full.set_output(transform="pandas")

    # define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", DecisionTreeRegressor(random_state=123)),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "regressor__max_depth": [8],
            "regressor__min_samples_leaf": [0.0008, 0.0009, 0.001],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)

    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    with open("output_reg/decision_tree.pkl", "wb") as file:
        pickle.dump(model, file)

    # use the following to store the best params of grid search
    with open("output_reg/decision_tree_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="output_reg/decision_tree_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/decision_tree_scores.pkl")
    joblib.dump(model_time, filename="output_reg/decision_tree_time.pkl")


# In[108]:


#decision_tree_reg(X_train, y_train)


# In[110]:


model_3 = joblib.load("output_reg/decision_tree.pkl")

with open("output_reg/decision_tree_best_params.pkl", 'rb') as f:
    model_3_best_params = pickle.load(f)

model_3_predictions = joblib.load("output_reg/decision_tree_predictions.pkl")
model_3_scores = joblib.load("output_reg/decision_tree_scores.pkl")
model_3_time = joblib.load("output_reg/decision_tree_time.pkl")

print("TIME: ", round(model_3_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_3_predictions)))

scores["model_3"] = [np.sqrt(-model_3_scores).mean(), model_3, "decision_tree"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_3"][0], 2), "\n")
display_scores(np.sqrt(-model_3_scores))

print("USED FEATURES: ", model_3["regressor"].n_features_in_)

model_3_feature_importances = pd.DataFrame(
    {
        "name": model_3["regressor"].feature_names_in_,
        "value": model_3["regressor"].feature_importances_,
        "abs_value": np.abs(model_3["regressor"].feature_importances_)
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_3_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_3_feature_importances.tail(6)[["name", "value"]], 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_3_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Random Forest

# In[23]:


def random_forest_reg(x, y):
    """Random Forest Regression."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Set output to pandas
    pipeline_full.set_output(transform="pandas")

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", RandomForestRegressor(random_state=123)),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 10, 20],
            "regressor__min_samples_leaf": [1, 2, 4],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_gs.fit(x, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    with open("output_reg/random_forest.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("output_reg/random_forest_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="output_reg/random_forest_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/random_forest_scores.pkl")
    joblib.dump(model_time, filename="output_reg/random_forest_time.pkl")


# In[111]:


#random_forest_reg(X_train, y_train)


# In[112]:


model_4 = joblib.load("output_reg/random_forest.pkl")

with open("output_reg/random_forest_best_params.pkl", 'rb') as f:
    model_4_best_params = pickle.load(f)

model_4_predictions = joblib.load("output_reg/random_forest_predictions.pkl")
model_4_scores = joblib.load("output_reg/random_forest_scores.pkl")
model_4_time = joblib.load("output_reg/random_forest_time.pkl")

print("TIME: ", round(model_4_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_4_predictions)))

scores["model_4"] = [np.sqrt(-model_4_scores).mean(), model_4, "random_forest"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_4"][0], 2), "\n")
display_scores(np.sqrt(-model_4_scores))

print("USED FEATURES: ", model_4["regressor"].n_features_in_)

model_4_feature_importances = pd.DataFrame(
    {
        "name": model_4["regressor"].feature_names_in_,
        "value": model_4["regressor"].feature_importances_,
        "abs_value": np.abs(model_4["regressor"].feature_importances_)
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_4_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_4_feature_importances.tail(6)[["name", "value"]], 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_4_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Support Vector Machine

# In[26]:


def svr_reg(x, y):
    """Support Vector Regression."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Set output to pandas
    pipeline_full.set_output(transform="pandas")

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", SVR()),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "regressor__C": [1, 10, 100],  # Regularization parameter
            "regressor__kernel": ['rbf'],  # Specifies the kernel type to be used in the algorithm
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_gs.fit(x, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    with open("output_reg/svr.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("output_reg/svr_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="output_reg/svr_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/svr_scores.pkl")
    joblib.dump(model_time, filename="output_reg/svr_time.pkl")
    


# In[115]:


#svr_reg(X_train, y_train)


# In[116]:


model_5 = joblib.load("output_reg/svr.pkl")

with open("output_reg/svr_best_params.pkl", 'rb') as f:
    model_5_best_params = pickle.load(f)

model_5_predictions = joblib.load("output_reg/svr_predictions.pkl")
model_5_scores = joblib.load("output_reg/svr_scores.pkl")
model_5_time = joblib.load("output_reg/svr_time.pkl")

print("TIME: ", round(model_5_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_5_predictions)))

scores["model_5"] = [np.sqrt(-model_5_scores).mean(), model_5, "svr"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_5"][0], 2), "\n")
display_scores(np.sqrt(-model_5_scores))

print("USED FEATURES: ", model_5["regressor"].n_features_in_)

print("OPTIMAL PARAMETERS:")
for param, value in model_5_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# In[29]:


def linear_svr_reg(x, y):
    """Linear Support Vector Regression."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first", dtype=bool)),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Set output to pandas
    pipeline_full.set_output(transform="pandas")

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", LinearSVR(random_state=123)),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "regressor__C": [0.1, 1, 10],  # Regularization parameter
            "regressor__epsilon": [0, 0.1, 0.2],  # Epsilon parameter in the epsilon-SVR model
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model_gs.fit(x, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    model_new = clone(model)
    model_scores = cross_val_score(model_new, x, y, scoring="neg_mean_squared_error", cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    with open("output_reg/linear_svr.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("output_reg/linear_svr_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="output_reg/linear_svr_predictions.pkl")
    joblib.dump(model_scores, filename="output_reg/linear_svr_scores.pkl")
    joblib.dump(model_time, filename="output_reg/linear_svr_time.pkl")


# In[117]:


#linear_svr_reg(X_train, y_train)


# In[118]:


model_6 = joblib.load("output_reg/linear_svr.pkl")

with open("output_reg/linear_svr_best_params.pkl", 'rb') as f:
    model_6_best_params = pickle.load(f)

model_6_predictions = joblib.load("output_reg/linear_svr_predictions.pkl")
model_6_scores = joblib.load("output_reg/linear_svr_scores.pkl")
model_6_time = joblib.load("output_reg/linear_svr_time.pkl")

print("TIME: ", round(model_6_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_6_predictions)))

scores["model_6"] = [np.sqrt(-model_6_scores).mean(), model_6, "linear_svr"]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_6"][0], 2), "\n")
display_scores(np.sqrt(-model_6_scores))

print("USED FEATURES: ", model_6["regressor"].n_features_in_)

print("OPTIMAL PARAMETERS:")
for param, value in model_6_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Best Model

# In[119]:


# Select best model from scores
best_key = min(scores, key=scores.get)
best_scores = scores[best_key]

print("Key: ", best_key, "\n")
print("BEST MODEL: ", best_scores[2], "\n")
print("Score: ", best_scores[0], "\n")


# In[120]:


# Bar plot of scores and model names
plt.figure(figsize=(10, 5))
sns.barplot(x=[score[2] for score in scores.values()], y=[score[0] for score in scores.values()])
plt.title("Scores of all models")
plt.savefig("assets/scores_reg.png", dpi=600, bbox_inches='tight')
plt.show()


# In[121]:


# Save best model
best_model = joblib.load(f"output_reg/{best_scores[2]}.pkl")

with open("output_reg/best_model_reg.pkl", "wb") as file:
    joblib.dump(best_model, file)


# In[122]:


# Test best model on test set
best_model_predictions = best_model.predict(X_test)
best_model_rmse = np.sqrt(mean_squared_error(y_test, best_model_predictions))
print("RMSE on test set: ", best_model_rmse, "\n")


# ## Classification

# In[123]:


df = pd.read_csv('project_1_train.csv')
df['Diabetes'].mask(df['Diabetes'] == 'Yes', 0, inplace=True)
df['Diabetes'].mask(df['Diabetes'] == 'No', 1, inplace=True)
df["Diabetes"] = df["Diabetes"].astype(float)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df = df.select_dtypes(include=numerics)

max_number_of_nas = 1500
newdf = df.loc[:, (df.isnull().sum(axis=0) <= max_number_of_nas)]

#newdf.dropna(inplace=True)
newdf = newdf.dropna(subset=["DirectChol", "Diabetes"])
newdf.info()


# In[124]:


y_clas = newdf['Diabetes']

x_clas = newdf.drop(columns=['Diabetes'])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(x_clas, y_clas, test_size=0.2, random_state=123)


# In[125]:


X_train_c


# In[126]:


y_train_c.value_counts()


# In[129]:


# Upsample the minority class
from sklearn.utils import resample

def upsample_data(X_train_c, y_train_c):
    """
    Upsample the minority class in the dataset.
    
    Parameters:
    X_train_c (DataFrame): The features of the training data.
    y_train_c (Series): The target of the training data, indicating the presence of Diabetes.
    
    Returns:
    DataFrame, Series: The upsampled features and target data.
    """

    # Combine the features and target into a single DataFrame for resampling
    train_data = pd.concat([X_train_c, y_train_c], axis=1)

    # Separate the majority and minority classes
    majority_class = train_data[train_data.Diabetes == 1.0]
    minority_class = train_data[train_data.Diabetes == 0.0]

    # Upsample the minority class
    minority_class_upsampled = resample(minority_class,
                                        replace=True,                  # sample with replacement
                                        n_samples=len(majority_class), # to match majority class size
                                        random_state=123)              # for reproducible results

    # Combine majority class with upsampled minority class
    upsampled_data = pd.concat([majority_class, minority_class_upsampled])

    # Separate features and target
    X_train_upsampled = upsampled_data.drop('Diabetes', axis=1)
    y_train_upsampled = upsampled_data['Diabetes']

    return X_train_upsampled, y_train_upsampled

X_train_c, y_train_c = upsample_data(X_train_c, y_train_c)


# In[130]:


y_train_c.value_counts()


# In[131]:


scores = {}  # containing scores and pipeline info


# ### Logistic Regression
# 

# In[132]:


def log_reg(x, y):
    """Define model 1"""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9)),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])


    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("classifier", LogisticRegression()),
    ])

    # possible fine-tuned parameters
    model_param = [
        # binary fit for each label
        {
            "classifier__multi_class": ["ovr"],  # binary fit for each label
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
            "classifier__C": [0.1],
        },
        # softmax regression
        {
            "classifier__multi_class": ["multinomial"],  # softmax regression
            "classifier__penalty": ["l2"],
            "classifier__solver": ["saga"],
            "classifier__C": [0.1],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="output_clf/log_reg_cl.pkl")

    # use the following to store the best params of grid search
    with open("output_clf/log_reg_cl_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions_cv, filename="output_clf/log_reg_cl_predictions_cv.pkl")
    joblib.dump(model_time, filename="output_clf/log_reg_cl_time.pkl")


# In[133]:


#log_reg(X_train_c, y_train_c)


# In[134]:


model_1 = joblib.load("output_clf/log_reg_cl.pkl")

with open("output_clf/log_reg_cl_best_params.pkl", 'rb') as f:
    model_1_best_params = pickle.load(f)

model_1_predictions_cv = joblib.load("output_clf/log_reg_cl_predictions_cv.pkl")
model_1_time = joblib.load("output_clf/log_reg_cl_time.pkl")

print("TIME: ", round(model_1_time, 2), " sec\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_c, model_1_predictions_cv), "\n")

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_c, model_1_predictions_cv))

model_1_score = f1_score(y_train_c, model_1_predictions_cv, average="weighted")
scores["model_1"] = [model_1_score, model_1, "log_reg_cl"]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_1_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_1_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# In[136]:


def log_reg_poly(x, y):
    """Define model 2"""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9)),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])


    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("classifier", LogisticRegression()),
    ])

    # possible fine-tuned parameters
    model_param = [
        # binary fit for each label
        {
            "classifier__multi_class": ["ovr"],  # binary fit for each label
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
            "classifier__C": [0.1],
        },
        # softmax regression
        {
            "classifier__multi_class": ["multinomial"],  # softmax regression
            "classifier__penalty": ["l2"],
            "classifier__solver": ["saga"],
            "classifier__C": [0.1],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="output_clf/log_reg_poly_cl.pkl")

    # use the following to store the best params of grid search
    with open("output_clf/log_reg_poly_cl_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions_cv, filename="output_clf/log_reg_poly_cl_predictions_cv.pkl")
    joblib.dump(model_time, filename="output_clf/log_reg_poly_cl_time.pkl")


# In[137]:


#log_reg_poly(X_train_c, y_train_c)


# In[138]:


model_2 = joblib.load("output_clf/log_reg_poly_cl.pkl")

with open("output_clf/log_reg_poly_cl_best_params.pkl", 'rb') as f:
    model_2_best_params = pickle.load(f)

model_2_predictions_cv = joblib.load("output_clf/log_reg_poly_cl_predictions_cv.pkl")
model_2_time = joblib.load("output_clf/log_reg_poly_cl_time.pkl")

print("TIME: ", round(model_2_time, 2), " sec\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_c, model_2_predictions_cv), "\n")

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_c, model_2_predictions_cv))

model_2_score = f1_score(y_train_c, model_2_predictions_cv, average="weighted")
scores["model_2"] = [model_2_score, model_2, "log_reg_poly_cl"]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_2_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_2_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Support Vector Machine

# In[139]:


def svc_clf(X, y):
    """Support Vector Machine Classification."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9)),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("classifier", SVC(random_state=123)),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "classifier__C": [0.1, 1, 10],  # Regularization parameter
            "classifier__kernel": ["rbf", "poly", "sigmoid"],  # Specifies the kernel type
            "classifier__gamma": ["scale", "auto"],  # Kernel coefficient
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(X, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    # Attention: Use a clone of the model for cross-validation predictions
    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, X, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    joblib.dump(model, filename="output_clf/svc_clf.pkl")
    with open("output_clf/svc_clf_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)
    joblib.dump(model_predictions_cv, filename="output_clf/svc_clf_predictions_cv.pkl")
    joblib.dump(model_time, filename="output_clf/svc_clf_time.pkl")


# In[140]:


#svc_clf(X_train_c, y_train_c)


# In[141]:


model_3 = joblib.load("output_clf/svc_clf.pkl")

with open("output_clf/svc_clf_best_params.pkl", 'rb') as f:
    model_3_best_params = pickle.load(f)

model_3_predictions_cv = joblib.load("output_clf/svc_clf_predictions_cv.pkl")
model_3_time = joblib.load("output_clf/svc_clf_time.pkl")

print("TIME: ", round(model_3_time, 2), " sec\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_c, model_3_predictions_cv), "\n")

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_c, model_3_predictions_cv))

model_3_score = f1_score(y_train_c, model_3_predictions_cv, average="weighted")
scores["model_3"] = [model_3_score, model_3, "svc_clf"]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_3_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_3_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# In[142]:


def linear_svc_clf(X, y):
    """Linear Support Vector Machine Classification."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9)),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("classifier", LinearSVC(random_state=123)),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "classifier__penalty": ["l2"],
            "classifier__loss": ["squared_hinge"],  # Typically used for linear SVM
            "classifier__C": [0.1, 1, 10],  # Regularization parameter
            "classifier__dual": [True, False],  # Dual or primal formulation
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(X, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    # Attention: Use a clone of the model for cross-validation predictions
    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, X, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    joblib.dump(model, filename="output_clf/linear_svc_clf.pkl")
    with open("output_clf/linear_svc_clf_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)
    joblib.dump(model_predictions_cv, filename="output_clf/linear_svc_clf_predictions_cv.pkl")
    joblib.dump(model_time, filename="output_clf/linear_svc_clf_time.pkl")


# In[143]:


#linear_svc_clf(X_train_c, y_train_c)


# In[144]:


model_4 = joblib.load("output_clf/linear_svc_clf.pkl")

with open("output_clf/linear_svc_clf_best_params.pkl", 'rb') as f:
    model_4_best_params = pickle.load(f)

model_4_predictions_cv = joblib.load("output_clf/linear_svc_clf_predictions_cv.pkl")
model_4_time = joblib.load("output_clf/linear_svc_clf_time.pkl")

print("TIME: ", round(model_4_time, 2), " sec\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_c, model_4_predictions_cv), "\n")

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_c, model_4_predictions_cv))

model_4_score = f1_score(y_train_c, model_4_predictions_cv, average="weighted")
scores["model_4"] = [model_4_score, model_4, "linear_svc_clf"]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_4_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_4_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Random Forest

# In[146]:


def random_forest_clf(X, y):
    """Random Forest Classification."""
    start = time.time()

    # Numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9)),
    ])

    # Categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),
        ("one_hot_encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
    ])

    # Combine numeric and categorical feature preparation
    pipeline_full = FeatureUnion(transformer_list=[
        ("pipeline_num", pipeline_num),
        ("pipeline_cat", pipeline_cat),
    ])

    # Define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("classifier", RandomForestClassifier(random_state=123)),
    ])

    # Possible fine-tuned parameters
    model_param = [
        {
            "classifier__n_estimators": [100, 200],  # Number of trees in the forest
            "classifier__max_depth": [None, 10, 20],  # Maximum depth of the tree
            "classifier__min_samples_leaf": [1, 2],  # Minimum number of samples required to be at a leaf node
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=5, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(X, y)

    model = model_gs.best_estimator_
    model_best_params = model_gs.best_params_

    # Attention: Use a clone of the model for cross-validation predictions
    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, X, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # Store model and results
    joblib.dump(model, filename="output_clf/random_forest_clf.pkl")
    with open("output_clf/random_forest_clf_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)
    joblib.dump(model_predictions_cv, filename="output_clf/random_forest_clf_predictions_cv.pkl")
    joblib.dump(model_time, filename="output_clf/random_forest_clf_time.pkl")


# In[147]:


#random_forest_clf(X_train_c, y_train_c)


# In[148]:


model_5 = joblib.load("output_clf/random_forest_clf.pkl")

with open("output_clf/random_forest_clf_best_params.pkl", 'rb') as f:
    model_5_best_params = pickle.load(f)

model_5_predictions_cv = joblib.load("output_clf/random_forest_clf_predictions_cv.pkl")
model_5_time = joblib.load("output_clf/random_forest_clf_time.pkl")

print("TIME: ", round(model_5_time, 2), " sec\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_c, model_5_predictions_cv), "\n")

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_c, model_5_predictions_cv))

model_5_score = f1_score(y_train_c, model_5_predictions_cv, average="weighted")
scores["model_5"] = [model_5_score, model_5, "random_forest_clf"]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_5_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_5_best_params.items():
    print(param, ": ", value, sep="")
print("\n")


# ### Best Model

# In[149]:


# Select best model from scores
best_key = max(scores, key=scores.get)
best_scores = scores[best_key]
y_pred_c = scores[best_key][1].predict(X_test_c)

print("Key: ", best_key, "\n")
print("BEST MODEL: ", best_scores[2], "\n")
print("Score: ", best_scores[0], "\n")


# In[150]:


# Bar plot of scores and model names
plt.figure(figsize=(10, 5))
sns.barplot(x=[score[2] for score in scores.values()], y=[score[0] for score in scores.values()])
plt.title("Scores of all models")
plt.savefig("assets/scores_clf.png", dpi=600, bbox_inches='tight')
plt.show()


# In[151]:


# Show confusion matrix of best model
sns.heatmap(confusion_matrix(y_test_c, y_pred_c), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion matrix of best model")
plt.savefig("assets/confusion_matrix.png", dpi=600, bbox_inches='tight')
plt.show()


# In[152]:


# Print classification report of best model
print(classification_report(y_test_c, y_pred_c, target_names=["Diabetes", "No Diabetes"]))


# In[153]:


# Save best model
best_model = joblib.load(f"output_clf/{best_scores[2]}.pkl")

with open("output_clf/best_model_clf.pkl", "wb") as file:
    joblib.dump(best_model, file)

