"""
Some machine learning techniquers (regression with linear regression, decision tree, random forest and support vector machines) applied to california housing dataset
ideas from Hands-On Machin-Learning with Scikit-Learn & TensorFlow (Aurelien Geron)
I used python 3.11
add folder (useful_files) to use the files of that folder: settings - project xy - project structure - add content root
(new folder is not allowed to intersect with existing root)

set models_to_run to True if you want to train the model, otherwise the stored model (if stored before) is used
"""

# add folder (useful_files) to use transformers.py and adder.py: settings - project xy - project structure - add content root
# new folder is not allowed to intersect with existing root

from transformers import Selector, MyPowerTransformer
from adder import NumAttributesAdder, CatAttributesAdder
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
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR, LinearSVR
from sklearn.base import clone
import pickle

np.set_printoptions(linewidth=10000)
np.set_printoptions(threshold=10000)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.min_rows", 100)
pd.set_option("display.width", 0)

# region load data, define train and test set, remove rows with na-entries in label, first look at train set
####################################################################################################################################################

# load data
housing = pd.read_csv("../housing/housing.csv")

# define label attribute
label_attribute = "median_house_value"

# remove rows where label attribute has na entries
# this is done for housing (train+test sets), for new data which is used for prediction only, this is not necessary
housing.dropna(axis=0, subset=label_attribute, inplace=True)

# define train and test set for x (features) and y (label)
x_train, x_test = train_test_split(housing, test_size=0.2, random_state=123)
del housing

print("SPLITTING:")
print(len(x_train), " train + ", len(x_test), " test\n", sep="")

print("INFO:")
print(x_train.info(), "\n")  # na-entries for total_bedrooms


def details(data):
    """have a look at data"""
    print("FIRST ENTRIES:")
    print(data.head(), "\n")

    print("OCEA_PROXIMITY VALUES:")
    print(data["ocean_proximity"].value_counts(), "\n")

    print("DESCRIBE:")
    print(round(data.describe(), 2), "\n")

    corr_matrix = data.corr(numeric_only=True)
    print("CORR MATRIX:")
    print(round(corr_matrix[label_attribute].sort_values(ascending=False), 2), "\n")

    data.hist(bins=50)

    scatter_matrix(data)

    data.plot(kind="scatter", x="median_income", y=label_attribute, alpha=0.1)

    data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=data["population"] / 100,
              label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

    plt.show()


# see more details
show_me_details = False
if show_me_details:
    details(x_train)
    # housing_median_age and median_house_value are cropped
    # median income cropped too but not too much
    # median-income is the most important variable

# try to add some variables temporarily
x_train_new = x_train.copy()
x_train_new["rooms_per_household"] = x_train_new["total_rooms"] / x_train_new["households"]
x_train_new["population_per_household"] = x_train_new["population"] / x_train_new["households"]
x_train_new["bedrooms_per_room"] = x_train_new["total_bedrooms"] / x_train_new["total_rooms"]

if show_me_details:
    details(x_train_new)
    # housing_median_age and median_house_value are cropped
    # median income cropped too but not too much
    # median-income is the most important variable (corr-matrix)

    # rooms_per_households, population_per_household and bedrooms_per_room is useful (see corr-matrix)

del x_train_new

y_train = x_train[label_attribute].copy()
x_train.drop(label_attribute, axis=1, inplace=True)

y_test = x_test[label_attribute].copy()
x_test.drop(label_attribute, axis=1, inplace=True)


# endregion


# region training
####################################################################################################################################################


def display_scores(model_score):
    """print the list of scores, the mean and the standard deviation"""
    print("SCORES OF CROSS VALIDATION:")
    print(np.round(model_score, decimals=1))
    print("MEAN SCORE: %0.1f" % model_score.mean())
    print("STD SCORE: %0.1f\n" % model_score.std())


models_to_run = {
    "model_1": False,  # linear regression
    "model_2": False,  # linear regression
    "model_3": False,  # decision tree
    "model_4": False,  # random forest
    "model_5": False,  # linear svr
    "model_6": False,  # svr
}

# dict where model name, model score and model are stored
scores = {}

# region 1 linear regression
############################

print("MODEL 1 (LINEAR REGRESSION)\n")


def run_model_1(x, y):
    """define model 1"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),  # select numeric attributes
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1
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
    joblib.dump(model, filename="model_1.pkl")
    joblib.dump(model_predictions, filename="model_1_predictions.pkl")
    joblib.dump(model_scores, filename="model_1_scores.pkl")
    joblib.dump(model_time, filename="model_1_time.pkl")


if models_to_run["model_1"]:
    run_model_1(x_train, y_train)

model_1 = joblib.load("model_1.pkl")
model_1_predictions = joblib.load("model_1_predictions.pkl")
model_1_scores = joblib.load("model_1_scores.pkl")
model_1_time = joblib.load("model_1_time.pkl")

print("TIME: ", round(model_1_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_1_predictions)))

scores["model_1"] = [np.sqrt(-model_1_scores).mean(), model_1]
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

# endregion


# region 2 linear regression
############################

print("MODEL 2 (LINEAR REGRESSION)\n")


def run_model_2(x, y):
    """define model 2"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),  # select numeric attributes
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("poly_features", PolynomialFeatures(include_bias=False)),  # polynomial features
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1
        ("power_transformer", MyPowerTransformer()),  # numerical features can be transformed via yeo-johnson method, here None is used
        ("pca", PCA(n_components=0.9)),  # pca with 90% of total variance
    ])

    # categorical feature preparation
    pipeline_cat = Pipeline([
        ("cat_attributes_adder", CatAttributesAdder()),  # adds some attributes (before Selector because we need numeric attributes)
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

    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("regressor", TransformedTargetRegressor()),  # define regressor later
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "pipeline_full__pipeline_num__power_transformer__method": ["yeo-johnson"],  # [None, "yeo-johnson"],
            "regressor__transformer": [PowerTransformer()],  # [None, PowerTransformer()],
            "regressor__regressor": [LinearRegression()],  # [LinearRegression(), ElasticNet(alpha=0.1, l1_ratio=0.8)],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)

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
    joblib.dump(model, filename="model_2.pkl")

    # use the following to store the best params of grid search
    with open("model_2_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_2_predictions.pkl")
    joblib.dump(model_scores, filename="model_2_scores.pkl")
    joblib.dump(model_time, filename="model_2_time.pkl")


if models_to_run["model_2"]:
    run_model_2(x_train, y_train)

model_2 = joblib.load("model_2.pkl")

with open("model_2_best_params.pkl", 'rb') as f:
    model_2_best_params = pickle.load(f)

model_2_predictions = joblib.load("model_2_predictions.pkl")
model_2_scores = joblib.load("model_2_scores.pkl")
model_2_time = joblib.load("model_2_time.pkl")

print("TIME: ", round(model_2_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_2_predictions)))

scores["model_2"] = [np.sqrt(-model_2_scores).mean(), model_2]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_2"][0], 2), "\n")
display_scores(np.sqrt(-model_2_scores))

print("USED FEATURES: ", model_2["regressor"].regressor_.n_features_in_)

model_2_feature_importances = pd.DataFrame(
    {
        "name": model_2["regressor"].feature_names_in_,
        "value": model_2["regressor"].regressor_.coef_,
        "abs_value": np.abs(model_2["regressor"].regressor_.coef_)
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_2_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_2_feature_importances.tail(6)[["name", "value"]], 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_2_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 3 decision tree
############################

print("MODEL 3 (DECISION TREE) \n")


def run_model_3(x, y):
    """define model 3"""
    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
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
            "regressor__min_samples_leaf": [0.0009],  # [0.0008, 0.0009, 0.001],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)

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
    joblib.dump(model, filename="model_3.pkl")

    # use the following to store the best params of grid search
    with open("model_3_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_3_predictions.pkl")
    joblib.dump(model_scores, filename="model_3_scores.pkl")
    joblib.dump(model_time, filename="model_3_time.pkl")


if models_to_run["model_3"]:
    run_model_3(x_train, y_train)

model_3 = joblib.load("model_3.pkl")

with open("model_3_best_params.pkl", 'rb') as f:
    model_3_best_params = pickle.load(f)

model_3_predictions = joblib.load("model_3_predictions.pkl")
model_3_scores = joblib.load("model_3_scores.pkl")
model_3_time = joblib.load("model_3_time.pkl")

print("TIME: ", round(model_3_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_3_predictions)))

scores["model_3"] = [np.sqrt(-model_3_scores).mean(), model_3]
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

# endregion


# region 4 random forest
############################

print("MODEL 4 (RANDOM FOREST) \n")


def run_model_4(x, y):
    """define model 4"""

    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
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

    # define full pipeline
    base_model = Pipeline([
        ("pipeline_full", pipeline_full),
        ("feature_selection", SelectFromModel(RandomForestRegressor(random_state=123))),
        ("regressor", RandomForestRegressor(random_state=123)),
    ])

    # set output to pandas, so we can use pandas tools again (e.g. for feature_names_in_)
    base_model.set_output(transform="pandas")

    # possible fine-tuned parameters
    model_param = [
        {
            "feature_selection__threshold": ["0.9*median"],  # ["median", "1.1*median"]
            "feature_selection__estimator__max_depth": [5],
            "feature_selection__estimator__n_estimators": [50],
            "feature_selection__estimator__max_leaf_nodes": [16],
            "regressor__n_estimators": [140],  # [100, 120, 130, 140, , 150, 160],
            # "regressor__max_features": [8],  # useless because of feature selection
            "regressor__max_depth": [None],  # [None, 15, 20],
            "regressor__min_samples_split": [5],  # [5, 7, 9],
            "regressor__bootstrap": [True],  # [True, False],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)

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
    joblib.dump(model, filename="model_4.pkl")

    # use the following to store the best params of grid search
    with open("model_4_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_4_predictions.pkl")
    joblib.dump(model_scores, filename="model_4_scores.pkl")
    joblib.dump(model_time, filename="model_4_time.pkl")


if models_to_run["model_4"]:
    run_model_4(x_train, y_train)

model_4 = joblib.load("model_4.pkl")

with open("model_4_best_params.pkl", 'rb') as f:
    model_4_best_params = pickle.load(f)

model_4_predictions = joblib.load("model_4_predictions.pkl")
model_4_scores = joblib.load("model_4_scores.pkl")
model_4_time = joblib.load("model_4_time.pkl")

print("TIME: ", round(model_4_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_4_predictions)))

scores["model_4"] = [np.sqrt(-model_4_scores).mean(), model_4]
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

# endregion


# region 5 linear svr
############################

print("MODEL 5 (LINEAR SVR) \n")


def run_model_5(x, y):
    """define model 4"""

    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1
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
        ("regressor", LinearSVR(dual="auto")),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "regressor__epsilon": [0.23],
            "regressor__C": [19000],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)

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
    joblib.dump(model, filename="model_5.pkl")

    # use the following to store the best params of grid search
    with open("model_5_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_5_predictions.pkl")
    joblib.dump(model_scores, filename="model_5_scores.pkl")
    joblib.dump(model_time, filename="model_5_time.pkl")


if models_to_run["model_5"]:
    run_model_5(x_train, y_train)

model_5 = joblib.load("model_5.pkl")

with open("model_5_best_params.pkl", 'rb') as f:
    model_5_best_params = pickle.load(f)

model_5_predictions = joblib.load("model_5_predictions.pkl")
model_5_scores = joblib.load("model_5_scores.pkl")
model_5_time = joblib.load("model_5_time.pkl")

print("TIME: ", round(model_5_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_5_predictions)))

scores["model_5"] = [np.sqrt(-model_5_scores).mean(), model_5]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_5"][0], 2), "\n")
display_scores(np.sqrt(-model_5_scores))

print("USED FEATURES: ", model_5["regressor"].n_features_in_)

model_5_feature_importances = pd.DataFrame(
    {
        "name": model_5["regressor"].feature_names_in_,
        "value": model_5["regressor"].coef_,
        "abs_value": np.abs(model_5["regressor"].coef_)
    }).sort_values(by="abs_value")

print("FEATURE VALUE IMPORTANCE:")
print("LOWEST ABS SCORE:")
print(round(model_5_feature_importances.head(6)[["name", "value"]], 2), "\n")
print("HIGHEST ABS SCORE:")
print(round(model_5_feature_importances.tail(6)[["name", "value"]], 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_5_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 6 svr
############################

print("MODEL 6 (SVR) \n")


def run_model_6(x, y):
    """define model 6"""

    start = time.time()

    # numeric feature preparation
    pipeline_num = Pipeline([
        ("selector", Selector("num features")),
        ("num_attributes_adder", NumAttributesAdder()),  # adds some attributes
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", StandardScaler()),  # scale to mean 0 and std 1
    ])

    # categorical feature preparation
    pipeline_cat = Pipeline([
        ("selector", Selector("cat features")),  # select categorical attributes except label attribute
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
        ("regressor", SVR()),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "regressor__C": [1e4],
            "regressor__kernel": ["rbf"],
            "regressor__degree": [2],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, scoring="neg_root_mean_squared_error", n_jobs=-1)

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
    joblib.dump(model, filename="model_6.pkl")

    # use the following to store the best params of grid search
    with open("model_6_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_6_predictions.pkl")
    joblib.dump(model_scores, filename="model_6_scores.pkl")
    joblib.dump(model_time, filename="model_6_time.pkl")


if models_to_run["model_6"]:
    run_model_6(x_train, y_train)

model_6 = joblib.load("model_6.pkl")

with open("model_6_best_params.pkl", 'rb') as f:
    model_6_best_params = pickle.load(f)

model_6_predictions = joblib.load("model_6_predictions.pkl")
model_6_scores = joblib.load("model_6_scores.pkl")
model_6_time = joblib.load("model_6_time.pkl")

print("TIME: ", round(model_6_time, 2), " sec\n")

print("RMSE: %0.1f\n" % np.sqrt(mean_squared_error(y_train, model_6_predictions)))

scores["model_6"] = [np.sqrt(-model_6_scores).mean(), model_6]
print("VALUE FOR COMPARISON: CV RMSE", round(scores["model_6"][0], 2), "\n")
display_scores(np.sqrt(-model_6_scores))

print("USED FEATURES: ", model_6["regressor"].n_features_in_)

# the following is not possible for svr with nonlinear kernel!
# model_6_feature_importances = pd.DataFrame(
#     {
#         "name": model_6["regressor"].feature_names_in_,
#         "value": model_6["regressor"].coef_,
#         "abs_value": np.abs(model_6["regressor"].coef_)
#     }).sort_values(by="abs_value")
#
# print("FEATURE VALUE IMPORTANCE:")
# print("LOWEST ABS SCORE:")
# print(round(model_6_feature_importances.head(6)[["name", "value"]], 2), "\n")
# print("HIGHEST ABS SCORE:")
# print(round(model_6_feature_importances.tail(6)[["name", "value"]], 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_6_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region testing
####################################################################################################################################################

print("TEST FINAL MODEL \n")

scores_min = min(scores, key=lambda dict_key: scores[dict_key][0])

print("RMSE OF CROSS VALIDATION AGAIN")
for k, v in scores.items():
    print(k, round(v[0], 2))

print("BEST RMSE OF CROSS VALIDATION")
print(scores_min, round(scores[scores_min][0], 2), "\n")

model_final = scores[scores_min][1]
model_final_predictions_test = model_final.predict(x_test)
print("RMSE FOR TEST SET: %0.1f\n" % np.sqrt(mean_squared_error(y_test, model_final_predictions_test)))

# endregion