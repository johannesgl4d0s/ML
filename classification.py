"""
Some machine learning techniquers (logistic regression, random forest, support vector machines, k neighbors) applied to mnist dataset
ideas from Hands-On Machin-Learning with Scikit-Learn & TensorFlow (Aurelien Geron)

I used python 3.11
add folder (useful_files) to use the files of that folder: settings - project xy - project structure - add content root
(new folder is not allowed to intersect with existing root)

set models_to_run to True if you want to train the model, otherwise the stored model (if stored before) is used
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
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
play = 0  # 0...original dataset, 1...small dataset, 2...tiny dataset

if play == 0:
    mnist = pd.read_csv("mnist.csv")
    train_rows = 60000
elif play == 1:
    mnist = pd.read_csv("mnist_small.csv")
    train_rows = 6000
else:
    mnist = pd.read_csv("mnist_tiny.csv")
    train_rows = 600

# define label attribute
label_attribute = "target"

# remove rows where label attribute has na entries
# this is done for mnist (train+test sets), for new data which is used for prediction only, this is not necessary
mnist.dropna(axis=0, subset=label_attribute, inplace=True)

# define train and test set for x (features) and y (label)
x_train, x_test = mnist[:train_rows].copy(), mnist[train_rows:].copy()
del mnist

y_train = x_train[label_attribute].copy()
x_train.drop(label_attribute, axis=1, inplace=True)
y_train_5 = (y_train == 5)

y_test = x_test[label_attribute].copy()
x_test.drop(label_attribute, axis=1, inplace=True)
y_test_5 = (y_test == 5)

print("SPLITTING:")
print(len(x_train), " train + ", len(x_test), " test\n", sep="")

print("INFO:")
print(x_train.info(), "\n")

# see an instance
show_me_instance = False
if show_me_instance:
    some_index = 10
    print(x_train.iloc[some_index].head())
    print(x_train.iloc[some_index].tail())

    some_digit = x_train.loc[some_index]
    some_digit = pd.to_numeric(some_digit)

    some_digit_image = some_digit.values.reshape(28, 28)
    print(some_digit_image, "\n")
    print(y_train.loc[some_index], "\n")

    # use color map binary (black white), use interpolation
    plt.imshow(some_digit_image, cmap="binary", interpolation="nearest")
    plt.axis("off")

    plt.show()

# endregion


# region training
####################################################################################################################################################

models_to_run = {
    "model_1": True,  # logistic regression 5-detector
    "model_2": True,  # logistic regression
    "model_3": True,  # support vector classifier
    "model_4": True,  # random forest classifier
    "model_5": True,  # k neighbors classifier
}

# dict where model name, model score and model are stored for 5-detector
scores_5_detector = {}

# dict where model name, model score and model are stored
scores = {}

# region 1 logistic regression 5-detector
############################

print("MODEL 1 (LOGISTIC REGRESSION 5-DETECTOR)\n")


def run_model_1(x, y):
    """Define model 1"""
    start = time.time()

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", MinMaxScaler()),  # scale to interval [0, 1]
        ("pca", PCA()),
        ("classifiier", LogisticRegression()),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "pca__n_components": [0.9],  # [0.85, 0.9, 0.95, 0.99, 0.999]
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, n_jobs=-1, scoring="f1_weighted")

    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="model_1.pkl")

    # use the following to store the best params of grid search
    with open("model_1_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_1_predictions.pkl")
    joblib.dump(model_predictions_cv, filename="model_1_predictions_cv.pkl")
    joblib.dump(model_time, filename="model_1_time.pkl")


if models_to_run["model_1"]:
    run_model_1(x_train, y_train_5)

model_1 = joblib.load("model_1.pkl")

with open("model_1_best_params.pkl", 'rb') as f:
    model_1_best_params = pickle.load(f)

model_1_predictions = joblib.load("model_1_predictions.pkl")
model_1_predictions_cv = joblib.load("model_1_predictions_cv.pkl")
model_1_time = joblib.load("model_1_time.pkl")

print("TIME: ", round(model_1_time, 2), " sec\n")

print("CONFUSION MATRIX:")
print(confusion_matrix(y_train_5, model_1_predictions), "\n")
print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train_5, model_1_predictions_cv), "\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_train_5, model_1_predictions))

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train_5, model_1_predictions_cv))

model_1_score = f1_score(y_train_5, model_1_predictions_cv, average="weighted")
scores_5_detector["model_1"] = [model_1_score, model_1]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_1_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_1_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 2 logistic regression
############################

print("MODEL 2 (LOGISTIC REGRESSION)\n")


def run_model_2(x, y):
    """Define model 2"""
    start = time.time()

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", MinMaxScaler()),  # scale to interval [0, 1]
        ("pca", PCA()),
        ("classifier", LogisticRegression()),
    ])

    # possible fine-tuned parameters
    model_param = [
        # binary fit for each label
        {
            "pca__n_components": [0.9],
            "classifier__multi_class": ["ovr"],  # binary fit for each label
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
            "classifier__C": [0.1],
        },
        # softmax regression
        {
            "pca__n_components": [0.9],
            "classifier__multi_class": ["multinomial"],  # softmax regression
            "classifier__penalty": ["l2"],
            "classifier__solver": ["saga"],
            "classifier__C": [0.1],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="model_2.pkl")

    # use the following to store the best params of grid search
    with open("model_2_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_2_predictions.pkl")
    joblib.dump(model_predictions_cv, filename="model_2_predictions_cv.pkl")
    joblib.dump(model_time, filename="model_2_time.pkl")


if models_to_run["model_2"]:
    run_model_2(x_train, y_train)

model_2 = joblib.load("model_2.pkl")

with open("model_2_best_params.pkl", 'rb') as f:
    model_2_best_params = pickle.load(f)

model_2_predictions = joblib.load("model_2_predictions.pkl")
model_2_predictions_cv = joblib.load("model_2_predictions_cv.pkl")
model_2_time = joblib.load("model_2_time.pkl")

print("TIME: ", round(model_2_time, 2), " sec\n")

print("CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_2_predictions), "\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_2_predictions_cv), "\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_train, model_2_predictions))

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train, model_2_predictions_cv))

model_2_score = f1_score(y_train, model_2_predictions_cv, average="weighted")
scores["model_2"] = [model_2_score, model_2]
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_2_score, 2), "\n")

print("OPTIMAL PARAMETERS:")
for param, value in model_2_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 3 support vector classifier
############################

print("MODEL 3 (SUPPORT VECTOR CLASSIFIER)\n")


def run_model_3(x, y):
    """Define model 3"""
    start = time.time()

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", MinMaxScaler()),  # scale to interval [0, 1]
        ("pca", PCA()),
        ("classifier", SVC()),
    ])

    # possible fine-tuned parameters
    model_param = [
        # softmax regression
        {
            "pca__n_components": [0.9],
            "classifier__kernel": ["rbf"],
            "classifier__degree": [2],  # [2, 3],
            "classifier__C": [10],  # [1, 1e-1, 1e-2, 10, 100],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="model_3.pkl")

    # use the following to store the best params of grid search
    with open("model_3_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_3_predictions.pkl")
    joblib.dump(model_predictions_cv, filename="model_3_predictions_cv.pkl")
    joblib.dump(model_time, filename="model_3_time.pkl")


if models_to_run["model_3"]:
    run_model_3(x_train, y_train)

model_3 = joblib.load("model_3.pkl")

with open("model_3_best_params.pkl", 'rb') as f:
    model_3_best_params = pickle.load(f)

model_3_predictions = joblib.load("model_3_predictions.pkl")
model_3_predictions_cv = joblib.load("model_3_predictions_cv.pkl")
model_3_time = joblib.load("model_3_time.pkl")

print("TIME: ", round(model_3_time, 2), " sec\n")

print("CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_3_predictions), "\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_3_predictions_cv), "\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_train, model_3_predictions))

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train, model_3_predictions_cv))

model_3_score = f1_score(y_train, model_3_predictions_cv, average="weighted")
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_3_score, 2), "\n")
scores["model_3"] = [model_3_score, model_3]

print("OPTIMAL PARAMETERS:")
for param, value in model_3_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 4 random forest classifier
############################

print("MODEL 4 (RANDOM FOREST CLASSIFIER)\n")


def run_model_4(x, y):
    """Define model 4"""
    start = time.time()

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", MinMaxScaler()),  # scale to interval [0, 1]
        ("pca", PCA()),
        ("classifier", RandomForestClassifier(random_state=123)),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "pca__n_components": [0.9],
            "classifier__n_estimators": [600],  # [10, 50, 100, 200, 400, 600, 800],
            "classifier__max_depth": [20],  # [None, 5, 10, 20, 30, 50],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="model_4.pkl")

    # use the following to store the best params of grid search
    with open("model_4_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_4_predictions.pkl")
    joblib.dump(model_predictions_cv, filename="model_4_predictions_cv.pkl")
    joblib.dump(model_time, filename="model_4_time.pkl")


if models_to_run["model_4"]:
    run_model_4(x_train, y_train)

model_4 = joblib.load("model_4.pkl")

with open("model_4_best_params.pkl", 'rb') as f:
    model_4_best_params = pickle.load(f)

model_4_predictions = joblib.load("model_4_predictions.pkl")
model_4_predictions_cv = joblib.load("model_4_predictions_cv.pkl")
model_4_time = joblib.load("model_4_time.pkl")

print("TIME: ", round(model_4_time, 2), " sec\n")

print("CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_4_predictions), "\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_4_predictions_cv), "\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_train, model_4_predictions))

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train, model_4_predictions_cv))

model_4_score = f1_score(y_train, model_4_predictions_cv, average="weighted")
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_4_score, 2), "\n")
scores["model_4"] = [model_4_score, model_4]

print("OPTIMAL PARAMETERS:")
for param, value in model_4_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region 5 k neighbors classifier
############################

print("MODEL 5 (K NEIGHBORS CLASSIFIER)\n")


def run_model_5(x, y):
    """Define model 5"""
    start = time.time()

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # missing values are replaced by column median of train_set
        ("scaler", MinMaxScaler()),  # scale to interval [0, 1]
        ("pca", PCA()),
        ("classifier", KNeighborsClassifier()),
    ])

    # possible fine-tuned parameters
    model_param = [
        {
            "pca__n_components": [0.9],
            "classifier__n_neighbors": [5],  # [5, 7, 9],
            "classifier__weights": ["distance"],
            "classifier__p": [1],  # [1, 2],
        }
    ]

    model_gs = GridSearchCV(base_model, model_param, cv=10, n_jobs=-1, scoring="f1_weighted")
    model_gs.fit(x, y)

    model = model_gs.best_estimator_

    model_best_params = model_gs.best_params_

    model_predictions = model.predict(x)

    # attention: it is not allowed to use a fitted model (like model) in cross_val_predict because information gets through
    # unfitted model (clone(model)) is used for cross validation

    model_new = clone(model)
    model_predictions_cv = cross_val_predict(model_new, x, y, cv=3, n_jobs=-1)

    end = time.time()
    model_time = end - start

    # store model
    joblib.dump(model, filename="model_5.pkl")

    # use the following to store the best params of grid search
    with open("model_5_best_params.pkl", "wb") as ff:
        pickle.dump(model_best_params, ff)

    joblib.dump(model_predictions, filename="model_5_predictions.pkl")
    joblib.dump(model_predictions_cv, filename="model_5_predictions_cv.pkl")
    joblib.dump(model_time, filename="model_5_time.pkl")


if models_to_run["model_5"]:
    run_model_5(x_train, y_train)

model_5 = joblib.load("model_5.pkl")

with open("model_5_best_params.pkl", 'rb') as f:
    model_5_best_params = pickle.load(f)

model_5_predictions = joblib.load("model_5_predictions.pkl")
model_5_predictions_cv = joblib.load("model_5_predictions_cv.pkl")
model_5_time = joblib.load("model_5_time.pkl")

print("TIME: ", round(model_5_time, 2), " sec\n")

print("CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_5_predictions), "\n")

print("CROSS VALIDATION CONFUSION MATRIX:")
print(confusion_matrix(y_train, model_5_predictions_cv), "\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_train, model_5_predictions))

print("CROSS CLASSIFICATION REPORT:")
print(classification_report(y_train, model_5_predictions_cv))

model_5_score = f1_score(y_train, model_5_predictions_cv, average="weighted")
print("VALUE FOR COMPARISON: WEIGHTED F1_SCORE:", round(model_5_score, 2), "\n")
scores["model_5"] = [model_5_score, model_5]

print("OPTIMAL PARAMETERS:")
for param, value in model_5_best_params.items():
    print(param, ": ", value, sep="")
print("\n")

# endregion


# region testing 5-detector
####################################################################################################################################################

print("TEST FINAL MODEL 5-DETECTOR\n")

scores_5_detector_max = max(scores_5_detector, key=lambda dict_key: scores_5_detector[dict_key][0])

print("SCORES OF CROSS VALIDATION AGAIN")
for k, v in scores_5_detector.items():
    print(k, round(v[0], 2))

print("BEST SCORE OF CROSS VALIDATION")
print(scores_5_detector_max, round(scores_5_detector[scores_5_detector_max][0], 2), "\n")

model_5_detector_final = scores_5_detector[scores_5_detector_max][1]
model_5_detector_final_predictions_test = model_5_detector_final.predict(x_test)

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test_5, model_5_detector_final_predictions_test))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test_5, model_5_detector_final_predictions_test))

# endregion

# region testing
####################################################################################################################################################

print("TEST FINAL MODEL \n")

scores_max = max(scores, key=lambda dict_key: scores[dict_key][0])

print("SCORES OF CROSS VALIDATION AGAIN")
for k, v in scores.items():
    print(k, round(v[0], 2))

print("BEST SCORE OF CROSS VALIDATION")
print(scores_max, round(scores[scores_max][0], 2), "\n")

model_final = scores[scores_max][1]
model_final_predictions_test = model_final.predict(x_test)

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, model_final_predictions_test))
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, model_final_predictions_test))

# endregion
#%%
