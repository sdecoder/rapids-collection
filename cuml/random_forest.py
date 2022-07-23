'''
Random Forest and Pickling
The Random Forest algorithm is a classification method which builds several decision trees, and aggregates each of their outputs to make a prediction.

In this notebook we will train a scikit-learn and a cuML Random Forest Classification model. Then we save the cuML model for future use with Python's pickling mechanism and demonstrate how to re-load it for prediction. We also compare the results of the scikit-learn, non-pickled and pickled cuML models.

Note that the underlying algorithm in cuML for tree node splits differs from that used in scikit-learn.

For information on converting your dataset to cuDF format, refer to the cuDF documentation

For additional information cuML's random forest model: https://docs.rapids.ai/api/cuml/stable/api.html#random-forest
'''


import cudf
import numpy as np
import pandas as pd
import pickle

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


print(f'[trace] Define Parameters')
# The speedup obtained by using cuML'sRandom Forest implementation
# becomes much higher when using larger datasets. Uncomment and use the n_samples
# value provided below to see the difference in the time required to run
# Scikit-learn's vs cuML's implementation with a large dataset.

# n_samples = 2*17
n_samples = 2**12
n_features = 399
n_info = 300
data_type = np.float32

print(f'[trace] Generate Data')
X,y = make_classification(n_samples=n_samples,
                          n_features=n_features,
                          n_informative=n_info,
                          random_state=123, n_classes=2)

X = pd.DataFrame(X.astype(data_type))
# cuML Random Forest Classifier requires the labels to be integers
y = pd.Series(y.astype(np.int32))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=0)


print(f'[trace] move data to gpu')
X_cudf_train = cudf.DataFrame.from_pandas(X_train)
X_cudf_test = cudf.DataFrame.from_pandas(X_test)
y_cudf_train = cudf.Series(y_train.values)

print(f'[trace] Scikit-learn Model')
sk_model = skrfc(n_estimators=40,
                 max_depth=16,
                 max_features=1.0,
                 random_state=10)

print(f'[trace] start to *train* the skilearn model')
sk_model.fit(X_train, y_train)
print(f'[trace] start to *evaluate* the skilearn model')
sk_predict = sk_model.predict(X_test)
sk_acc = accuracy_score(y_test, sk_predict)

print(f'[trace] start to *fit* cuML Model')
cuml_model = curfc(n_estimators=40,
                   max_depth=16,
                   max_features=1.0,
                   random_state=10)

cuml_model.fit(X_cudf_train, y_cudf_train)

print(f'[trace] start to *evaluate* cuML Model')
fil_preds_orig = cuml_model.predict(X_cudf_test)
fil_acc_orig = accuracy_score(y_test.to_numpy(), fil_preds_orig)

print(f'[trace] Pickle the cuML random forest classification model')
filename = 'cuml_random_forest_model.sav'
# save the trained cuml model into a file
pickle.dump(cuml_model, open(filename, 'wb'))
# delete the previous model to ensure that there is no leakage of pointers.
# this is not strictly necessary but just included here for demo purposes.
del cuml_model
# load the previously saved cuml model from a file
pickled_cuml_model = pickle.load(open(filename, 'rb'))

print(f'[trace] Predict using the pickled model')
pred_after_pickling = pickled_cuml_model.predict(X_cudf_test)
fil_acc_after_pickling = accuracy_score(y_test.to_numpy(), pred_after_pickling)

print(f'[trace] Compare Results')
print("CUML accuracy of the RF model before pickling: %s" % fil_acc_orig)
print("CUML accuracy of the RF model after pickling: %s" % fil_acc_after_pickling)
print("SKL accuracy: %s" % sk_acc)
print("CUML accuracy before pickling: %s" % fil_acc_orig)
