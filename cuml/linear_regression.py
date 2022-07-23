import cudf
from cuml import make_regression, train_test_split
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.metrics.regression import r2_score
from sklearn.linear_model import LinearRegression as skLinearRegression

print(f'[trace] Define Parameters')
n_samples = 2**20 #If you are running on a GPU with less than 16GB RAM, please change to 2**19 or you could run out of memory
n_features = 399
random_state = 23

print(f'[trace] Generate Data')
X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)

X = cudf.DataFrame(X)
y = cudf.DataFrame(y)[0]
X_cudf, X_cudf_test, y_cudf, y_cudf_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)

# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
X_train = X_cudf.to_pandas()
X_test = X_cudf_test.to_pandas()
y_train = y_cudf.to_pandas()
y_test = y_cudf_test.to_pandas()

print(f'[trace] Fit, predict and evaluate')
ols_sk = skLinearRegression(fit_intercept=True,
                            normalize=True,
                            n_jobs=-1)

ols_sk.fit(X_train, y_train)
predict_sk = ols_sk.predict(X_test)
print(f'[trace] evaluating r2 score for scikit learn')
r2_score_sk = r2_score(y_cudf_test, predict_sk)

print(f"[trace] cuML Model: Fit, predict and evaluate")
ols_cuml = cuLinearRegression(fit_intercept=True,
                              normalize=True,
                              algorithm='eig')

ols_cuml.fit(X_cudf, y_cudf)
predict_cuml = ols_cuml.predict(X_cudf_test)
print(f'[trace] evaluating r2 score for cuML')
r2_score_cuml = r2_score(y_cudf_test, predict_cuml)

print(f'[trace] Compare Results:')
print("R^2 score (SKL):  %s" % r2_score_sk)
print("R^2 score (cuML): %s" % r2_score_cuml)
