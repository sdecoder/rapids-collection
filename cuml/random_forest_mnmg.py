'''
Random Forests Multi-node, Multi-GPU demo
The experimental cuML multi-node, multi-GPU (MNMG) implementation of random forests leverages Dask to do embarrassingly-parallel model fitting. For a random forest with N trees being fit by W workers, each worker will build N / W trees. During inference, predictions from all N trees will be combined.

The caller is responsible for partitioning the data efficiently via Dask. To build an accurate model, it's important to ensure that each worker has a representative chunk of the data. This can come by distributing the data evenly after ensuring that it is well shuffled. Or, given sufficient memory capacity, the caller can replicate the data to all workers. This approach will most closely simulate the single-GPU building approach.

Note: cuML 0.9 contains the first, experimental preview release of the MNMG random forest model. The API is subject to change in future releases, and some known limitations remain (listed in the documentation).

For more information on MNMG Random Forest models, see the documentation:

https://docs.rapids.ai/api/cuml/stable/api.html#cuml.dask.ensemble.RandomForestClassifier
https://docs.rapids.ai/api/cuml/stable/api.html#cuml.dask.ensemble.RandomForestRegressor

'''

import numpy as np
import sklearn

import pandas as pd
import cudf
import cuml

from sklearn import model_selection

from cuml import datasets
from cuml.metrics import accuracy_score
from cuml.dask.common import utils as dask_utils
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import dask_cudf

from cuml.dask.ensemble import RandomForestClassifier as cumlDaskRF
from sklearn.ensemble import RandomForestClassifier as sklRF

def main():
  # This will use all GPUs on the local host by default
  cluster = LocalCUDACluster(threads_per_worker=1)
  c = Client(cluster)

  # Query the client for all connected workers
  workers = c.has_what().keys()
  n_workers = len(workers)
  n_streams = 8  # Performance optimization

  '''
  Define Parameters
  In addition to the number of examples, random forest fitting performance depends heavily on the number of columns in a dataset 
  and (especially) on the maximum depth to which trees are allowed to grow. 
  Lower max_depth values can greatly speed up fitting, though going too low may reduce accuracy.
  '''

  # Data parameters
  train_size = 100000
  test_size = 1000
  n_samples = train_size + test_size
  n_features = 20

  # Random Forest building parameters
  max_depth = 12
  n_bins = 16
  n_trees = 1000

  print(f'[trace] Generate Data on host')
  X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                      n_clusters_per_class=1, n_informative=int(n_features / 3),
                                      random_state=123, n_classes=5)
  X = X.astype(np.float32)
  y = y.astype(np.int32)
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

  n_partitions = n_workers

  def distribute(X, y):
    # First convert to cudf (with real data, you would likely load in cuDF format to start)
    X_cudf = cudf.DataFrame(X)
    y_cudf = cudf.Series(y)

    # Partition with Dask
    # In this case, each worker will train on 1/n_partitions fraction of the data
    X_dask = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)
    y_dask = dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    # Persist to cache the data in active memory
    X_dask, y_dask = \
      dask_utils.persist_across_workers(c, [X_dask, y_dask], workers=workers)

    return X_dask, y_dask

  print(f'[trace] Build a scikit-learn model (single node)')
  X_train_dask, y_train_dask = distribute(X_train, y_train)
  X_test_dask, y_test_dask = distribute(X_test, y_test)
  # Use all avilable CPU cores
  skl_model = sklRF(max_depth=max_depth, n_estimators=n_trees, n_jobs=-1)
  skl_model.fit(X_train.get(), y_train.get())

  print(f'[trace] Train the distributed cuML model')
  cuml_model = cumlDaskRF(max_depth=max_depth, n_estimators=n_trees, n_bins=n_bins, n_streams=n_streams)
  cuml_model.fit(X_train_dask, y_train_dask)

  wait(cuml_model.rfs)  # Allow asynchronous training tasks to finish
  print(f'[trace] Predict and check accuracy')
  skl_y_pred = skl_model.predict(X_test.get())
  cuml_y_pred = cuml_model.predict(X_test_dask).compute().to_numpy()

  # Due to randomness in the algorithm, you may see slight variation in accuracies
  print("SKLearn accuracy:  ", accuracy_score(y_test, skl_y_pred))
  print("CuML accuracy:     ", accuracy_score(y_test, cuml_y_pred))
  pass

if __name__ == '__main__':
  main()
  pass
