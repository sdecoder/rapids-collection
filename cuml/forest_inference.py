import cupy
import os

from cuml.testing.utils import array_equal
from cuml.common.import_utils import has_xgboost

from cuml.datasets import make_classification
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split


def main():

  if has_xgboost():
    import xgboost as xgb
    print(f"[trace] import xgboost")
  else:
    raise ImportError("Please install xgboost using the conda package,"
                      "e.g.: conda install -c conda-forge xgboost")

  # synthetic data size
  n_rows = 10000
  n_columns = 100
  n_categories = 2
  random_state = cupy.random.RandomState(43210)

  # fraction of data used for model training
  train_size = 0.8

  # trained model output filename
  model_path = 'xgb.model'

  # num of iterations for which xgboost is trained
  num_rounds = 100

  # maximum tree depth in each training round
  max_depth = 20

  # create the dataset
  X, y = make_classification(
    n_samples=n_rows,
    n_features=n_columns,
    n_informative=int(n_columns / 5),
    n_classes=n_categories,
    random_state=42
  )

  # convert the dataset to float32
  X = X.astype('float32')
  y = y.astype('float32')

  # split the dataset into training and validation splits
  X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8)

  def train_xgboost_model(
      X_train,
      y_train,
      model_path='xgb.model',
      num_rounds=100,
      max_depth=20
  ):
    # set the xgboost model parameters

    print(f"[trace] train the xgboost model")
    params = {
      'verbosity': 0,
      'eval_metric': 'error',
      'objective': 'binary:logistic',
      'max_depth': max_depth,
      'tree_method': 'gpu_hist'
    }

    # convert training data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # train the xgboost model
    trained_model = xgb.train(params, dtrain, num_rounds)

    # save the trained xgboost model
    trained_model.save_model(model_path)
    return trained_model

  def predict_xgboost_model(X_validation, y_validation, xgb_model):
    # predict using the xgboost model
    print(f"[trace] predict using xgboost model")
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    predictions = xgb_model.predict(dvalidation)

    # convert the predicted values from xgboost into class labels
    predictions = cupy.around(predictions)
    return predictions

  # train the xgboost model
  print(f"[trace] train the xgboost model")
  xgboost_model = train_xgboost_model(
    X_train,
    y_train,
    model_path,
    num_rounds,
    max_depth
  )

  # test the xgboost model
  print(f"[trace] test the xgboost model")
  trained_model_preds = predict_xgboost_model(
    X_validation,
    y_validation,
    xgboost_model
  )

  print(f'[trace] Loaded the saved model: Use FIL to load the saved xgboost model')
  from cuml import ForestInference
  fil_model = ForestInference.load(
    filename=model_path,
    algo='BATCH_TREE_REORG',
    output_class=True,
    threshold=0.50,
    model_type='xgboost'
  )

  # perform prediction on the model loaded from path
  fil_preds = fil_model.predict(X_validation)

  print(f"[trace]Distributed FIL with Dask")
  from dask_cuda import LocalCUDACluster
  from distributed import Client, wait, get_worker

  import dask.dataframe
  import dask.array
  import dask_cudf

  from cuml import ForestInference
  import time

  print(f'[trace] Create a LocalCUDACluster')
  cluster = LocalCUDACluster()
  client = Client(cluster)

  workers = client.has_what().keys()
  n_workers = len(workers)
  n_partitions = n_workers

  rows = 1_000_000
  cols = 100

  print(f'[trace] Generate synthetic query/inference data')
  x = dask.array.random.random(
    size=(rows, cols),
    chunks=(rows // n_partitions, cols)
  ).astype('float32')

  df = dask_cudf.from_dask_dataframe(
    dask.dataframe.from_array(x)
  )

  print(f'[trace] Persist data in GPU memory')
  df = df.persist()
  wait(df)

  print(f'[trace] Pre-load FIL model on each worker')

  def worker_init(model_file='xgb.model'):
    worker = get_worker()

    worker.data["fil_model"] = ForestInference.load(
      filename=model_file,
      algo='BATCH_TREE_REORG',
      output_class=True,
      threshold=0.50,
      model_type='xgboost'
    )

  client.run(worker_init)

  def predict(input_df):
    worker = get_worker()
    return worker.data["fil_model"].predict(input_df)

  distributed_predictions = df.map_partitions(predict, meta="float")
  tic = time.perf_counter()
  distributed_predictions.compute()
  toc = time.perf_counter()
  fil_inference_time = toc - tic
  total_samples = len(df)
  print(f' {total_samples:,} inferences in {fil_inference_time:.5f} seconds'
        f' -- {int(total_samples / fil_inference_time):,} inferences per second ')
  pass

if __name__ == '__main__':
  main()
  pass