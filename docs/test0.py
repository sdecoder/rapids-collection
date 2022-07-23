import cuml
import cupy as cp

def test0():
  ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
  ary = cp.asarray(ary)
  prev_output_type = cuml.global_settings.output_type
  cuml.set_global_output_type('cudf')
  dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
  dbscan_float.fit(ary)

  # cuML output type
  print(f'[trace] dbscan_float.labels_: {dbscan_float.labels_}')
  print(f'[trace] type: {type(dbscan_float.labels_)}')
  print(f'[trace] cuml.set_global_output_type(prev_output_type): {cuml.set_global_output_type(prev_output_type)}')

  pass

def test1():
  ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
  ary = cp.asarray(ary)
  with cuml.using_output_type('cudf'):
    dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    dbscan_float.fit(ary)

    print("cuML output inside 'with' context")
    print(dbscan_float.labels_)
    print(type(dbscan_float.labels_))

    # use cuml again outside the context manager
    dbscan_float2 = cuml.DBSCAN(eps=1.0, min_samples=1)
    dbscan_float2.fit(ary)

    # cuML default output
    print(f'[trace] dbscan_float.labels_: {dbscan_float.labels_}')
    print(f'[trace] type: {type(dbscan_float.labels_)}')

  pass

if __name__ == '__main__':

  test1()
  pass
