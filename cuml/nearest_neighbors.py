import cudf
import numpy as np
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors

print(f'[trace] Define Parameters')
n_samples = 2**17
n_features = 40

n_query = 2**13
n_neighbors = 4
random_state = 0

print(f'[trace] Generate Data')
device_data, _ = make_blobs(n_samples=n_samples,
                            n_features=n_features,
                            centers=5,
                            random_state=random_state)

device_data = cudf.DataFrame(device_data)
host_data = device_data.to_pandas()

print(f'[trace] fit using Scikit-learn Model')
knn_sk = skNearestNeighbors(algorithm="brute", n_jobs=-1)
knn_sk.fit(host_data)
D_sk, I_sk = knn_sk.kneighbors(host_data[:n_query], n_neighbors)

print(f'[trace] fit using cuML Model')
knn_cuml = cuNearestNeighbors()
knn_cuml.fit(device_data)
D_cuml, I_cuml = knn_cuml.kneighbors(device_data[:n_query], n_neighbors)

print(f'[trace] compare Results')
'''
cuML currently uses FAISS for exact nearest neighbors search, which limits inputs to single-precision. 
This results in possible round-off errors when floats of different magnitude are added. 
As a result, it's very likely that the cuML results will not match Sciklearn's nearest neighbors exactly. 
You can read more in the FAISS wiki.
'''

passed = np.allclose(D_sk, D_cuml.to_numpy(), atol=1e-3)
print('compare knn: cuml vs sklearn distances %s'%('equal'if passed else 'NOT equal'))
sk_sorted = np.sort(I_sk, axis=1)
cuml_sorted = np.sort(I_cuml.to_numpy(), axis=1)

diff = sk_sorted - cuml_sorted

passed = (len(diff[diff!=0]) / n_samples) < 1e-9
print('compare knn: cuml vs sklearn indexes %s'%('equal'if passed else 'NOT equal'))
