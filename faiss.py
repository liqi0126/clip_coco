import faiss
import numpy as np
import time

import faiss
import numpy as np
import time

# from api import *
# h: hidden dimension
# query: encoded input (can be image/text) [#number of query, h]
# database: encoded database  [#number of samples in database, h]
# res = faiss_search(query, database)


def faiss_search(query, database, distance_metric='l2', k=10, method='exact', use_PCA=False, 
                 use_GPU=False, dim_PCA=128, 
                 n_partitions=50, n_probe=10,
                 n_centroids_pq = 8, n_bits_pq = 8
                 ):
    """ Similarity search with Faiss
    Parameters
    ----------
    distance_metric = {'l2', 'inner'}
    k: int
      number of k nearest neighbors.
    method: str
      name for search methods ['exact', 'ivf', 'ivf_probe','pq']
    dim_PCA: int
      embedding dim after PCA
    n_partitions: int
      specify how many partitions (Voronoi cells) we’d like our index to have for Inverted File Index (IVF) index
    n_probe: int
      increase the number of nearby cells to search too with n_probe for Inverted File Index (IVF) index
    n_centroids_pq: int
      number of centroid IDs in final compressed vectors for PQ (Product quantization for nearest neighbor search)
    n_bits_pq: int
      number of bits in each centroid for PQ (Product quantization for nearest neighbor search)
    Returns
    -------
    I: indexes of top-k neighbors
    time(s): time used for search
    D: raw distances between the neighbors and the queries
    """
    emb_texts = database
    emb_images = query
    d = emb_texts.shape[1]
    assert method in ['exact','ivf','ivf_probe','pq'], f"{method} method not supported"

    if use_PCA:
      # PCA reduction
      mat = faiss.PCAMatrix(d, dim_PCA)
      mat.train(emb_texts)
      emb_texts = mat.apply_py(emb_texts)
      emb_images = mat.apply_py(emb_images)
      
      d = dim_PCA

    # search_methods = ['l2', 'ivf', 'ivf_probe', 'pq']
    if distance_metric == 'l2':
      index = faiss.IndexFlatL2(d) 
    elif distance_metric == 'inner':
      index = faiss.IndexFlatIP(d) 

    if method == 'exact':
        #index = faiss.IndexFlatL2(d)   # build the index
        index.add(emb_texts)                  # add vectors to the index
        
    elif method == 'ivf':
        #quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(index, d, n_partitions)
        index.train(emb_texts)
        assert index.is_trained
        index.add(emb_texts)

    elif method == 'ivf_probe':
        
        #quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(index, d, n_partitions)
        index.train(emb_texts)
        assert index.is_trained
        index.add(emb_texts)
        # increase the number of nearby cells to search too with nprobe.
        index.nprobe = n_probe

    elif method == 'pq':
        #quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
        index = faiss.IndexIVFPQ(index, d, n_partitions, n_centroids_pq, n_bits_pq)
        index.train(emb_texts)
        index.add(emb_texts)
        index.nprobe = n_probe  # align to previous IndexIVFFlat nprobe value

    if use_GPU:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # search
    print(index.ntotal, 'number of embeddings in the database',
            'using search method:', method,',distance metric:',distance_metric)
    t1 = time.perf_counter()
    D, I = index.search(emb_images, k)
    t2 = time.perf_counter()
    # print('time taken to search:',t2-t1)
    # print(I[:3])                  # indexes of words/sentence, sorted by increasing distance
    # print(D.shape)               # floating-point matrix with the corresponding squared distances.
    return {
        "result": I,
        "time": t2 - t1,
        "raw_distance": D
    }


if __name__ == "__main__":
  # database embeddings:

  d = 512                           # embedding dimension
  dim_PCA = 128
  # nb = 100000                      # database size
  nb = 100000                     # database size
  # nq = 10000                       # number of queries
  nq = 100                       # number of queries
  np.random.seed(1234)             # make reproducible
  k = 10                        # we want to see k nearest neighbors
  emb_texts = np.random.random((nb, d)).astype('float32')
  emb_texts[:, 0] += np.arange(nb) / 1000.
      # # query embeddings:
  emb_images = np.random.random((nq, d)).astype('float32')
  emb_images[:, 0] += np.arange(nq) / 1000.

  search_methods = ['exact', 'ivf', 'ivf_probe','pq']
  distance_mrt = ['l2','inner']
  for s in search_methods:
    for dis in distance_mrt:
      res = faiss_search(emb_images, emb_texts, k=10, distance_metric=dis, method=s, use_PCA=True, use_GPU=True, dim_PCA=256)
      #print(res['time'], s, dis)
