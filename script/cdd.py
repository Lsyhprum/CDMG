import argparse
import random
import struct
import re
import os
import numpy as np
import time
import faiss
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#######################################################
# Nearest neighbor search functions
#######################################################

def get_nearestneighbors(xq, xb, k, device, needs_exact=True, verbose=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index = faiss.IndexFlatIP(xq.shape[1])
    else:
        index = faiss.index_factory(xq.shape[1], "HNSW32")
        index.hnsw.efSearch = 64
    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    start = time.time()
    index.add(xb)
    D, I = index.search(xq, k)
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))

    return I, D

#######################################################
# Data file read
#######################################################

def xbin_write(x, fname):
    f = open(fname, "wb")
    n, d = x.shape
    np.array([n, d], dtype='uint32').tofile(f)
    x.tofile(f)

def ibin_read(fname):
    a = np.fromfile(fname, dtype='int32')
    n, d = a[0], a[1]
    return a[2:n*d+2].reshape(n, d).copy()

def fbin_read(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    n, d = a[0], a[1]
    return a.view('float32')[2:].reshape(n, d).copy()

def recall(res, gt, K):
    cnt = 0

    for i in range(res.shape[0]):
        for a in res[i][0 : K]:
            if a in gt[i][0 : K]:
                cnt += 1
    return float(cnt / (K * res.shape[0]))

def QNNQ(xb1, xb2, xq, K):
    I, _ = get_nearestneighbors(xq, xb1, K+1, 'cuda', needs_exact=True)
    I_nn, _ = get_nearestneighbors(xb2[I[:,0]], xb2, K+1, 'cuda', needs_exact=True)
    # return recall2(I_nn[:,1:], I[:,1:], K)
    return recall(I_nn[:,1:], I[:,1:], K)

#######################################################
# Main
#######################################################

if __name__ == "__main__":
    faiss.omp_set_num_threads(20)

    xb = fbin_read("/dataset/t2i/base.1M.fbin")
    xt = fbin_read('/dataset/t2i/query.learn.100K.fbin')

    anchor_num = 15

    start_time = time.time()
    I, _ = get_nearestneighbors(xb, xt, anchor_num, 'cpu', needs_exact=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"1: {execution_time:.6f} s")

    aux = np.mean(xt[I], axis=1)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"2: {execution_time:.6f} s")
    
    xb_torch = torch.from_numpy(xb)
    xb_fusion = np.array(xb_torch + aux, dtype=np.float32)

    xbin_write(xb_fusion, '/dataset/t2i/t2i.base.t100K.fusion_k15.fbin')