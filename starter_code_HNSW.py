import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    with h5py.File("sift-128-euclidean.hdf5", "r") as f:
        xb = f["train"][:]
        xq = f["test"][:]
    print(f"Loaded base vectors: {xb.shape}, query vectors: {xq.shape}")

    M = 16
    efConstruction = 200
    efSearch = 200
    d = xb.shape[1]

    print("Building HNSW index...")
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch

    index.add(xb)
    print(f"Added {index.ntotal} vectors to the index")

    k = 10
    query_vector = xq[0:1]
    D, I = index.search(query_vector, k)
    top10 = I[0].tolist()
    print("Top 10 nearest neighbors:", top10)

    with open("output.txt", "w") as f:
        for idx in top10:
            f.write(f"{idx}\n")

    print(f"Wrote top-10 neighbor indices to output.txt")

if __name__ == "__main__":
    evaluate_hnsw()
