
import faiss 
import numpy as np 


def compute_neighbor_accuracy(query_features, ref_features, query_ref_map):
    query_files, query_fvecs = np.asarray(list(query_features.keys())), np.concatenate(list(query_features.values()), axis=0).astype(np.float32)
    ref_files, ref_fvecs = np.asarray(list(ref_features.keys())), np.concatenate(list(ref_features.values()), axis=0).astype(np.float32)
    
    index = faiss.IndexFlatIP(ref_fvecs.shape[1])
    index.add(ref_fvecs)
    _, ref_neighbors = index.search(query_fvecs, 1)
    ref_predictions = ref_files[ref_neighbors.reshape(-1,)] 
    return (ref_predictions == ref_files).mean()